"""Non-mocked tests pinning what the model actually computes.

Every other test in the suite mocks model loading, so none of them can observe
a change in the numbers coming out of the encoder. That gap is not theoretical:
mlx-transformers 0.3.0 moved the forward pass from float16 to float32 compute
without a release note -- ``get_extended_attention_mask`` stopped inheriting the
attention mask's dtype and now hardcodes ``mx.float32``, and that one float32
tensor promotes every downstream activation despite ``from_pretrained(
float16=True)``. 225 of the 227 non-blank rows in ``samples/`` changed their
rounded confidence, and the mocked suite stayed green throughout.

``test_weights_are_float16_but_logits_are_float32`` is the sharp tripwire here.
The pinned confidences cannot play that role: fp16's spacing below 1.0 is
2**-11 (~4.9e-4), so fp16 rounding never moves a 4-decimal confidence by more
than ~2.4e-4 and no choice of input makes that reliably visible. They guard a
different class of regression -- a changed checkpoint, tokenizer, softmax axis,
or id2label mapping -- all of which move confidences far more than the
tolerance.

Marked ``integration`` and deselected by default (see ``[tool.pytest
.ini_options]`` in pyproject.toml) because they load the real ~1.4 GB
checkpoint. Run them with::

    uv run pytest -m integration
"""

import mlx.core as mx  # ty: ignore[unresolved-import]
import pandas as pd
import pytest

from streamlit_app import process_dataframe

pytestmark = pytest.mark.integration

# Deliberately unambiguous inputs, so the labels are stable properties of the
# model rather than of a decision boundary. Confidences are what
# process_dataframe writes, i.e. already rounded to 4 decimals.
PINNED = [
    ("I absolutely love this product, it works perfectly.", "positive", 0.9989),
    ("This is terrible and it broke immediately.", "negative", 0.9995),
]

# Loose enough for kernel-level jitter across mlx versions and Apple Silicon
# generations, tight enough that any of the drift described above trips it.
CONFIDENCE_TOLERANCE = 1e-4


def test_weights_are_float16_but_logits_are_float32(real_model):
    """float16=True buys weight memory, not fp16 compute -- pin both halves.

    If this starts failing with float16 logits, mlx-transformers has restored
    dtype inheritance in get_extended_attention_mask and the confidences below
    will quantize; if it fails with float16 weights, from_pretrained stopped
    honouring float16=True.
    """
    model, tokenizer = real_model

    weights = model.parameters()["roberta"]["embeddings"]["word_embeddings"]["weight"]
    assert weights.dtype == mx.float16

    inputs = tokenizer(
        [text for text, _, _ in PINNED],
        return_tensors="np",
        padding=True,
        truncation=True,
    )
    logits = model(**{k: mx.array(v) for k, v in inputs.items()}).logits
    mx.eval(logits)

    assert logits.dtype == mx.float32


def test_id2label_still_maps_int_ids_to_upper_case_labels(real_model):
    """process_dataframe indexes id2label with an int from mx.argmax."""
    model, _ = real_model

    assert model.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}


def test_pinned_confidences_and_labels(real_model):
    model, tokenizer = real_model

    df = pd.DataFrame({"text": [text for text, _, _ in PINNED]})
    result = process_dataframe(df, "text", model, tokenizer)

    for (text, label, confidence), (_, row) in zip(
        PINNED, result.iterrows(), strict=True
    ):
        assert row["Sentiment"] == label, text
        assert row["Confidence"] == pytest.approx(
            confidence, abs=CONFIDENCE_TOLERANCE
        ), text
