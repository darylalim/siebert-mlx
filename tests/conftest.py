"""Patch model-loading before streamlit_app is imported at collection time.

Module-level patches (not fixtures) are required because streamlit_app.py
executes load_model() at import time, which would otherwise attempt to
download model weights and connect to the network.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_model_patch = patch(
    "mlx_transformers.models.RobertaForSequenceClassification",
    new_callable=lambda: lambda *a, **kw: MagicMock(),
)
_model_patch.start()

_download_patch = patch(
    "huggingface_hub.snapshot_download",
    return_value=str(Path(__file__).parent),
)
_download_patch.start()

_torch_load_patch = patch("torch.load", return_value={})
_torch_load_patch.start()

_save_patch = patch("safetensors.torch.save_file")
_save_patch.start()

_PATCHES = (_model_patch, _download_patch, _torch_load_patch, _save_patch)


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run tests marked `integration`, which load the real ~1.4 GB checkpoint",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `integration` tests unless --integration is passed.

    Skipping rather than deselecting via `-m`: a `-m 'not integration'` in
    addopts makes a path-targeted run of the integration module collect nothing
    and exit 5, which breaks the `pytest tests/<file>` pattern the docs teach
    for every other suite. Skipped tests report a reason and exit 0.
    """
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="needs --integration (loads the real ~1.4 GB checkpoint)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="module")
def real_model():
    """Yield ``(model, tokenizer)`` from the actual checkpoint, mocks lifted.

    Undoing the patches above is not enough on its own: they are started before
    streamlit_app is imported, so streamlit_app's own module globals are bound
    to the mocks and have to be pointed back at the real objects too. The
    lazily imported torch.load/save_file need no such treatment.

    Module-scoped so an integration module pays the (~1.4 GB, cold cache) load
    once, and load_model's st.cache_resource entry is cleared on both sides so
    a MagicMock cached by another test cannot leak in, or out.
    """
    import huggingface_hub
    import mlx_transformers.models

    import streamlit_app

    # Track what actually stopped: if a stop() raises mid-loop, restarting only
    # the ones that stopped keeps the rest single-started. Leaving the loop
    # outside the try would strand the mocks off for the whole session, and
    # every later test would then attempt a live download.
    stopped = []
    try:
        for patcher in reversed(_PATCHES):
            patcher.stop()
            stopped.append(patcher)
        with patch.multiple(
            streamlit_app,
            RobertaForSequenceClassification=mlx_transformers.models.RobertaForSequenceClassification,
            snapshot_download=huggingface_hub.snapshot_download,
        ):
            streamlit_app.load_model.clear()
            try:
                yield streamlit_app.load_model()
            finally:
                streamlit_app.load_model.clear()
    finally:
        for patcher in reversed(stopped):
            patcher.start()
