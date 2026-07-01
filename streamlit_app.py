import os
from pathlib import Path
from typing import cast

import mlx.core as mx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from mlx_transformers.models import RobertaForSequenceClassification
from transformers import (
    AutoConfig,
    AutoTokenizer,
    logging as hf_logging,
)

load_dotenv()

BATCH_SIZE = 8
STYLE_ROW_CAP = 2000
SAMPLE_DATA_PATH = Path(__file__).parent / "samples" / "mixed_sample.csv"


def detect_text_column(df: pd.DataFrame) -> str | None:
    return next(
        (
            col
            for col in df.columns
            if pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
        ),
        None,
    )


def _ensure_safetensors(model_path: str, token: str | None) -> Path:
    """Download model and convert pytorch_model.bin to safetensors if needed."""
    local_dir = Path(
        snapshot_download(
            repo_id=model_path,
            allow_patterns=["model.safetensors", "pytorch_model.bin", "config.json"],
            token=token,
        )
    )
    safetensors_path = local_dir / "model.safetensors"
    if not safetensors_path.exists():
        import torch
        from safetensors.torch import save_file

        pt_weights = torch.load(
            local_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        save_file(pt_weights, safetensors_path)
    return local_dir


@st.cache_resource
def load_model():
    """Load model and tokenizer once via @st.cache_resource in float16."""
    model_path = "siebert/sentiment-roberta-large-english"
    token = os.environ.get("HF_TOKEN")
    hf_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(model_path, token=token)
    local_dir = _ensure_safetensors(model_path, token)
    model = RobertaForSequenceClassification(config)
    model.from_pretrained(str(local_dir), float16=True)
    # Force the (lazy) float16 weights to materialize on the thread that loads
    # the model. MLX streams are thread-local and Streamlit runs each rerun on a
    # fresh thread; without this, the cached weights stay as pending ops bound to
    # the loader thread's GPU stream, and a later rerun's mx.eval fails with
    # "There is no Stream(gpu, 0) in current thread."
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer):
    """Classify texts in batches; returns a copy with Sentiment and Confidence columns."""
    texts = df[text_column].fillna("").astype(str).tolist()
    sentiments = [""] * len(texts)
    confidences = [0.0] * len(texts)
    progress_bar = st.progress(0)

    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if not valid:
        progress_bar.progress(1.0)
    else:
        id2label = model.config.id2label
        indices, valid_texts = zip(*valid, strict=True)
        total = len(valid_texts)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            inputs = tokenizer(
                list(valid_texts[start:end]),
                return_tensors="np",
                padding=True,
                truncation=True,
            )
            inputs = {k: mx.array(v) for k, v in inputs.items()}

            probs = mx.softmax(model(**inputs).logits, axis=-1)
            max_probs = mx.max(probs, axis=-1)
            preds = mx.argmax(probs, axis=-1)
            mx.eval(max_probs, preds)

            # preds/max_probs are 1-D, so .tolist() is always a list here;
            # cast narrows mlx's `int | float | list` return type for the checker.
            batch_preds = cast(list[int], preds.tolist())
            batch_confs = cast(list[float], max_probs.tolist())
            for idx, pred, conf in zip(
                indices[start:end], batch_preds, batch_confs, strict=True
            ):
                sentiments[idx] = id2label[pred].lower()
                confidences[idx] = round(conf, 4)

            progress_bar.progress(end / total)

    progress_bar.empty()
    result = df.copy()
    result["Sentiment"] = sentiments
    result["Confidence"] = confidences
    return result


st.set_page_config(page_title="SiEBERT MLX", page_icon=":material/sentiment_satisfied:")

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

st.title("SiEBERT MLX")

st.session_state.setdefault("uploader_key", 0)


def _clear_results():
    st.session_state.pop("result_df", None)
    st.session_state.pop("result_col", None)


def _reset_uploader():
    # Forget the last upload's id and mint a fresh (empty) file_uploader widget.
    st.session_state.pop("_uploaded_id", None)
    st.session_state["uploader_key"] += 1


def _load_sample():
    st.session_state["df"] = pd.read_csv(SAMPLE_DATA_PATH)
    st.session_state["source_name"] = "mixed_sample"
    _reset_uploader()
    _clear_results()


def _reset():
    for key in ["df", "source_name"]:
        st.session_state.pop(key, None)
    _reset_uploader()
    _clear_results()


def _render_results(result_df, source_name):
    csv_data = result_df.to_csv(index=False)

    if result_df["Sentiment"].eq("").all():
        st.info(
            "All values in this column are empty. No classification was performed.",
            icon=":material/info:",
        )
    else:
        st.success("Classification complete!", icon=":material/check_circle:")

        # total > 0 guaranteed: the df.empty and all-blank branches exit before
        # here. A horizontal container (not st.columns) lets the metric cards
        # wrap on narrow screens, per Streamlit dashboard guidance.
        total = len(result_df)
        classified = result_df[result_df["Sentiment"] != ""]
        pos_count = int((classified["Sentiment"] == "positive").sum())
        neg_count = int((classified["Sentiment"] == "negative").sum())
        avg_conf = classified["Confidence"].mean() if len(classified) else 0.0

        with st.container(horizontal=True):
            st.metric("Total rows", total, border=True)
            st.metric(
                "Positive",
                f"{pos_count} ({pos_count / total * 100:.0f}%)",
                border=True,
            )
            st.metric(
                "Negative",
                f"{neg_count} ({neg_count / total * 100:.0f}%)",
                border=True,
            )
            st.metric("Avg confidence", f"{avg_conf:.1%}", border=True)

        with st.container(border=True):
            st.markdown("**Sentiment distribution**")
            dist_df = pd.DataFrame(
                {
                    "Sentiment": ["positive", "negative"],
                    "Count": [pos_count, neg_count],
                }
            )
            st.bar_chart(dist_df, x="Sentiment", y="Count", horizontal=True)

        with st.container(border=True):
            st.markdown("**Results**")
            # Styler does value-based coloring; column_config does formatting
            # (per Streamlit guidance). The tint is a subtle, theme-safe rgba so
            # it reads on light and dark themes.
            sentiment_tint = {
                "positive": "background-color: rgba(33, 195, 84, 0.12)",
                "negative": "background-color: rgba(255, 75, 75, 0.12)",
            }
            # The Styler builds a per-cell style for every row, which defeats
            # st.dataframe's virtualization; skip the (cosmetic) tint above
            # STYLE_ROW_CAP. The CSV download uses the unstyled result_df.
            display_df = result_df
            if len(result_df) <= STYLE_ROW_CAP:
                display_df = result_df.style.map(
                    lambda v: sentiment_tint.get(v, ""), subset=["Sentiment"]
                )
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Sentiment": st.column_config.TextColumn(
                        "Sentiment",
                        help="Predicted sentiment (blank for empty or missing text).",
                    ),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Model confidence in the predicted sentiment.",
                        format="percent",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )

    st.download_button(
        label="Download",
        data=csv_data,
        file_name=f"{source_name}_sentiment.csv",
        mime="text/csv",
        icon=":material/download:",
        key="download",
    )


uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    key=f"uploader_{st.session_state['uploader_key']}",
)
st.button("Sample", key="sample", icon=":material/dataset:", on_click=_load_sample)

# Load a freshly uploaded file once. Guarding on file_id stops the persisted
# uploader value from being re-read on every rerun, which would otherwise undo
# Reset and clobber a Sample selection. _uploaded_id is advanced only after a
# successful read, so a failed upload keeps re-showing its error (instead of
# vanishing on the next rerun) and never leaves the previous file's data on screen.
if uploaded_file is not None and uploaded_file.file_id != st.session_state.get(
    "_uploaded_id"
):
    try:
        new_df = pd.read_csv(uploaded_file)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ):
        # Drop any previously loaded data so the failed upload can't keep
        # presenting the old file's preview/results as if it were this one.
        for key in ["df", "source_name"]:
            st.session_state.pop(key, None)
        _clear_results()
        st.error("Could not read this file. Please check it's a valid CSV.")
    else:
        st.session_state["_uploaded_id"] = uploaded_file.file_id
        st.session_state["df"] = new_df
        st.session_state["source_name"] = uploaded_file.name.rsplit(".", 1)[0]
        _clear_results()

df = st.session_state.get("df")
source_name = st.session_state.get("source_name", "")

if df is not None:
    if df.empty:
        st.warning("This CSV has no rows. Please upload a file with data.")
    elif (default_col := detect_text_column(df)) is None:
        st.warning("No text columns detected. Please check your CSV.")
    else:
        columns = df.columns.tolist()
        text_column = st.selectbox(
            "Text column",
            options=columns,
            index=columns.index(default_col),
            help="Select the column containing English text for sentiment classification.",
        )

        st.caption("Preview of selected column")
        st.dataframe(df[[text_column]].head(), width="stretch", hide_index=True)

        # Horizontal container (not fixed-width columns) so each button is as
        # wide as its label+icon needs and neither wraps to a second line.
        with st.container(horizontal=True):
            classify_clicked = st.button(
                "Classify",
                type="primary",
                icon=":material/play_arrow:",
                key="classify",
            )
            st.button("Reset", icon=":material/refresh:", key="reset", on_click=_reset)

        if classify_clicked:
            with st.spinner("Classifying..."):
                st.session_state["result_df"] = process_dataframe(
                    df, text_column, model, tokenizer
                )
                st.session_state["result_col"] = text_column

        # Render persisted results so post-classify reruns (e.g. the Download
        # click or a theme toggle) don't collapse the view or re-run inference.
        # Invalidate when the selected column no longer matches what was run.
        result_df = st.session_state.get("result_df")
        if result_df is not None and st.session_state.get("result_col") == text_column:
            _render_results(result_df, source_name)
