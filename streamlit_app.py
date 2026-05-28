import os
from pathlib import Path

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
SAMPLE_DATA_PATH = Path(__file__).parent / "samples" / "mixed_sample.csv"


def detect_text_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if df[col].dtype == "object"), None)


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
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer):
    """Classify texts in batches; returns a copy with Sentiment and Confidence columns."""
    texts = df[text_column].astype(str).tolist()
    sentiments = [""] * len(texts)
    confidences = [0.0] * len(texts)
    progress_bar = st.progress(0)

    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if not valid:
        progress_bar.progress(1.0)
    else:
        id2label = model.config.id2label
        indices, valid_texts = zip(*valid)
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

            for idx, pred, conf in zip(
                indices[start:end], preds.tolist(), max_probs.tolist()
            ):
                sentiments[idx] = id2label[pred].lower()
                confidences[idx] = round(conf, 4)

            progress_bar.progress(end / total)

    result = df.copy()
    result["Sentiment"] = sentiments
    result["Confidence"] = confidences
    return result


st.set_page_config(page_title="SiEBERT Pipeline")

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

st.title("SiEBERT Pipeline")
st.caption(
    "Classify sentiment in English text with the SiEBERT model on Apple Silicon with MLX."
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.button("Sample", key="sample")
st.caption("Your data is processed locally and never leaves your machine.")

if use_sample:
    st.session_state["df"] = pd.read_csv(SAMPLE_DATA_PATH)
    st.session_state["source_name"] = "mixed_sample"
elif uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state["source_name"] = uploaded_file.name.rsplit(".", 1)[0]
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ):
        st.error("Could not read this file. Please check it's a valid CSV.")

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
        st.dataframe(df[[text_column]].head(), width="stretch")

        col_classify, col_reset, _ = st.columns([1, 1, 6])
        with col_classify:
            classify_clicked = st.button("Classify", type="primary", key="classify")
        with col_reset:
            if st.button("Reset", key="reset"):
                for key in ["df", "source_name"]:
                    st.session_state.pop(key, None)
                st.rerun()

        if classify_clicked:
            with st.spinner("Classifying..."):
                result_df = process_dataframe(df, text_column, model, tokenizer)

            csv_data = result_df.to_csv(index=False)

            if result_df["Sentiment"].eq("").all():
                st.info(
                    "All values in this column are empty. "
                    "No classification was performed."
                )
            else:
                st.success("Classification complete!")

                total = len(result_df)
                classified = result_df[result_df["Sentiment"] != ""]
                pos_count = int((classified["Sentiment"] == "positive").sum())
                neg_count = int((classified["Sentiment"] == "negative").sum())
                avg_conf = classified["Confidence"].mean() if len(classified) else 0.0

                # total > 0 guaranteed: df.empty and all-blank branches exit above
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total rows", total)
                m2.metric("Positive", f"{pos_count} ({pos_count / total * 100:.0f}%)")
                m3.metric("Negative", f"{neg_count} ({neg_count / total * 100:.0f}%)")
                m4.metric("Avg confidence", f"{avg_conf:.1%}")

                st.dataframe(result_df, width="stretch")

            st.download_button(
                label="Download",
                data=csv_data,
                file_name=f"{source_name}_sentiment.csv",
                mime="text/csv",
                key="download",
            )
