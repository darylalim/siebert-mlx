# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SiEBERT MLX** (`siebert-mlx`) is a Streamlit web app that classifies sentiment in English text with SiEBERT (`siebert/sentiment-roberta-large-english`) on Apple Silicon via MLX. Users upload a CSV (or try the built-in sample data), select the text column, classify, and download results with "Sentiment" and "Confidence" columns. Guided step-by-step UI with auto-detected text columns, summary metrics, and a Reset button to start over.

## Commands

```bash
# Setup
uv sync

# Run
uv run streamlit run streamlit_app.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check .

# Test
uv run pytest                                          # all tests
uv run pytest tests/test_streamlit_app.py              # unit tests
uv run pytest tests/test_app_flow.py                   # AppTest flow tests
uv run pytest tests/test_streamlit_app.py::test_name   # single test
```

Use `ruff` for all linting and formatting. Run `uv run ruff check --fix .` to auto-fix lint issues. Use `ty` for type checking. Use `pytest` for unit testing.

## Architecture

Single-file application (`streamlit_app.py`, ~195 lines):

1. **`detect_text_column`** — returns first string-dtype column name via `next()` generator
2. **`_ensure_safetensors`** — downloads model via `snapshot_download` (prefers `model.safetensors`, falls back to `pytorch_model.bin`), converts to safetensors if needed; `torch` and `safetensors` are lazy-imported only when conversion is required
3. **`load_model`** — loads config via `AutoConfig`, constructs `RobertaForSequenceClassification`, loads weights via `from_pretrained` with `float16=True`; cached with `@st.cache_resource`; authenticates with `HF_TOKEN`
4. **`process_dataframe`** — pre-filters blanks, batches valid texts (`BATCH_SIZE=8`), tokenizes with `return_tensors="np"` and converts to `mx.array`, classifies via softmax over logits; uses `.tolist()` for batch conversion
5. **UI** — guided step-by-step flow: file upload or sample data → column auto-detect and preview → classify → summary metrics → results table → CSV download

## Key Patterns

- MLX for all inference on Apple Silicon (no device management needed)
- `hf_logging.set_verbosity_error()` suppresses expected checkpoint warnings
- Confidence via `mx.softmax(logits, axis=-1)` with `mx.max` and `mx.argmax`; `mx.eval()` before `.tolist()`; labels from `model.config.id2label`
- Empty/whitespace-only texts skipped; get sentiment `""` and confidence `0.0`
- Tokenizer uses `return_tensors="np"` converted to `mx.array`, with `truncation=True` (512 token limit) and `padding=True`
- `process_dataframe` returns a copy; input DataFrame is not mutated
- `st.session_state` persists loaded DataFrame across Streamlit reruns; `st.button` returns `True` only on the rerun immediately after a click, then `False` on subsequent reruns
- Walrus operator (`:=`) in UI guards to combine detect + check into one `elif`
- `SAMPLE_DATA_PATH` points to `samples/mixed_sample.csv` for the "Sample" button
- Uses Streamlit default theme settings (no custom `.streamlit/config.toml`)
- Dependencies managed by `uv` with lockfile (`uv.lock`); `[tool.uv] override-dependencies` unpins `streamlit` from `mlx-transformers`'s exact pin so the latest Streamlit is installed

## Tests

- `tests/conftest.py` — module-level patches for `RobertaForSequenceClassification`, `snapshot_download`, `torch.load`, and `safetensors.torch.save_file` to prevent model downloads and weight conversion during test collection
- `tests/test_streamlit_app.py` — unit tests for `detect_text_column`, `_ensure_safetensors`, `load_model`, `process_dataframe`, `BATCH_SIZE`, and `SAMPLE_DATA_PATH`; uses class-scoped `autouse` fixture for Streamlit mock in `TestProcessDataframe` and per-test decorator mocks for model loading
- `tests/test_app_flow.py` — end-to-end flow tests via `streamlit.testing.v1.AppTest`: initial render, Sample button loads CSV into `session_state`, selectbox label/help text, Classify+Reset visibility, Reset clears state; relies on `conftest.py` patches so no network access

## Sample Data

- `samples/mixed_sample.csv` — 20-row sample (4 from each domain), loaded by the "Sample" button via `SAMPLE_DATA_PATH`
- `samples/product_reviews.csv` — 40 e-commerce product reviews
- `samples/movie_reviews.csv` — 40 film and TV opinions
- `samples/social_media.csv` — 40 tweets and social media posts
- `samples/restaurant_reviews.csv` — 40 dining and food service reviews
- `samples/app_reviews.csv` — 40 mobile/web app store reviews
