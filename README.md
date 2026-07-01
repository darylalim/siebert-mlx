# SiEBERT MLX

Streamlit application for sentiment classification in English text using [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english) on Apple Silicon with MLX.

## Features

- Upload a CSV or try built-in sample data
- Auto-detects text columns with manual override
- Binary sentiment (positive/negative) with confidence scores
- Summary metric cards: total rows, positive/negative counts, average confidence
- Sentiment-distribution chart
- Styled results table with CSV download
- Results persist across interactions; one-click Reset to start over
- Light and dark themes
- Batched MLX inference in float16 on Apple Silicon
- Handles empty, whitespace-only, and malformed input

## Setup

```bash
uv sync
```

Set a [Hugging Face token](https://huggingface.co/settings/tokens) for authenticated model downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Sample Data

`samples/` contains example CSVs:

- `mixed_sample.csv` — 20-row mixed sample, loaded by the **Sample** button
- `product_reviews.csv`, `movie_reviews.csv`, `social_media.csv`, `restaurant_reviews.csv`, `app_reviews.csv` — 40 rows each, one per domain
- `blank_cells.csv` — 10-row edge-case sample with missing and whitespace-only cells in the text column

## Testing

```bash
uv run pytest                              # all tests
uv run pytest tests/test_streamlit_app.py  # unit tests
uv run pytest tests/test_app_flow.py       # AppTest flow tests
```

## Citation

If you use SiEBERT in your work, please cite the following paper:

> Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). More than a Feeling: Accuracy and Application of Sentiment Analysis. *International Journal of Research in Marketing*, 40(1), 75-87.

```bibtex
@article{hartmann2023,
  title = {More than a Feeling: Accuracy and Application of Sentiment Analysis},
  journal = {International Journal of Research in Marketing},
  volume = {40},
  number = {1},
  pages = {75-87},
  year = {2023},
  doi = {https://doi.org/10.1016/j.ijresmar.2022.05.005},
  url = {https://www.sciencedirect.com/science/article/pii/S0167811622000477},
  author = {Jochen Hartmann and Mark Heitmann and Christian Siebert and Christina Schamp},
}
```
