from unittest.mock import MagicMock, patch

import mlx.core as mx
import pandas as pd
import pytest

from streamlit_app import (
    BATCH_SIZE,
    SAMPLE_DATA_PATH,
    STYLE_ROW_CAP,
    _ensure_safetensors,
    detect_text_column,
    load_model,
    process_dataframe,
)

# --- BATCH_SIZE ---


def test_batch_size_is_positive_int():
    assert isinstance(BATCH_SIZE, int)
    assert BATCH_SIZE > 0


# --- STYLE_ROW_CAP ---


def test_style_row_cap_is_positive_int():
    assert isinstance(STYLE_ROW_CAP, int)
    assert STYLE_ROW_CAP > 0


# --- SAMPLE_DATA_PATH ---


def test_sample_data_path_exists():
    assert SAMPLE_DATA_PATH.exists()
    assert SAMPLE_DATA_PATH.suffix == ".csv"


# --- detect_text_column ---


class TestDetectTextColumn:
    def test_returns_first_object_column(self):
        df = pd.DataFrame({"id": [1, 2], "review": ["good", "bad"], "score": [5, 1]})
        assert detect_text_column(df) == "review"

    def test_skips_non_object_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert detect_text_column(df) is None

    def test_returns_first_when_multiple_object_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "text": ["hi", "bye"]})
        assert detect_text_column(df) == "name"

    def test_returns_none_for_empty_dataframe(self):
        assert detect_text_column(pd.DataFrame()) is None


# --- load_model ---


class TestEnsureSafetensors:
    @patch("safetensors.torch.save_file")
    @patch("torch.load", return_value={"weight": "data"})
    @patch("streamlit_app.snapshot_download")
    def test_converts_when_safetensors_missing(
        self, mock_download, mock_torch_load, mock_save, tmp_path
    ):
        pt_path = tmp_path / "pytorch_model.bin"
        pt_path.touch()
        mock_download.return_value = str(tmp_path)

        result = _ensure_safetensors("model/name", "token")

        assert result == tmp_path
        mock_torch_load.assert_called_once_with(
            pt_path, map_location="cpu", weights_only=True
        )
        mock_save.assert_called_once_with(
            {"weight": "data"}, tmp_path / "model.safetensors"
        )

    @patch("streamlit_app.snapshot_download")
    def test_skips_conversion_when_safetensors_exists(self, mock_download, tmp_path):
        (tmp_path / "model.safetensors").touch()
        mock_download.return_value = str(tmp_path)

        result = _ensure_safetensors("model/name", "token")

        assert result == tmp_path

    @patch("streamlit_app.snapshot_download")
    def test_passes_token_to_snapshot_download(self, mock_download, tmp_path):
        (tmp_path / "model.safetensors").touch()
        mock_download.return_value = str(tmp_path)

        _ensure_safetensors("model/name", "my-token")

        mock_download.assert_called_once_with(
            repo_id="model/name",
            allow_patterns=["model.safetensors", "pytorch_model.bin", "config.json"],
            token="my-token",
        )


class TestLoadModel:
    @patch.dict("os.environ", {"HF_TOKEN": "test-token"})
    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_loads_correct_model(
        self, mock_model_cls, mock_config_cls, mock_tok_cls, mock_ensure
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model()

        mock_config_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            token="test-token",
        )
        mock_ensure.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", "test-token"
        )
        mock_model_cls.assert_called_once_with(
            mock_config_cls.from_pretrained.return_value
        )
        mock_model_cls.return_value.from_pretrained.assert_called_once_with(
            "/fake/local/dir",
            float16=True,
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token="test-token"
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_loads_without_token(
        self, mock_model_cls, mock_config_cls, mock_tok_cls, mock_ensure
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model()

        mock_config_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            token=None,
        )
        mock_ensure.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", None
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token=None
        )

    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.hf_logging")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_suppresses_hf_warnings(
        self,
        mock_model_cls,
        mock_config_cls,
        mock_tok_cls,
        mock_hf_logging,
        mock_ensure,
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model()
        mock_hf_logging.set_verbosity_error.assert_called_once()

    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_returns_model_and_tokenizer(
        self, mock_model_cls, mock_config_cls, mock_tok_cls, mock_ensure
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        load_model.clear()
        model, tokenizer = load_model()

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    @patch("streamlit_app.mx")
    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_materializes_weights_on_load_thread(
        self, mock_model_cls, mock_config_cls, mock_tok_cls, mock_ensure, mock_mx
    ):
        # Weights must be eval'd on the loading thread; otherwise the lazy
        # float16 arrays stay bound to that thread's (thread-local) MLX GPU
        # stream and a later Streamlit rerun thread fails with
        # "There is no Stream(gpu, 0) in current thread."
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model()

        mock_mx.eval.assert_called_once_with(
            mock_model_cls.return_value.parameters.return_value
        )


# --- process_dataframe ---


def _make_mock_tokenizer():
    """Create a mock tokenizer returning dict-like output for mx.array conversion."""
    return MagicMock()


def _make_mock_model(sentiments):
    """Create a mock model returning logits for the given sentiment strings."""
    model = MagicMock()
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    logits = [[0.0, 1.0] if s == "positive" else [1.0, 0.0] for s in sentiments]

    mock_output = MagicMock()
    mock_output.logits = mx.array(logits)
    model.return_value = mock_output
    return model


class TestProcessDataframe:
    @pytest.fixture(autouse=True)
    def _mock_st(self):
        with patch("streamlit_app.st") as mock_st:
            self.mock_progress = MagicMock()
            mock_st.progress.return_value = self.mock_progress
            yield

    def test_adds_sentiment_column(self):
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert "Sentiment" in result.columns
        assert len(result) == 2

    def test_classifies_positive(self):
        df = pd.DataFrame({"text": ["great"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "positive"

    def test_classifies_negative(self):
        df = pd.DataFrame({"text": ["terrible"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["negative"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "negative"

    def test_maps_labels_to_lowercase(self):
        df = pd.DataFrame({"text": ["great", "awful"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == "negative"

    def test_batching_multiple_batches(self):
        n = BATCH_SIZE + 3
        df = pd.DataFrame({"text": [f"review {i}" for i in range(n)]})

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

        batch1_output = MagicMock()
        batch1_output.logits = mx.array([[0.0, 1.0]] * BATCH_SIZE)
        batch2_output = MagicMock()
        batch2_output.logits = mx.array([[0.0, 1.0]] * 3)
        model.side_effect = [batch1_output, batch2_output]

        result = process_dataframe(df, "text", model, _make_mock_tokenizer())

        assert len(result) == n
        assert model.call_count == 2

    def test_progress_bar_reaches_completion(self):
        df = pd.DataFrame({"text": ["review"]})
        process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        last_call_arg = self.mock_progress.progress.call_args_list[-1][0][0]
        assert last_call_arg == pytest.approx(1.0)

    def test_progress_bar_cleared_after_run(self):
        # The bar is removed once classification finishes, so a lingering 100%
        # bar doesn't sit above the results.
        df = pd.DataFrame({"text": ["review"]})
        process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        self.mock_progress.empty.assert_called_once()

    def test_uses_correct_text_column(self):
        df = pd.DataFrame({"col_a": ["ignore"], "col_b": ["use this"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "col_b", _make_mock_model(["positive"]), tokenizer)
        assert "use this" in tokenizer.call_args[0][0]

    def test_tokenizer_uses_numpy_tensors(self):
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "text", _make_mock_model(["positive"]), tokenizer)
        assert tokenizer.call_args[1]["return_tensors"] == "np"

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({"text": ["review"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert "Sentiment" not in df.columns
        assert "Sentiment" in result.columns

    def test_skips_missing_text_cells(self):
        # Regression: pandas 3.0 keeps a missing cell as float NaN after
        # astype(str), so a blank cell must be coerced and skipped (sentiment
        # "", confidence 0.0) rather than crashing on NaN.strip().
        df = pd.DataFrame({"text": ["good", None, "bad"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert len(result) == 3
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == ""
        assert result["Confidence"].iloc[1] == 0.0
        assert result["Sentiment"].iloc[2] == "negative"

    def test_processes_blank_cells_sample_file(self):
        # Integration: the bundled blank_cells.csv must flow through the real
        # read -> detect -> process path without crashing, skipping its missing
        # (NaN) and whitespace-only cells while classifying the rest.
        df = pd.read_csv(SAMPLE_DATA_PATH.parent / "blank_cells.csv")
        assert detect_text_column(df) == "text"

        blank_ids = [3, 4, 8]  # 3 & 8 missing (NaN), 4 whitespace-only
        valid_count = len(df) - len(blank_ids)
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"] * valid_count),
            _make_mock_tokenizer(),
        )

        assert len(result) == len(df)
        blanks = result[result["id"].isin(blank_ids)]
        assert (blanks["Sentiment"] == "").all()
        assert (blanks["Confidence"] == 0.0).all()
        valids = result[~result["id"].isin(blank_ids)]
        assert (valids["Sentiment"] == "positive").all()
        assert (valids["Confidence"] > 0).all()

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({"text": []})
        model = MagicMock()
        result = process_dataframe(df, "text", model, MagicMock())

        assert "Sentiment" in result.columns
        assert "Confidence" in result.columns
        assert len(result) == 0
        model.assert_not_called()
        self.mock_progress.progress.assert_called_once_with(1.0)
        self.mock_progress.empty.assert_called_once()

    def test_tokenizer_called_with_truncation(self):
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "text", _make_mock_model(["positive"]), tokenizer)
        assert tokenizer.call_args[1]["truncation"] is True

    def test_tokenizer_called_with_padding(self):
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "text", _make_mock_model(["positive"]), tokenizer)
        assert tokenizer.call_args[1]["padding"] is True

    def test_uses_id2label_mapping(self):
        df = pd.DataFrame({"text": ["review"]})

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_output = MagicMock()
        mock_output.logits = mx.array([[0.0, 1.0]])
        model.return_value = mock_output

        result = process_dataframe(df, "text", model, _make_mock_tokenizer())
        assert result["Sentiment"].iloc[0] == "positive"

    def test_adds_confidence_column(self):
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert "Confidence" in result.columns
        assert len(result["Confidence"]) == 2
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0

    def test_handles_all_blank_texts(self):
        df = pd.DataFrame({"text": ["", "  ", "\t"]})
        model = MagicMock()
        result = process_dataframe(df, "text", model, MagicMock())

        assert len(result) == 3
        assert all(s == "" for s in result["Sentiment"])
        assert all(c == 0.0 for c in result["Confidence"])
        model.assert_not_called()
        self.mock_progress.progress.assert_called_once_with(1.0)
        self.mock_progress.empty.assert_called_once()

    def test_handles_mixed_blank_text(self):
        df = pd.DataFrame({"text": ["good product", "", "  ", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == ""
        assert result["Confidence"].iloc[1] == 0.0
        assert result["Sentiment"].iloc[2] == ""
        assert result["Confidence"].iloc[2] == 0.0
        assert result["Sentiment"].iloc[3] == "negative"

    def test_confidence_values_in_valid_range(self):
        df = pd.DataFrame({"text": [f"text {i}" for i in range(5)]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(
                ["positive", "negative", "positive", "negative", "positive"]
            ),
            _make_mock_tokenizer(),
        )
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0
