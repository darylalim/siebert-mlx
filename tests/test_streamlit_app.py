from unittest.mock import MagicMock, patch

# See the matching note in streamlit_app.py: mlx 0.32.0 dropped its .pyi stubs.
import mlx.core as mx  # ty: ignore[unresolved-import]
import pandas as pd
import pytest

from streamlit_app import (
    BATCH_SIZE,
    CONFIDENCE_COL,
    LONG_TEXT_CHARS,
    SAMPLE_DATA_PATH,
    SENTIMENT_COL,
    STYLE_ROW_CAP,
    TEXT_COL_WIDTH,
    GeneratedColumns,
    _ensure_safetensors,
    _generated_columns,
    _is_long_text,
    _render_results,
    _unique_column_name,
    detect_text_column,
    load_model,
    process_dataframe,
)

# Module-level, not an inline default: GeneratedColumns(...) in a signature is a
# call in a default argument and trips ruff B008.
PLAIN_COLS = GeneratedColumns(SENTIMENT_COL, CONFIDENCE_COL)

# Width fixtures use a realistic review, not "good"/"bad". A 4-char column
# auto-sizes far narrower than TEXT_COL_WIDTH and is deliberately left
# uncapped, so a short fixture would assert the cap against an input that
# never gets it -- which is exactly how these tests read before the
# length-aware predicate landed.
LONG_REVIEW = (
    "This product exceeded every expectation I had and I would buy it again today"
)
SHORT_LABEL = "positive"

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


# --- _is_long_text ---


class TestIsLongText:
    """The width cap is worth spending only above `LONG_TEXT_CHARS`.

    Measured against a `width="content"` grid: one text column auto-sizes to
    281px at 48 characters and 305px at 52, so `TEXT_COL_WIDTH` (300px) is
    crossed at ~51. Below that the cap makes a column wider than it would
    have been.
    """

    def test_short_values_are_not_long(self):
        assert not _is_long_text(pd.Series(["positive", "negative"]), "label")

    def test_long_values_are_long(self):
        assert _is_long_text(pd.Series([LONG_REVIEW]), "text")

    def test_boundary_is_exclusive(self):
        # Exactly at the threshold the cap is neutral, so it is not spent.
        assert not _is_long_text(pd.Series(["x" * LONG_TEXT_CHARS]), "t")
        assert _is_long_text(pd.Series(["x" * (LONG_TEXT_CHARS + 1)]), "t")

    def test_measures_the_longest_row_not_the_average(self):
        # One long review in a column of short ones still forces the width.
        assert _is_long_text(pd.Series(["ok", "ok", LONG_REVIEW]), "text")

    def test_header_counts_toward_the_width(self):
        assert _is_long_text(pd.Series(["ok"]), "n" * (LONG_TEXT_CHARS + 1))

    def test_empty_column_is_not_long(self):
        assert not _is_long_text(pd.Series([], dtype="object"), "text")

    def test_all_missing_column_is_not_long(self):
        # .str.len().max() is NaN here; int(NaN) would raise.
        assert not _is_long_text(pd.Series([None, None], dtype="object"), "text")

    def test_non_string_column_label_does_not_raise(self):
        assert not _is_long_text(pd.Series(["ok"]), 7)


# --- generated column names ---


class TestUniqueColumnName:
    def test_returns_the_base_when_free(self):
        assert _unique_column_name("Sentiment", {"text"}) == "Sentiment"

    def test_suffixes_once_then_counts_up(self):
        taken = {"Sentiment"}
        assert _unique_column_name("Sentiment", taken) == "Sentiment (model)"
        taken.add("Sentiment (model)")
        assert _unique_column_name("Sentiment", taken) == "Sentiment (model) 2"
        taken.add("Sentiment (model) 2")
        assert _unique_column_name("Sentiment", taken) == "Sentiment (model) 3"


class TestGeneratedColumns:
    def test_uses_the_plain_names_when_free(self):
        assert _generated_columns(pd.DataFrame({"text": ["a"]})) == PLAIN_COLS

    def test_renames_the_model_column_not_the_source_column(self):
        cols = _generated_columns(pd.DataFrame({"text": ["a"], "Sentiment": ["neg"]}))
        assert cols == ("Sentiment (model)", CONFIDENCE_COL)

    def test_renames_only_the_colliding_name(self):
        cols = _generated_columns(pd.DataFrame({"text": ["a"], "Confidence": [1]}))
        assert cols == (SENTIMENT_COL, "Confidence (model)")

    def test_renames_both_when_both_collide(self):
        cols = _generated_columns(
            pd.DataFrame({"Sentiment": ["a"], "Confidence": [1], "text": ["b"]})
        )
        assert cols == ("Sentiment (model)", "Confidence (model)")

    def test_counts_up_past_a_second_collision(self):
        # A downloaded result that is re-uploaded already carries
        # "Sentiment (model)"; neither it nor the original may be displaced.
        cols = _generated_columns(
            pd.DataFrame({"Sentiment": ["a"], "Sentiment (model)": ["b"]})
        )
        assert cols.sentiment == "Sentiment (model) 2"

    def test_resolves_on_a_zero_row_frame(self):
        # df.columns is non-empty while len(df) is 0; the loop must not care.
        assert _generated_columns(pd.DataFrame({"text": []})) == PLAIN_COLS

    def test_handles_non_string_column_labels(self):
        # df.columns need not hold strings; membership must not raise.
        assert _generated_columns(pd.DataFrame({1: [], 2.5: []})) == PLAIN_COLS

    def test_is_pure(self):
        # The UI persists what process_dataframe returned rather than
        # recomputing, but purity is what makes the two agree by construction.
        df = pd.DataFrame({"text": ["a"], "Sentiment": ["neg"]})
        assert _generated_columns(df) == _generated_columns(df)

    def test_exposes_named_fields(self):
        cols = _generated_columns(pd.DataFrame({"text": ["a"]}))
        assert (cols.sentiment, cols.confidence) == (SENTIMENT_COL, CONFIDENCE_COL)


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
        # Written to a sibling temp file, then renamed into place -- so the
        # path save_file sees is a *.tmp next to the destination, not the
        # destination itself.
        (weights, written_to) = mock_save.call_args.args
        assert weights == {"weight": "data"}
        assert written_to.parent == tmp_path
        assert written_to.name.endswith(".tmp")
        assert (tmp_path / "model.safetensors").exists()
        assert list(tmp_path.glob("*.tmp")) == []

    @patch("safetensors.torch.save_file", side_effect=OSError("No space left"))
    @patch("torch.load", return_value={"weight": "data"})
    @patch("streamlit_app.snapshot_download")
    def test_failed_conversion_leaves_no_partial_checkpoint(
        self, mock_download, mock_torch_load, mock_save, tmp_path
    ):
        """A half-written model.safetensors would be accepted forever after."""
        (tmp_path / "pytorch_model.bin").touch()
        mock_download.return_value = str(tmp_path)

        with pytest.raises(OSError, match="No space left"):
            _ensure_safetensors("model/name", "token")

        assert not (tmp_path / "model.safetensors").exists()
        assert list(tmp_path.glob("*.tmp")) == []

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

    @patch.dict("os.environ", {"HF_TOKEN": ""}, clear=True)
    @patch("streamlit_app._ensure_safetensors", return_value="/fake/local/dir")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoConfig")
    @patch("streamlit_app.RobertaForSequenceClassification")
    def test_blank_token_is_normalized_to_none(
        self, mock_model_cls, mock_config_cls, mock_tok_cls, mock_ensure
    ):
        """CI passes HF_TOKEN from a secret that may not exist, arriving as ""."""
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
        result, _ = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
        )
        assert "Sentiment" in result.columns
        assert len(result) == 2

    def test_classifies_positive(self):
        df = pd.DataFrame({"text": ["great"]})
        result, _ = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "positive"

    def test_classifies_negative(self):
        df = pd.DataFrame({"text": ["terrible"]})
        result, _ = process_dataframe(
            df,
            "text",
            _make_mock_model(["negative"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "negative"

    def test_maps_labels_to_lowercase(self):
        df = pd.DataFrame({"text": ["great", "awful"]})
        result, _ = process_dataframe(
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

        result, _ = process_dataframe(df, "text", model, _make_mock_tokenizer())

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
        result, _ = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert "Sentiment" not in df.columns
        assert "Sentiment" in result.columns

    def test_preserves_a_source_sentiment_column(self):
        # THE regression: result["Sentiment"] = sentiments used to replace a
        # labeled dataset's ground truth in the results table AND in the
        # downloaded CSV. pandas accepts the assignment silently, so this was
        # data loss with no error to catch.
        df = pd.DataFrame({"text": ["great"], "Sentiment": ["ground truth"]})
        result, cols = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert cols == ("Sentiment (model)", CONFIDENCE_COL)
        assert result["Sentiment"].iloc[0] == "ground truth"
        assert result["Sentiment (model)"].iloc[0] == "positive"
        assert df.columns.tolist() == ["text", "Sentiment"]

    def test_preserves_a_source_confidence_column(self):
        df = pd.DataFrame({"text": ["great"], "Confidence": ["mine"]})
        result, cols = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert cols == (SENTIMENT_COL, "Confidence (model)")
        assert result["Confidence"].iloc[0] == "mine"
        assert 0.0 <= result["Confidence (model)"].iloc[0] <= 1.0

    def test_preserves_both_source_columns(self):
        df = pd.DataFrame(
            {"text": ["great"], "Sentiment": ["gt"], "Confidence": ["mine"]}
        )
        result, cols = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert cols == ("Sentiment (model)", "Confidence (model)")
        assert result["Sentiment"].iloc[0] == "gt"
        assert result["Confidence"].iloc[0] == "mine"

    def test_classifies_a_column_named_sentiment(self):
        # The text column IS the colliding one: its own text must survive.
        # `texts` is read at the top of the function, long before the write.
        df = pd.DataFrame({"Sentiment": ["great product"]})
        result, cols = process_dataframe(
            df,
            "Sentiment",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
        )
        assert result["Sentiment"].iloc[0] == "great product"
        assert result[cols.sentiment].iloc[0] == "positive"

    def test_appends_in_order_without_a_collision(self):
        df = pd.DataFrame({"text": ["great"]})
        result, _ = process_dataframe(
            df, "text", _make_mock_model(["positive"]), _make_mock_tokenizer()
        )
        assert result.columns.tolist() == ["text", "Sentiment", "Confidence"]

    def test_round_trip_keeps_every_earlier_run(self):
        # Re-classifying a downloaded result must append, never displace: each
        # run's columns keep the names they were given.
        df = pd.DataFrame({"text": ["great"]})
        first, first_cols = process_dataframe(
            df, "text", _make_mock_model(["positive"]), _make_mock_tokenizer()
        )
        second, second_cols = process_dataframe(
            first, "text", _make_mock_model(["negative"]), _make_mock_tokenizer()
        )
        assert first_cols == PLAIN_COLS
        assert second_cols == ("Sentiment (model)", "Confidence (model)")
        assert second["Sentiment"].iloc[0] == "positive"
        assert second["Sentiment (model)"].iloc[0] == "negative"
        assert second.columns.tolist() == [
            "text",
            "Sentiment",
            "Confidence",
            "Sentiment (model)",
            "Confidence (model)",
        ]

    def test_skips_missing_text_cells(self):
        # Regression: pandas 3.0 keeps a missing cell as float NaN after
        # astype(str), so a blank cell must be coerced and skipped (sentiment
        # "", confidence 0.0) rather than crashing on NaN.strip().
        df = pd.DataFrame({"text": ["good", None, "bad"]})
        result, _ = process_dataframe(
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
        result, _ = process_dataframe(
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
        result, cols = process_dataframe(df, "text", model, MagicMock())

        # Pins the zero-row / one-column edge the naming loop must survive.
        assert cols == PLAIN_COLS
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

        result, _ = process_dataframe(df, "text", model, _make_mock_tokenizer())
        assert result["Sentiment"].iloc[0] == "positive"

    def test_adds_confidence_column(self):
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        result, _ = process_dataframe(
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
        result, _ = process_dataframe(df, "text", model, MagicMock())

        assert len(result) == 3
        assert all(s == "" for s in result["Sentiment"])
        assert all(c == 0.0 for c in result["Confidence"])
        model.assert_not_called()
        self.mock_progress.progress.assert_called_once_with(1.0)
        self.mock_progress.empty.assert_called_once()

    def test_handles_mixed_blank_text(self):
        df = pd.DataFrame({"text": ["good product", "", "  ", "bad product"]})
        result, _ = process_dataframe(
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
        result, _ = process_dataframe(
            df,
            "text",
            _make_mock_model(
                ["positive", "negative", "positive", "negative", "positive"]
            ),
            _make_mock_tokenizer(),
        )
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0


# --- _render_results column_config ---


class TestRenderResultsColumnConfig:
    """Pins the width cap that keeps Confidence's percentage on the grid.

    Without it a free-text column is "sized to fit the cell contents" and
    pushes Confidence past the 736px content cap, clipping the percentage at
    every window width. Asserted against the kwargs we pass to st.dataframe —
    our own call, not a Streamlit proto — so it does not churn across versions.
    """

    @pytest.fixture(autouse=True)
    def _mock_st(self):
        with patch("streamlit_app.st") as mock_st:
            self.mock_st = mock_st
            yield

    def _config_for(self, df, generated_cols=PLAIN_COLS):
        _render_results(df, "sample", generated_cols)
        return self.mock_st.dataframe.call_args.kwargs["column_config"]

    def test_long_review_fixture_actually_exceeds_the_threshold(self):
        # Guards every fixture below: if LONG_TEXT_CHARS is ever raised past
        # the fixture's length, the width tests would silently stop testing
        # the cap instead of failing.
        assert len(LONG_REVIEW) > LONG_TEXT_CHARS
        assert len(SHORT_LABEL) <= LONG_TEXT_CHARS

    def test_caps_the_free_text_column(self):
        config = self._config_for(
            pd.DataFrame(
                {
                    "text": [LONG_REVIEW, LONG_REVIEW],
                    "Sentiment": ["positive", "negative"],
                    "Confidence": [0.99, 0.98],
                }
            )
        )
        assert "text" in config
        # The base Column, not TextColumn: setting a width must not also
        # declare the column's type. TextColumn(width=...) emits
        # type_config={"type": "text"} alongside the width, which is what used
        # to make capping a numeric column mean rendering it as text.
        self.mock_st.column_config.Column.assert_any_call(width=TEXT_COL_WIDTH)
        self.mock_st.column_config.TextColumn.assert_called_once()  # the pair only

    def test_leaves_short_numeric_columns_auto_sized(self):
        # Padding a 2-char id out to TEXT_COL_WIDTH would spend the budget the
        # cap exists to protect. Nothing about the dtype does this now -- the
        # id is left alone because it is *short*, exactly like a short label.
        config = self._config_for(
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "text": [LONG_REVIEW, LONG_REVIEW],
                    "Sentiment": ["positive", "negative"],
                    "Confidence": [0.99, 0.98],
                }
            )
        )
        assert "text" in config
        assert "id" not in config

    def test_leaves_short_text_columns_auto_sized(self):
        # The whole point of the length predicate: an 8-char label column
        # auto-sizes to ~67px, so capping it at 300 spends 233px of the same
        # budget the cap protects -- measured as a 206px overflow that pushed
        # Confidence off the grid on a labeled CSV. Same reasoning as `id`.
        config = self._config_for(
            pd.DataFrame(
                {
                    "text": [LONG_REVIEW, LONG_REVIEW],
                    "label": [SHORT_LABEL, SHORT_LABEL],
                    "Sentiment": ["positive", "negative"],
                    "Confidence": [0.99, 0.98],
                }
            )
        )
        assert "text" in config
        assert "label" not in config

    def test_caps_a_numeric_column_under_a_long_header(self):
        # int64/float64 values can never reach LONG_TEXT_CHARS on their own (19
        # and ~24 characters at most), so a long column *name* is the only way
        # a numeric column overflows -- and it genuinely does, auto-sizing to
        # its header. This used to be excluded by dtype and was therefore
        # unfixable; the base Column caps it without touching the type, so it
        # is now treated like any other over-wide column.
        long_name = "n" * (LONG_TEXT_CHARS + 1)
        config = self._config_for(
            pd.DataFrame(
                {
                    long_name: [1, 2],
                    "text": [LONG_REVIEW, LONG_REVIEW],
                    "Sentiment": ["positive", "negative"],
                    "Confidence": [0.99, 0.98],
                }
            )
        )
        assert long_name in config
        assert "text" in config

    def test_caps_on_a_long_header_over_short_values(self):
        # glide sizes a column to the widest of its content *and* its title,
        # so the predicate measures the header too. Values are 8 chars; the
        # name alone is what carries this column past the threshold.
        long_name = "reviewer sentiment as recorded by the annotation team"
        assert len(long_name) > LONG_TEXT_CHARS
        config = self._config_for(
            pd.DataFrame(
                {
                    long_name: [SHORT_LABEL],
                    "Sentiment": ["positive"],
                    "Confidence": [0.99],
                }
            )
        )
        assert long_name in config

    def test_generated_columns_are_not_width_capped(self):
        # The generated pair keeps its own config rather than being handed a
        # width entry, even though Sentiment is text dtype.
        config = self._config_for(
            pd.DataFrame({"Sentiment": ["positive"], "Confidence": [0.99]})
        )
        assert set(config) == {"Sentiment", "Confidence"}
        # assert_called_once_with, not assert_called_once: both entries omit
        # the positional label so the header self-syncs with the frame, and
        # only pinning the full kwargs makes a re-added literal label fail.
        self.mock_st.column_config.TextColumn.assert_called_once_with(
            help="Predicted sentiment (blank for empty or missing text)."
        )
        self.mock_st.column_config.ProgressColumn.assert_called_once_with(
            help="Model confidence in the predicted sentiment.",
            format="percent",
            min_value=0.0,
            max_value=1.0,
        )

    def test_caps_the_preserved_source_column_not_the_generated_one(self):
        # Under a collision the roles invert: the user's free-text Sentiment
        # column is what needs the cap, and the model's renamed column keeps
        # its own config. A leftover literal exclusion tuple fails the
        # `set(config)` assertion (measured: "Sentiment" drops out while
        # "Sentiment (model)" wrongly gains a width entry). The call_count
        # stays 3 through that mutation, so it is corroboration, not the
        # tripwire.
        config = self._config_for(
            pd.DataFrame(
                {
                    "text": [LONG_REVIEW],
                    # Long, so this test still isolates the *exclusion tuple*:
                    # a short source column is uncapped by the length predicate
                    # instead, which is a different mechanism with its own test
                    # (test_does_not_cap_a_short_source_sentiment_column).
                    "Sentiment": [LONG_REVIEW],
                    "Sentiment (model)": ["positive"],
                    "Confidence": [0.99],
                }
            ),
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        assert set(config) == {"text", "Sentiment", "Sentiment (model)", "Confidence"}
        # Two width caps (text, the preserved source Sentiment) and exactly one
        # TextColumn -- the model's own column. The counts stay put through the
        # literal-exclusion mutation, so they are corroboration, not the
        # tripwire; the `set(config)` assertion above is.
        assert self.mock_st.column_config.Column.call_count == 2
        self.mock_st.column_config.TextColumn.assert_called_once()
        self.mock_st.column_config.ProgressColumn.assert_called_once()

    def test_does_not_cap_a_short_source_sentiment_column(self):
        # The real labeled-CSV shape, and the one that overflowed: an 8-char
        # ground-truth column must be left auto-sized so Confidence keeps its
        # place on the grid. Measured 876/670 before, 670/670 after.
        config = self._config_for(
            pd.DataFrame(
                {
                    "text": [LONG_REVIEW],
                    "Sentiment": [SHORT_LABEL],
                    "Sentiment (model)": ["positive"],
                    "Confidence": [0.99],
                }
            ),
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        assert set(config) == {"text", "Sentiment (model)", "Confidence"}
        assert "Sentiment" not in config

    def test_keys_both_generated_columns_when_both_are_renamed(self):
        # The only case that pins `column_config[confidence_col]`: with the
        # confidence name left plain, a mutant keyed to the literal
        # "Confidence" lands on the same key and nothing fails.
        config = self._config_for(
            pd.DataFrame(
                {
                    "text": [LONG_REVIEW],
                    "Sentiment": [LONG_REVIEW],
                    "Confidence": [LONG_REVIEW],
                    "Sentiment (model)": ["positive"],
                    "Confidence (model)": [0.99],
                }
            ),
            GeneratedColumns("Sentiment (model)", "Confidence (model)"),
        )
        assert set(config) == {
            "text",
            "Sentiment",
            "Confidence",
            "Sentiment (model)",
            "Confidence (model)",
        }


# --- _render_results generated-column reads ---


class TestRenderResultsGeneratedColumns:
    """Pins that every generated-column read follows the resolved names.

    Under a collision the source column and the model's column hold the same
    kind of values, so a leftover literal binds the metrics, the tint and the
    all-blank guard to the user's data: silently wrong numbers rather than a
    visible error.
    """

    @pytest.fixture(autouse=True)
    def _mock_st(self):
        with patch("streamlit_app.st") as mock_st:
            self.mock_st = mock_st
            yield

    def test_metrics_read_the_generated_columns(self):
        # Reading the literal "Sentiment" would count the ground-truth
        # vocabulary (0 positive, 0 negative); reading the literal "Confidence"
        # would call .mean() on a text column and blow up the whole render.
        #
        # The source column is deliberately *blank in a row the model
        # classified*: that makes the `classified = result_df[... != ""]` row
        # filter load-bearing too, not just the three counters below it. With
        # matching blankness the mutant filter selects the same rows and this
        # test passes either way -- measured, so do not "simplify" the fixture.
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["", "POSITIVE"],
                    "Confidence": ["n/a", "n/a"],
                    "Sentiment (model)": ["positive", "negative"],
                    "Confidence (model)": [0.9, 0.7],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence (model)"),
        )
        values = [call.args[1] for call in self.mock_st.metric.call_args_list]
        assert values == [2, "1 (50%)", "1 (50%)", "80.0%"]

    def test_tints_the_generated_column(self):
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["gt", "gt"],
                    "Sentiment (model)": ["positive", "negative"],
                    "Confidence": [0.99, 0.97],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        html = self.mock_st.dataframe.call_args.args[0].to_html()
        assert "rgba(33, 195, 84, 0.12)" in html

    def test_does_not_tint_a_source_lookalike_column(self):
        # Only the *source* column carries tintable values here, so any tint at
        # all means the Styler subset is bound to the user's ground truth.
        # Asserted on the tint values we pass rather than on generated cell
        # ids, which would churn with pandas' HTML.
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["positive", "negative"],
                    "Sentiment (model)": ["mixed", "mixed"],
                    "Confidence": [0.99, 0.97],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        html = self.mock_st.dataframe.call_args.args[0].to_html()
        assert "background-color" not in html

    def test_all_blank_guard_reads_the_generated_column(self):
        # Ground-truth labels are non-blank while the model classified nothing;
        # reading the source column would show metrics over an empty result.
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["gt", "gt"],
                    "Sentiment (model)": ["", ""],
                    "Confidence": [0.0, 0.0],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        self.mock_st.metric.assert_not_called()

    def test_notice_names_both_renamed_columns(self):
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["gt"],
                    "Confidence": ["mine"],
                    "Sentiment (model)": ["positive"],
                    "Confidence (model)": [0.99],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence (model)"),
        )
        message = self.mock_st.info.call_args.args[0]
        assert "**Sentiment (model)**" in message
        assert "**Confidence (model)**" in message
        assert "columns named Sentiment and Confidence" in message

    def test_notice_names_only_the_column_that_moved(self):
        # The two names resolve independently, so this file renames only
        # Sentiment. Bolding an unmoved Confidence alongside it would read as
        # "both of these are new names" -- true sentence, wrong impression.
        _render_results(
            pd.DataFrame(
                {
                    "Sentiment": ["gt"],
                    "Sentiment (model)": ["positive"],
                    "Confidence": [0.99],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment (model)", "Confidence"),
        )
        message = self.mock_st.info.call_args.args[0]
        assert "a column named Sentiment," in message
        assert "**Sentiment (model)**" in message
        assert "Confidence" not in message

    def test_notice_handles_a_confidence_only_collision(self):
        # The mirror case: a source Confidence column with no source Sentiment.
        # Naming SENTIMENT_COL here would point the user at a column that is
        # not in their file at all.
        _render_results(
            pd.DataFrame(
                {
                    "Confidence": ["mine"],
                    "Sentiment": ["positive"],
                    "Confidence (model)": [0.99],
                }
            ),
            "sample",
            GeneratedColumns("Sentiment", "Confidence (model)"),
        )
        message = self.mock_st.info.call_args.args[0]
        assert "a column named Confidence," in message
        assert "**Confidence (model)**" in message
        assert "Sentiment" not in message

    def test_no_notice_without_a_collision(self):
        _render_results(
            pd.DataFrame({"Sentiment": ["positive"], "Confidence": [0.99]}),
            "sample",
            PLAIN_COLS,
        )
        self.mock_st.info.assert_not_called()
