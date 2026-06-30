"""End-to-end flow tests for streamlit_app.py using Streamlit's AppTest framework.

Complement the unit tests in test_streamlit_app.py by exercising the script
top-to-bottom: widget rendering, button clicks, session_state mutations, and
the conditional reveal of the column selector after data is loaded. Model
loading is mocked at the conftest level so no network access is required.
"""

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = "streamlit_app.py"
TIMEOUT = 30


def _new_app():
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


def test_app_starts_without_exception():
    at = _new_app().run()
    assert not at.exception


def test_initial_render_shows_title_and_caption():
    at = _new_app().run()
    assert any(t.value == "SiEBERT MLX" for t in at.title)
    assert any("processed locally" in c.value for c in at.caption)


def test_initial_render_has_uploader_and_sample_button():
    at = _new_app().run()
    assert len(at.file_uploader) == 1
    assert at.button(key="sample").label == "Sample"


def test_no_column_selector_before_data_loaded():
    at = _new_app().run()
    assert len(at.selectbox) == 0


def test_sample_button_populates_session_state():
    at = _new_app().run()
    at.button(key="sample").click().run()
    assert "df" in at.session_state
    assert at.session_state["source_name"] == "mixed_sample"


def test_selectbox_has_expected_label_and_help_text():
    at = _new_app().run()
    at.button(key="sample").click().run()
    assert len(at.selectbox) == 1
    assert at.selectbox[0].label == "Text column"
    assert at.selectbox[0].help == (
        "Select the column containing English text for sentiment classification."
    )


def test_classify_and_reset_buttons_appear_after_sample():
    at = _new_app().run()
    at.button(key="sample").click().run()
    assert at.button(key="classify").label == "Classify"
    assert at.button(key="reset").label == "Reset"


def test_reset_button_clears_session_state():
    at = _new_app().run()
    at.button(key="sample").click().run()
    assert "df" in at.session_state
    assert "source_name" in at.session_state

    at.button(key="reset").click().run()
    assert "df" not in at.session_state
    assert "source_name" not in at.session_state


def test_empty_dataframe_shows_warning():
    at = _new_app()
    at.session_state["df"] = pd.DataFrame()
    at.run()
    assert any("no rows" in w.value for w in at.warning)
    assert len(at.selectbox) == 0


def test_no_text_columns_shows_warning():
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"score": [1, 2, 3]})
    at.run()
    assert any("No text columns" in w.value for w in at.warning)
    assert len(at.selectbox) == 0


def _classified_state(at):
    """Seed session_state as if a classification has already been run."""
    at.session_state["df"] = pd.DataFrame({"text": ["great", "awful"]})
    at.session_state["source_name"] = "x"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "Sentiment": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    return at


def test_results_persist_from_session_state_without_reclassify():
    # Results render from stored state on a plain rerun (no Classify click),
    # so a post-classify interaction never re-runs inference.
    at = _classified_state(_new_app()).run()
    assert any("Classification complete" in s.value for s in at.success)
    assert len(at.metric) == 4


def test_reset_clears_classification_results():
    # Reset clears the persisted result, not just df/source_name.
    at = _classified_state(_new_app()).run()
    at.button(key="reset").click().run()
    for key in ["df", "source_name", "result_df", "result_col"]:
        assert key not in at.session_state


def test_results_hidden_when_selected_column_changes():
    # Switching the column invalidates the displayed result until re-classify.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"text": ["great", "awful"], "other": ["a", "b"]}
    )
    at.session_state["source_name"] = "x"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "other": ["a", "b"],
            "Sentiment": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.run()
    assert any("Classification complete" in s.value for s in at.success)

    at.selectbox[0].set_value("other").run()
    assert not any("Classification complete" in s.value for s in at.success)
