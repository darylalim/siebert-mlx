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
    assert any(t.value == "SiEBERT Pipeline" for t in at.title)
    assert any("SiEBERT" in c.value for c in at.caption)


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
