"""End-to-end flow tests for streamlit_app.py using Streamlit's AppTest framework.

Complement the unit tests in test_streamlit_app.py by exercising the script
top-to-bottom: widget rendering, button clicks, session_state mutations, and
the conditional reveal of the column selector after data is loaded. The weight
download is mocked at the conftest level, but AutoConfig/AutoTokenizer are not,
so these still need the hub or a warm ~/.cache/huggingface for ~1.2 MB of
config/tokenizer files -- a cold cache with no network fails at collection.
"""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from streamlit_app import CONFIDENCE_COL, SENTIMENT_COL, STYLE_ROW_CAP

# Absolute, because AppTest.from_file resolves a *relative* path against the
# calling file (this module's tests/ directory) as of streamlit 1.61, where it
# previously resolved against the working directory. An absolute path is used
# as given under both.
APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")
TIMEOUT = 30


def _new_app():
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


def test_app_starts_without_exception():
    at = _new_app().run()
    assert not at.exception


def test_initial_render_shows_title():
    at = _new_app().run()
    assert any(t.value == "SiEBERT MLX" for t in at.title)


def test_initial_render_has_uploader_and_sample_button():
    at = _new_app().run()
    assert len(at.file_uploader) == 1
    assert at.button(key="sample").label == "Sample"


def test_landing_page_states_what_it_wants():
    # The requirements used to surface only as post-hoc rejections after the
    # user had already chosen a file, and the 512-token truncation was
    # invisible everywhere in the UI.
    at = _new_app().run()
    captions = " ".join(c.value for c in at.caption)
    assert "text column" in captions
    assert "512 tokens" in captions


def test_preview_blanks_missing_cells():
    # Same rule as the results grid: the default paints a missing cell as the
    # literal word "None", and blank_cells.csv has one inside the first five
    # rows this preview shows.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["great", None]})
    at.session_state["source_name"] = "x"
    at.run()
    # HasField, not `== ""`: an unset proto string field *defaults* to "", so
    # the bare equality passes with the fix removed and pins nothing.
    assert at.dataframe[0].proto.HasField("placeholder")
    assert at.dataframe[0].proto.placeholder == ""


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


def test_unusable_csv_still_offers_reset():
    # Reset used to live only beside Classify in the `else` arm, so a file that
    # loaded but could not be classified showed a warning with no way back --
    # the arms that most need an escape were the two that did not have one.
    for df, expected in [
        (pd.DataFrame(), "no rows"),
        (pd.DataFrame({"score": [1, 2, 3]}), "No text columns"),
    ]:
        at = _new_app()
        at.session_state["df"] = df
        at.session_state["source_name"] = "unusable"
        at.run()
        assert any(expected in w.value for w in at.warning)
        # Pinned here because nothing else in this file can see it: every other
        # alert assertion matches `.value` (== proto.body), which is blind to
        # the icon field, so the three failure states rendered bare for as long
        # as they did with all tests green. `Element.__getattr__` falls back to
        # getattr(self.proto, name), so `.icon` reads the proto directly.
        assert all(w.icon == ":material/warning:" for w in at.warning)
        assert at.button(key="reset").label == "Reset"

        at.button(key="reset").click().run()
        assert "df" not in at.session_state
        assert "source_name" not in at.session_state
        assert not at.warning


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
    # A plain tuple, which works precisely because _render_results unpacks
    # rather than doing attribute access on a GeneratedColumns.
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
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
    for key in [
        "df",
        "source_name",
        "result_df",
        "result_col",
        "result_generated_cols",
    ]:
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
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert any("Classification complete" in s.value for s in at.success)

    at.selectbox[0].set_value("other").run()
    assert not any("Classification complete" in s.value for s in at.success)


def test_upload_loads_dataframe_into_session_state():
    at = _new_app().run()
    at.file_uploader[0].upload("reviews.csv", b"text\ngreat\nawful\n").run()
    assert "df" in at.session_state
    assert at.session_state["source_name"] == "reviews"


def test_upload_then_reset_clears_data_and_stays_cleared():
    # Regression: the uploader keeps its value across reruns, and the upload
    # branch used to re-read it every run, so Reset was instantly undone. The
    # dynamic uploader key + file_id guard make Reset stick.
    at = _new_app().run()
    at.file_uploader[0].upload("reviews.csv", b"text\ngreat\nawful\n").run()
    assert "df" in at.session_state

    at.button(key="reset").click().run()
    assert "df" not in at.session_state

    at.run()  # a later rerun must NOT re-load the lingering upload
    assert "df" not in at.session_state


def test_removing_the_uploaded_file_clears_data_and_results():
    # The has-file -> no-file transition. Clicking the uploader's X makes
    # st.file_uploader return None, which fails the file_id guard, so the whole
    # upload branch was skipped and `df` was read straight back out of
    # session_state -- leaving the preview, metrics, chart, results table and a
    # Download button all describing a file the uploader reported as absent.
    at = _new_app().run()
    at.file_uploader[0].upload("reviews.csv", b"text\ngreat\nawful\n").run()
    assert "df" in at.session_state
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "Sentiment": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)

    at.file_uploader[0].clear().run()
    for key in ["df", "source_name", "result_df", "result_col"]:
        assert key not in at.session_state
    assert "result_generated_cols" not in at.session_state
    assert not any("Classification complete" in s.value for s in at.success)

    at.run()  # and it stays cleared, rather than flapping on the next rerun
    assert "df" not in at.session_state


def test_sample_survives_the_empty_uploader_it_is_loaded_alongside():
    # The removal branch is guarded on _uploaded_id, not on a bare `is None`:
    # the uploader is empty by construction on the rerun that _load_sample
    # populates df, so a bare guard would wipe the sample it just loaded.
    at = _new_app().run()
    at.button(key="sample").click().run()
    assert at.session_state["source_name"] == "mixed_sample"

    at.run()
    assert at.session_state["source_name"] == "mixed_sample"


def test_stale_upload_does_not_override_sample():
    # Regression: a lingering uploaded file must not clobber a later Sample pick
    # on subsequent reruns.
    at = _new_app().run()
    at.file_uploader[0].upload("reviews.csv", b"text\ngreat\nawful\n").run()
    assert at.session_state["source_name"] == "reviews"

    at.button(key="sample").click().run()
    assert at.session_state["source_name"] == "mixed_sample"

    at.run()
    assert at.session_state["source_name"] == "mixed_sample"


def test_sample_clears_previous_results():
    # Loading Sample over an existing result clears the stale classification.
    at = _classified_state(_new_app()).run()
    at.button(key="sample").click().run()
    assert at.session_state["source_name"] == "mixed_sample"
    assert "result_df" not in at.session_state
    assert "result_col" not in at.session_state
    assert "result_generated_cols" not in at.session_state


def test_malformed_upload_shows_error_and_clears_data():
    # A malformed upload shows the error AND clears any previously loaded data,
    # so the failed file can't keep presenting the old file's preview/results.
    at = _new_app().run()
    at.file_uploader[0].upload("good.csv", b"text\ngreat\nawful\n").run()
    assert "df" in at.session_state

    at.file_uploader[0].upload("bad.csv", b"").run()  # empty -> EmptyDataError
    assert any("Could not read" in e.value for e in at.error)
    assert "df" not in at.session_state


def test_malformed_upload_offers_reset():
    # The third unusable-file state, and the one that needed Reset most: this
    # arm clears df, so the `if df is not None:` block never runs and none of
    # its _reset_button() call sites fire, while the error is deliberately
    # sticky. Reset also bumps uploader_key, which retires the stale
    # _uploaded_id and takes the error with it.
    at = _new_app().run()
    at.file_uploader[0].upload("bad.csv", b"").run()
    assert any("Could not read" in e.value for e in at.error)
    # See the icon note in test_unusable_csv_still_offers_reset.
    assert all(e.icon == ":material/error:" for e in at.error)
    assert at.button(key="reset").label == "Reset"

    at.button(key="reset").click().run()
    assert not at.error
    assert "_uploaded_id" not in at.session_state


def test_upload_error_persists_across_reruns():
    # _uploaded_id is not advanced on a failed read, so the error re-renders on a
    # later rerun instead of silently vanishing.
    at = _new_app().run()
    at.file_uploader[0].upload("bad.csv", b"").run()
    assert any("Could not read" in e.value for e in at.error)

    at.run()  # benign rerun
    assert any("Could not read" in e.value for e in at.error)
    assert "df" not in at.session_state


def test_result_survives_plain_rerun_after_upload():
    # The file_id guard must stop the upload branch from re-reading (and calling
    # _clear_results) on a plain rerun that keeps the same uploaded file, which
    # would wipe a just-computed result. Fails if the guard is removed.
    at = _new_app().run()
    at.file_uploader[0].upload("reviews.csv", b"text\ngreat\nawful\n").run()
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "Sentiment": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)

    at.run()  # plain rerun; uploader still holds the same file
    assert "result_df" in at.session_state
    assert any("Classification complete" in s.value for s in at.success)


def test_large_result_skips_styler_without_error():
    # Above STYLE_ROW_CAP the Styler tint is skipped; the results still render.
    n = STYLE_ROW_CAP + 1
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["good"] * n})
    at.session_state["source_name"] = "big"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["good"] * n,
            "Sentiment": ["positive"] * n,
            "Confidence": [0.9] * n,
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert not at.exception
    assert len(at.metric) == 4


def test_skipped_rows_get_their_own_metric():
    # Positive and Negative divide by total, so skipped rows make the two
    # percentages fall short of 100 with nothing on the page saying why -- on
    # blank_cells.csv that reads 40% + 30%. The fifth card names the remainder.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["great", "", "awful"]})
    at.session_state["source_name"] = "x"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "", "awful"],
            "Sentiment": ["positive", "", "negative"],
            "Confidence": [0.99, 0.0, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert len(at.metric) == 5
    assert at.metric[4].label == "Skipped"
    assert at.metric[4].value == "1"


def test_no_skipped_metric_when_every_row_classified():
    # The card is conditional: the common path stays at exactly four, so no
    # file grows a permanently-zero metric.
    at = _classified_state(_new_app()).run()
    assert len(at.metric) == 4
    assert not any(m.label == "Skipped" for m in at.metric)


def test_auto_detect_reapplies_to_a_file_with_the_same_headers():
    # The selectbox is keyed to the loaded dataset. Unkeyed, its identity was a
    # hash of (label, options, index, ...), so two files with the same headers
    # AND the same detected column shared one widget and a manual override on
    # the first was silently reapplied to the second.
    at = _new_app().run()
    at.file_uploader[0].upload("a.csv", b"note,comment\nx,great\n").run()
    assert at.selectbox[0].value == "note"  # auto-detected: first text column

    at.selectbox[0].set_value("comment").run()
    assert at.selectbox[0].value == "comment"

    # Same headers, different file: auto-detect must win again.
    at.file_uploader[0].upload("b.csv", b"note,comment\ny,awful\n").run()
    assert at.selectbox[0].value == "note"


def test_wide_result_under_the_row_cap_still_renders():
    # STYLE_ROW_CAP counts rows; streamlit rejects a Styler on *cells*
    # (styler.data.size > pd.options.styler.render.max_elements, 262,144). A
    # frame under the row cap but over the cell budget therefore passed the
    # guard, got wrapped, and raised StreamlitAPIException inside st.dataframe
    # -- uncaught, so the script aborted and the user lost both the results
    # table and the Download button after paying for the classification.
    # The sibling test above covers the tall case; this is the wide one.
    rows, cols = 700, 400  # 280,000 cells, well under STYLE_ROW_CAP rows
    assert rows <= STYLE_ROW_CAP
    assert rows * (cols + 2) > int(pd.options.styler.render.max_elements)

    source = {f"c{i}": ["good"] * rows for i in range(cols)}
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(source)
    at.session_state["source_name"] = "wide"
    at.session_state["result_df"] = pd.DataFrame(
        {
            **source,
            "Sentiment": ["positive"] * rows,
            "Confidence": [0.9] * rows,
        }
    )
    at.session_state["result_col"] = "c0"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert not at.exception
    assert len(at.metric) == 4
    # The two things the abort took with it, asserted directly.
    assert len(at.dataframe) > 0
    assert len(at.download_button) == 1


def test_all_blank_result_shows_info_not_metrics():
    # An all-blank classification renders st.info, not the success/metrics path.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["", "  "]})
    at.session_state["source_name"] = "x"
    at.session_state["result_df"] = pd.DataFrame(
        {"text": ["", "  "], "Sentiment": ["", ""], "Confidence": [0.0, 0.0]}
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert any("No classification was performed" in i.value for i in at.info)
    assert len(at.metric) == 0
    assert not any("Classification complete" in s.value for s in at.success)


def test_collision_renames_the_model_column_and_metrics_follow_it():
    # End-to-end tripwire for the labeled-dataset case: the ground-truth
    # Sentiment column survives, the notice explains the renamed headers, and
    # the metrics count the model's column. Reading the literal "Sentiment"
    # would report "0 (0%)" positives against the ground-truth vocabulary.
    # Hand-seeded rather than clicking Classify because conftest's mocked model
    # returns a MagicMock that mx.softmax rejects.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"text": ["great", "awful"], "Sentiment": ["POS", "NEG"]}
    )
    at.session_state["source_name"] = "labeled"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "Sentiment": ["POS", "NEG"],
            "Sentiment (model)": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = ("Sentiment (model)", CONFIDENCE_COL)
    at.run()
    assert not at.exception
    assert any("Sentiment (model)" in i.value for i in at.info)
    assert len(at.metric) == 4
    assert at.metric[1].value == "1 (50%)"
    assert at.metric[2].value == "1 (50%)"


def test_all_blank_colliding_file_shows_both_notices():
    # The rename notice sits ABOVE the all-blank split deliberately: this file
    # classified nothing, but its download still carries the renamed headers,
    # so the user still needs the explanation. Moving the notice into the else
    # arm leaves every other test green -- this is the only thing pinning it.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"text": ["", "  "], "Sentiment": ["gt", "gt"]}
    )
    at.session_state["source_name"] = "labeled_blank"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["", "  "],
            "Sentiment": ["gt", "gt"],
            "Sentiment (model)": ["", ""],
            "Confidence": [0.0, 0.0],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = ("Sentiment (model)", CONFIDENCE_COL)
    at.run()
    assert not at.exception
    assert len(at.info) == 2
    assert any("Sentiment (model)" in i.value for i in at.info)
    assert any("No classification was performed" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_results_missing_the_generated_pair_degrade_instead_of_crashing():
    # A session that predates result_generated_cols: streamlit's file watcher
    # reruns an edited script against the session_state already in memory, so
    # the three result_* keys can come apart across a code change even though
    # they are written in one branch and popped together. Indexing the missing
    # key rendered a KeyError traceback in place of the whole page; the guard
    # degrades it to the pre-classify view instead.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["great", "awful"]})
    at.session_state["source_name"] = "stale_session"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "Sentiment": ["positive", "negative"],
            "Confidence": [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.run()
    assert not at.exception
    assert len(at.metric) == 0
    assert not any("Classification complete" in s.value for s in at.success)
