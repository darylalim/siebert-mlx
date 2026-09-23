"""End-to-end flow tests for streamlit_app.py using Streamlit's AppTest framework.

Complement the unit tests in test_streamlit_app.py by exercising the script
top-to-bottom: widget rendering, button clicks, session_state mutations, and
the conditional reveal of the column selector after data is loaded. The weight
download is mocked at the conftest level, but AutoConfig/AutoTokenizer are not,
so these still need the hub or a warm ~/.cache/huggingface for ~1.2 MB of
config/tokenizer files -- a cold cache with no network fails at collection.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    CONFIDENCE_COL,
    LONG_TEXT_CHARS,
    SENTIMENT_COL,
    STYLE_ROW_CAP,
    TEXT_COL_WIDTH,
)

# Absolute, because AppTest.from_file resolves a *relative* path against the
# calling file (this module's tests/ directory) as of streamlit 1.61, where it
# previously resolved against the working directory. An absolute path is used
# as given under both.
APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")
TIMEOUT = 30
PREVIEW_LABEL = "**Preview**"
GET_STARTED_LABEL = "**Get started**"
RESULTS_LABEL = "**Results**"


def _new_app():
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


def _has_card(block, label):
    """Whether `block` holds a card headed by the bold markdown `label`."""
    return any(m.value == label for m in block.markdown)


def _results_rendered(at):
    """Whether the results table's card is on the page.

    The observable every "did results render" assertion here uses. It used to
    be the "Classification complete!" callout, which is now a toast fired only
    on the run that classified -- and these tests hand-seed results rather than
    click Classify, so it never appears in them at all.
    """
    return _has_card(at.main, RESULTS_LABEL)


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
    # The truncation note is in the page caption, which renders in every
    # state; the requirement is in the Get started card, which is where the
    # landing page tells the user what to do.
    assert any("512 tokens" in c.value for c in at.main.caption)
    assert any("with a text column" in m.value for m in at.main.markdown)


def test_preview_blanks_missing_cells():
    # Same rule as the results grid: the default paints a missing cell as the
    # literal word "None", and the preview shows the whole file, so every NA
    # row of blank_cells.csv lands in it.
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
    assert _results_rendered(at)
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
    assert _results_rendered(at)

    at.selectbox[0].set_value("other").run()
    assert not _results_rendered(at)


def _node_text(node):
    """Label or string value of a tree node, or None -- and never raising.

    Defensive rather than required as of streamlit 1.64.0: under 1.61.1 an
    UnknownElement's .value (the bar chart's vega_lite_chart, whose proto id is
    "") indexed session_state and raised KeyError. 1.64.0 catches that and falls
    back to a proto field, but the guard keeps the walk safe across versions.
    """
    try:
        label = getattr(node, "label", None)
    except Exception:
        label = None
    if isinstance(label, str):
        return label
    try:
        value = getattr(node, "value", None)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _top_level_index(at, text):
    """Position, among the main body's direct children, of the subtree holding
    `text` -- every label asserted on below sits inside a container."""
    for index, node in enumerate(at.main.children.values()):
        stack = [node]
        while stack:
            current = stack.pop()
            if _node_text(current) == text:
                return index
            stack.extend(getattr(current, "children", {}).values())
    return None


def _block_holding(at, text):
    """The main-area block whose *direct* children include the node reading `text`."""
    stack = [at.main]
    while stack:
        block = stack.pop()
        children = list(getattr(block, "children", {}).values())
        if any(_node_text(child) == text for child in children):
            return block
        stack.extend(children)
    return None


def test_inputs_live_in_the_sidebar_and_data_in_the_main_area():
    # The layout split. Every other lookup in this file searches the whole
    # element tree, so any of these widgets could wander back into the main
    # area -- or the preview into the sidebar -- with the rest of the suite
    # green. Asserted per block, and as the exact sidebar button order, which
    # is also the order the steps are taken in.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["great", "awful"]})
    at.session_state["source_name"] = "x"
    at.run()
    assert len(at.sidebar.file_uploader) == 1
    assert len(at.sidebar.selectbox) == 1
    assert [b.key for b in at.sidebar.button] == ["sample", "classify", "reset"]
    assert len(at.main.file_uploader) == 0
    assert len(at.main.selectbox) == 0
    assert len(at.main.button) == 0
    assert _has_card(at.main, PREVIEW_LABEL)
    assert len(at.main.dataframe) == 1


def test_landing_page_says_where_the_inputs_went():
    # With the uploader in the sidebar the main area is otherwise a title over
    # empty space, and a collapsed sidebar leaves nothing on screen saying where
    # to start. Landing page only: it would be noise beside a loaded file.
    at = _new_app().run()
    assert _has_card(at.main, GET_STARTED_LABEL)
    # Content-sized: three short lines in a card stretched across the wide
    # main area read as an empty banner.
    assert _block_holding(at, GET_STARTED_LABEL).proto.width_config.use_content
    # Nothing is loaded, so there is nothing to reset.
    assert not [b for b in at.button if b.key == "reset"]

    at.button(key="sample").click().run()
    assert not _has_card(at.main, GET_STARTED_LABEL)


def test_preview_shows_the_whole_file_with_the_selected_column_first():
    # The picker lists headers only; the values beside a wrongly auto-detected
    # column are what show which header actually holds the text. The selected
    # column leads, and its header says why.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"id": ["a1", "a2"], "note": ["x", "y"], "text": ["great", "awful"]}
    )
    at.session_state["source_name"] = "x"
    at.run()
    assert at.selectbox[0].value == "id"  # auto-detect: first text column
    preview = at.main.dataframe[0]
    assert list(preview.value.columns) == ["id", "note", "text"]
    assert list(preview.proto.column_order) == ["id", "note", "text"]

    at.selectbox[0].set_value("text").run()
    preview = at.main.dataframe[0]
    # Moved to the front, the rest keep their file order.
    assert list(preview.proto.column_order) == ["text", "id", "note"]
    # Named, since after Sample nothing else on the page says what is loaded.
    assert any(
        c.value.startswith("x — 2 rows, 3 columns.") and "`text`" in c.value
        for c in at.main.caption
    )
    config = json.loads(preview.proto.columns)
    assert config["text"]["help"] == "The column that will be classified."
    # Ordered, not pinned: a pinned column is left out of the grid's spare
    # width, which drew a single-column file as one narrow column beside an
    # empty grid.
    assert "pinned" not in config["text"]
    assert "id" not in config


def test_preview_gives_way_to_the_results_table():
    # The results table is the whole file plus the two generated columns, so
    # leaving the preview up would repeat the file directly below the frame
    # that already contains it.
    at = _classified_state(_new_app()).run()
    assert not at.exception
    assert not _has_card(at.main, PREVIEW_LABEL)
    # The results grid, and only it.
    assert len(at.dataframe) == 1


def test_chart_shares_the_metric_row_above_the_results_table():
    # One summary band -- the metric cards and the distribution chart -- then
    # the table. On its own full-width row the chart spent ~190px drawing two
    # bars, and below the table it was off the bottom of the screen.
    at = _classified_state(_new_app()).run()
    metrics = _top_level_index(at, "Total rows")
    chart = _top_level_index(at, "**Sentiment distribution**")
    results = _top_level_index(at, RESULTS_LABEL)
    assert None not in (metrics, chart, results)
    assert metrics == chart < results


def test_download_rides_in_the_results_header_row():
    # Beside the thing it downloads rather than under a table that can be
    # taller than the screen, with the source named next to the label. Every
    # other Download lookup searches the whole tree, which stays green wherever
    # in it the button lands.
    at = _classified_state(_new_app()).run()
    header = _block_holding(at, RESULTS_LABEL)
    assert header is not None
    assert [child.type for child in header.children.values()] == [
        "markdown",
        "caption",
        "space",
        "download_button",
    ]
    assert list(header.children.values())[1].value == "x"
    assert len(at.sidebar.download_button) == 0


def test_preview_survives_an_all_blank_result():
    # The one branch where "results exist" and "a results table rendered" come
    # apart: _render_results emits an info callout and no table, so keying the
    # preview off the results *guard* left the page with no tabular data
    # anywhere -- on exactly the run where seeing the column is what tells the
    # user they picked the wrong one.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame({"text": ["", "   ", None]})
    at.session_state["source_name"] = "blank"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["", "   ", None],
            SENTIMENT_COL: ["", "", ""],
            CONFIDENCE_COL: [0.0, 0.0, 0.0],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert any("No classification was performed" in i.value for i in at.info)
    assert _has_card(at.main, PREVIEW_LABEL)
    assert len(at.dataframe) == 1
    # No results card on this branch, so Download renders on its own.
    assert len(at.download_button) == 1


def test_sample_button_says_what_it_loads():
    # The label alone does not say what Sample loads, and the Get started
    # card -- the only other place that names it -- is gone once a file is
    # loaded. Same class of kwarg as the selectbox help pinned above: drop it
    # and every other test stays green while the affordance silently
    # disappears.
    at = _new_app().run()
    assert at.button(key="sample").help == (
        "Load the built-in sample CSV instead of uploading a file."
    )


def test_preview_returns_when_the_column_change_invalidates_results():
    # The preview is hidden on exactly the guard that shows the results, so it
    # comes back the moment the user is choosing a column again rather than
    # staying gone for the rest of the session.
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"text": ["great", "awful"], "other": ["a", "b"]}
    )
    at.session_state["source_name"] = "x"
    at.session_state["result_df"] = pd.DataFrame(
        {
            "text": ["great", "awful"],
            "other": ["a", "b"],
            SENTIMENT_COL: ["positive", "negative"],
            CONFIDENCE_COL: [0.99, 0.97],
        }
    )
    at.session_state["result_col"] = "text"
    at.session_state["result_generated_cols"] = (SENTIMENT_COL, CONFIDENCE_COL)
    at.run()
    assert not _has_card(at.main, PREVIEW_LABEL)

    at.selectbox[0].set_value("other").run()
    assert _has_card(at.main, PREVIEW_LABEL)


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
    assert not _results_rendered(at)

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
    # The third unusable-file state, and the one that needed Reset most: the
    # failed read clears df, so read_failed is the only arm of the state chain
    # that can offer Reset, while the error is deliberately sticky. Reset also
    # bumps uploader_key, which retires the stale _uploaded_id and takes the
    # error with it.
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
    assert _results_rendered(at)


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
    # What the abort takes is the grid: Download now renders in the Results
    # header row, ahead of it, so it survives the abort (verified by reverting
    # the cell clause: an exception, 0 dataframes, 1 download button) and its
    # count here is corroboration, not the tripwire.
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
    assert not _results_rendered(at)


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
    assert not _results_rendered(at)


def test_page_is_wide():
    # The layout the sidebar split exists for, and the one thing no element in
    # the tree can show: AppTest builds `at` from delta messages only, so the
    # page_config_changed message never reaches it. Spy on the public call
    # instead -- the script's `import streamlit as st` is this same module, so
    # the patched attribute is the one it calls.
    with patch("streamlit.set_page_config", wraps=streamlit.set_page_config) as spy:
        _new_app().run()
    spy.assert_called_once()
    assert spy.call_args.kwargs["layout"] == "wide"
    # Left at "auto": "expanded" differs only at 768px and below, where the
    # sidebar overlays the page -- including the Get started card written for
    # exactly the collapsed case.
    assert "initial_sidebar_state" not in spy.call_args.kwargs


def test_file_problems_are_reported_in_the_main_area():
    # Each unusable-file arm splits its UI: the message in the main area,
    # Reset in the sidebar. Every other alert and Reset lookup in this file
    # searches the whole tree, so either could cross over with the suite green
    # -- and a message in the sidebar leaves nothing on screen once the sidebar
    # is collapsed.
    at = _new_app().run()
    at.file_uploader[0].upload("bad.csv", b"").run()
    assert len(at.main.error) == 1
    assert not at.sidebar.error
    assert [b.key for b in at.sidebar.button] == ["sample", "reset"]
    assert not _has_card(at.main, GET_STARTED_LABEL)

    for df in [pd.DataFrame(), pd.DataFrame({"score": [1, 2, 3]})]:
        at = _new_app()
        at.session_state["df"] = df
        at.session_state["source_name"] = "unusable"
        at.run()
        assert len(at.main.warning) == 1
        assert not at.sidebar.warning
        assert [b.key for b in at.sidebar.button] == ["sample", "reset"]


def test_preview_caps_long_columns_like_the_results_table():
    # The preview shows the whole file, so a long column that is not the
    # selected one is on screen too. The selected column's entry is built
    # apart from _width_caps (it also carries the help), so it is pinned both
    # ways: capped when long, left at its natural width when short.
    long_text = "word " * 20
    assert len(long_text) > LONG_TEXT_CHARS
    at = _new_app()
    at.session_state["df"] = pd.DataFrame(
        {"id": ["a1", "a2"], "title": [long_text] * 2, "text": [long_text] * 2}
    )
    at.session_state["source_name"] = "x"
    at.run()
    assert at.selectbox[0].value == "id"  # auto-detect: short, selected
    config = json.loads(at.main.dataframe[0].proto.columns)
    assert "width" not in config["id"]
    assert config["title"]["width"] == TEXT_COL_WIDTH
    assert config["text"]["width"] == TEXT_COL_WIDTH

    at.selectbox[0].set_value("text").run()
    config = json.loads(at.main.dataframe[0].proto.columns)
    assert config["text"]["width"] == TEXT_COL_WIDTH  # long, selected
    assert config["title"]["width"] == TEXT_COL_WIDTH
    assert "id" not in config


def _descendant_types(block):
    """Element types of every node under `block`, depth-first."""
    types = []
    for child in getattr(block, "children", {}).values():
        types.append(child.type)
        types.extend(_descendant_types(child))
    return types


class _FakeModel:
    """Stands in for the checkpoint so Classify can actually be clicked.

    conftest's module-level mock returns a bare MagicMock, whose logits
    mx.softmax rejects, which is why every other results test here hand-seeds
    session_state. This one answers with real mx logits, so a click drives
    the real process_dataframe against the real tokenizer.
    """

    calls = 0

    def __init__(self, config):
        self.config = config

    def from_pretrained(self, *args, **kwargs):
        pass

    def parameters(self):
        return {}

    def __call__(self, input_ids, **kwargs):
        type(self).calls += 1
        rows = input_ids.shape[0]
        return SimpleNamespace(
            logits=mx.array([[0.0, 2.0] if i % 2 else [2.0, 0.0] for i in range(rows)])
        )


@pytest.fixture
def fake_model():
    # load_model is st.cache_resource'd, so whichever model class was patched
    # in when it first ran would otherwise be served to every later test --
    # cleared on both sides so this one cannot leak in or out.
    _FakeModel.calls = 0
    streamlit.cache_resource.clear()
    with patch("mlx_transformers.models.RobertaForSequenceClassification", _FakeModel):
        yield _FakeModel
    streamlit.cache_resource.clear()


def test_classify_click_announces_once_and_renders_in_the_main_area(fake_model):
    # The only test that clicks Classify. "Classification complete!" is a toast
    # on the run that classified, not a callout re-rendered with the results:
    # as a callout it reappeared on every later rerun, announcing a run that
    # had not happened. The progress bar is written where the classify branch
    # runs -- the main area -- and must not follow the button into the sidebar.
    at = _new_app().run()
    at.button(key="sample").click().run()
    at.button(key="classify").click().run()
    assert not at.exception
    assert fake_model.calls > 0
    assert _results_rendered(at)
    assert [t.value for t in at.toast] == ["Classification complete!"]
    assert not at.success
    # process_dataframe clears its bar with .empty() before returning, so no
    # "progress" element survives the run anywhere -- asserting its absence
    # from the sidebar passed with the bar drawn there. The cleared slot does
    # survive, as an "empty" element, and that is what shows where it was.
    assert "empty" in [child.type for child in at.main.children.values()]
    assert "empty" not in _descendant_types(at.sidebar)

    calls = fake_model.calls
    at.run()  # a plain rerun re-renders from session_state
    assert fake_model.calls == calls
    assert _results_rendered(at)
    assert not at.toast
