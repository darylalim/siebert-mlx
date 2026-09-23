import os
import stat
import uuid
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple, cast

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
# Preferred names for the two generated columns. Constants rather than inline
# literals so the collision rule, the rename notice and the tests agree on what
# "uncollided" means.
SENTIMENT_COL = "Sentiment"
CONFIDENCE_COL = "Confidence"
# The sentiment color language, defined once for the two places the app
# speaks it: the results-table tint and the distribution chart. Streamlit's
# own semantic green/red, both mid-lightness, so they stay legible against
# the built-in light background and the app's dark theme alike -- which is
# what lets them be hardcoded at all, and was already the standing argument
# for the tint. The dark side is pinned in tests/test_theme.py, against
# .streamlit/config.toml, so a change to either end is checked. The built-in
# *categorical* palette is not an option here: it adapts per mode but carries
# no positive/negative meaning, so it drew the two bars in two shades of the
# same blue.
POSITIVE_COLOR = "#21c354"
NEGATIVE_COLOR = "#ff4b4b"
# The chart wants the solid hue; the table wants a wash behind text. Same
# two colors at two strengths, so this is the only thing that differs.
TINT_ALPHA = 0.12
# Height of the distribution chart, which shares the metric row: tall enough
# for two labelled bars plus the x axis and its "Count" title, and no taller,
# so the band stays close to the height of the metric cards beside it.
CHART_HEIGHT = 120
# Name of the chart frame's color column. A constant for the same reason
# SENTIMENT_COL is one: it is the frame's key *and* the color= kwarg, so a
# typo in either builds the chart against a column that is not there. It is
# also user-visible -- st.bar_chart puts every field in the tooltip, and
# streamlit's own suppression for literal-color columns does not fire (it
# tests `getattr(color_enc, 'legend', True) is not None`, and altair returns
# a _PropertySetter there rather than the None that was set), so hovering a
# bar shows this name beside its hex. Hence "Color" and not "_color".
CHART_COLOR_COL = "Color"
# Pixel cap for long columns in the preview and results tables. Sized against
# the narrow end of the wide layout, where the grid is the window less ~494px
# with the sidebar open: 606px at a 1100px window. The generated pair takes
# 228px (Sentiment 78 + Confidence 150), so 300 keeps the sample (528px) and
# blank_cells.csv (578px) on the grid there, while 400 overflowed both
# (628/606, 678/606) and no cap -- glide's own 500px auto-size maximum --
# overflowed further. On a wide window the cap costs nothing: the free-text
# columns take the grid's spare width on top of it. Past two-ish text columns
# nothing fixed can prevent a horizontal scroll, and that is the correct
# outcome.
TEXT_COL_WIDTH = 300
# Longest cell (or header) a column can hold before the cap is worth spending.
# Measured, not guessed: rendering one `width="content"` text column at a range
# of lengths puts the natural width at 281px for 48 characters and 305px for
# 52, so auto-sizing crosses TEXT_COL_WIDTH at ~51. Below that the cap makes a
# column *wider* than it would have been and spends the very budget it exists
# to protect -- the same reason numeric columns are left alone.
LONG_TEXT_CHARS = 50
SAMPLE_DATA_PATH = Path(__file__).parent / "samples" / "mixed_sample.csv"


def _is_text_dtype(series: pd.Series) -> bool:
    """String- or object-dtype, matching the pandas 3.0 default `str` dtype."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _tint(hex_color: str) -> str:
    """CSS background-color for a sentiment cell, from the shared hex.

    Derived rather than written out a second time: the chart needs the solid
    hex and the table needs that same hue at TINT_ALPHA, and two independent
    literals would drift into meaning different greens.
    """
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"background-color: rgba({r}, {g}, {b}, {TINT_ALPHA})"


def _count(n: int, noun: str) -> str:
    """`n` with thousands separators and `noun`, pluralized: "1 row", "1,200 rows"."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def detect_text_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if _is_text_dtype(df[col])), None)


def _is_long_text(series: pd.Series, name: object) -> bool:
    """True when this column would auto-size wider than `TEXT_COL_WIDTH`.

    The header counts: glide sizes a column to the widest of its content *and*
    its title, so a short-valued column under a long name is still wide.
    All-NA and empty columns have no length at all and are never long.
    """
    longest = series.astype(str).str.len().max()
    longest = 0 if pd.isna(longest) else int(longest)
    return max(len(str(name)), longest) > LONG_TEXT_CHARS


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
        # Convert through a sibling temp file and os.replace (atomic on POSIX,
        # same directory so it never crosses a filesystem). Writing straight to
        # model.safetensors means an interrupted conversion -- Ctrl-C, OOM, a
        # full disk -- leaves a truncated 1.4 GB file that the exists() check
        # above then accepts forever, so every later load fails on the corrupt
        # header until somebody deletes it by hand. The unique temp name also
        # keeps two concurrent converters (the app and `pytest --integration`)
        # off each other's partial writes; whichever replaces last wins, and
        # both files were complete. The finally clears the temp file on every
        # failure path and is a no-op once replace has consumed it.
        #
        # Created by hand with 0o666 rather than via mkstemp (always 0600): a
        # plain create, so the kernel applies the umask, and the resulting mode
        # is what the checkpoint should end up with -- read back from the file
        # rather than via os.umask(), which would briefly change it for every
        # thread in the process. O_EXCL keeps mkstemp's guarantee that the name
        # is ours alone.
        tmp_path = local_dir / f"model.safetensors.{uuid.uuid4().hex}.tmp"
        os.close(os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        try:
            # Inside the try, so the finally covers it too. The create stays
            # outside: if O_EXCL fails, the path is not ours to unlink.
            mode = stat.S_IMODE(tmp_path.stat().st_mode)
            save_file(pt_weights, tmp_path)
            # safetensors 0.8.0 writes through its own temp file and renames it
            # over ours, landing 0600 whatever the umask (0.7.0 kept the mode),
            # so reapply it -- otherwise os.replace carries an owner-only
            # checkpoint into a Hugging Face cache that may be shared.
            os.chmod(tmp_path, mode)
            os.replace(tmp_path, safetensors_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return local_dir


# The spinner lives on the decorator rather than in a `with st.spinner(...)`
# around the call site. `show_spinner` defaults to True, so wrapping the call
# stacked two spinners on a cache miss -- ours plus streamlit's own "Running
# `load_model()`.", which leaks an internal symbol onto the one screen the user
# is stuck staring at. Passing the text here replaces that default instead of
# competing with it. `show_time` because a cold cache is a ~1.4 GB download plus
# the safetensors conversion: a static spinner for minutes is indistinguishable
# from a hang, and an elapsed counter is the cheapest way to say "still working".
@st.cache_resource(show_spinner="Loading model...", show_time=True)
def load_model():
    """Load model and tokenizer once via @st.cache_resource in float16."""
    model_path = "siebert/sentiment-roberta-large-english"
    # `or None` because CI passes HF_TOKEN through from a secret that may not
    # exist, which arrives as "" rather than unset; huggingface_hub wants None
    # for an anonymous download, not an empty string.
    token = os.environ.get("HF_TOKEN") or None
    hf_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(model_path, token=token)
    local_dir = _ensure_safetensors(model_path, token)
    model = RobertaForSequenceClassification(config)
    model.from_pretrained(str(local_dir), float16=True)
    # Force the (lazy) float16 weights to materialize on the thread that loads
    # the model. MLX streams are thread-local and Streamlit runs each rerun on a
    # fresh thread; without this, the cached weights stay as pending ops bound to
    # the loader thread's streams, and a later rerun's mx.eval fails with
    # "There is no Stream(cpu, 1) in current thread." (mlx 0.32; 0.31 reported
    # Stream(gpu, 0) for the same bug).
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


class GeneratedColumns(NamedTuple):
    """Names of the two columns `process_dataframe` appends to its result."""

    sentiment: str
    confidence: str


def _model_namespace(base: str) -> str:
    """The prefix this app reserves for `base`'s generated column."""
    return f"{base} (model)"


def _namespace_members(columns: Collection[object], base: str) -> list[str]:
    """Source columns already occupying `base`'s reserved name space.

    Both `base` itself and anything under the `(model)` prefix. Non-string
    labels are legal in a DataFrame and can only ever match `base` exactly.
    """
    prefix = _model_namespace(base)
    return [
        str(col)
        for col in columns
        if col == base or (isinstance(col, str) and col.startswith(prefix))
    ]


def _unique_column_name(base: str, taken: Collection[object]) -> str:
    """`base`, or `base (model)` / `base (model) N` when the name space is spoken for.

    "Spoken for" is the whole `(model)` name space, not just `base` itself. A
    source column already called `Sentiment (model)` holds the user's data under
    the header this app uses to mean "the model's output", so handing the
    prediction a bare `Sentiment` would invert that convention -- and silently,
    because `Sentiment` was free, so nothing would register as a collision and
    no notice would fire. Reachable from a downloaded result whose ground-truth
    column was dropped before re-uploading.
    """
    if not _namespace_members(taken, base):
        return base
    # "(model)" rather than a bare "_1": the header is the only explanation of
    # the rename that travels with the downloaded CSV, and "_1" reads as a
    # duplicate of the user's column rather than as the model's output. Bare
    # "(model)" first and only then a counter, because a single collision is
    # the case that actually happens and it should read as a name.
    candidate = _model_namespace(base)
    n = 2
    while candidate in taken:
        candidate = f"{_model_namespace(base)} {n}"
        n += 1
    return candidate


def _generated_columns(df: pd.DataFrame) -> GeneratedColumns:
    """The one definition of what `process_dataframe`'s two new columns are called.

    ("Sentiment", "Confidence") unless the source frame already uses those
    names, in which case the *model's* column is renamed and the user's keeps
    its name, its data and its position. That direction is the whole fix: the
    result is the input frame **plus** two columns, never minus or renamed, so
    a script reading `Sentiment` out of the download still gets the file's own
    data instead of silently getting predictions.

    Resolved against the *input* frame, never the half-built result. `taken` is
    iterated rather than only probed, because "is this name free?" is a question
    about the whole `(model)` name space and not just one string.
    """
    taken: set[object] = set(df.columns)
    sentiment = _unique_column_name(SENTIMENT_COL, taken)
    # `| {sentiment}` is belt-and-braces, not load-bearing: every name this
    # returns is prefixed by its own base, and the two bases differ, so the
    # pair can never collide with each other (confirmed exhaustively over all
    # 256 subsets of the plausible name space -- the guard changed the answer
    # zero times). Kept so a third generated column could not reintroduce the
    # problem, but do not mistake it for the thing keeping them distinct.
    confidence = _unique_column_name(CONFIDENCE_COL, taken | {sentiment})
    return GeneratedColumns(sentiment, confidence)


def process_dataframe(df, text_column, model, tokenizer):
    """Classify texts in batches; returns (result copy, names of its two new columns).

    The result is the input frame plus two columns, named by
    `_generated_columns`: "Sentiment"/"Confidence" unless the source frame
    already uses those names. Callers must render and persist the returned
    names rather than assuming the literals.
    """
    texts = df[text_column].fillna("").astype(str).tolist()
    sentiments = [""] * len(texts)
    confidences = [0.0] * len(texts)
    progress_bar = st.progress(0)

    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    # Group similar lengths together before batching. Each batch is tokenized
    # with padding=True, i.e. padded to the longest sequence *in that batch*, so
    # in file order a single long review drags the other seven rows of its batch
    # through all 24 encoder layers at its width -- attention being quadratic in
    # that width. Sorting confines the long rows to their own batches instead of
    # spreading their cost across every batch that happens to contain one.
    # Character length is a free proxy for token length; the exact order does
    # not matter, only that neighbours are similar. Row order in the returned
    # frame is untouched: `indices` below still carries the original positions
    # and the write-back is by `idx`, not by batch position.
    valid.sort(key=lambda pair: len(pair[1]))

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

            # return_dict=True explicitly: left as None, mlx-transformers'
            # RobertaForSequenceClassification.__call__ falls back to
            # config.use_return_dict, a property transformers has deprecated (it
            # logs a warning that only set_verbosity_error was hiding) and is
            # pruning deprecations like it. Removal would be an AttributeError on
            # the first classify that the mocked suite cannot see. Same logits.
            probs = mx.softmax(model(**inputs, return_dict=True).logits, axis=-1)
            max_probs = mx.max(probs, axis=-1)
            preds = mx.argmax(probs, axis=-1)
            mx.eval(max_probs, preds)

            # preds/max_probs are 1-D, so .tolist() is always a list here; cast
            # narrows mlx's `list_or_scalar` (a scalar or nested list) for the checker.
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
    # Never assign the literal names: a labeled CSV that already carries a
    # ground-truth Sentiment column would have it silently replaced, on screen
    # and in the downloaded file. Resolved against `df` -- the source, before
    # the generated columns exist -- so both writes are strictly additive.
    cols = _generated_columns(df)
    result[cols.sentiment] = sentiments
    result[cols.confidence] = confidences
    return result, cols


# Wide, with the inputs in the sidebar: the main area is left to the data --
# the preview before a run, the metrics and results table after it -- and the
# table is the one element on the page that can always use more width.
#
# The sidebar state is left at the default "auto" (expanded on desktop,
# collapsed at 768px and below). "expanded" differs only on those narrow
# viewports, where the sidebar overlays the main area instead of pushing it
# aside, so it would open a phone on an overlay covering the Get started card
# -- the one element written for a collapsed sidebar. Either way, below 768px
# the sidebar stays open over the results of whatever was just clicked in it
# until the user closes it; that is the cost of a sidebar layout on a phone.
st.set_page_config(
    page_title="SiEBERT MLX",
    page_icon=":material/sentiment_satisfied:",
    layout="wide",
)

st.title("SiEBERT MLX")

st.session_state.setdefault("uploader_key", 0)


def _clear_results():
    # All three keys are written together by the classify branch and must die
    # together: result_generated_cols names columns *inside* result_df, so a
    # survivor would be applied to the next file's results, tinting and sizing
    # a column that is not the model's output.
    st.session_state.pop("result_df", None)
    st.session_state.pop("result_col", None)
    st.session_state.pop("result_generated_cols", None)


def _reset_uploader():
    # Forget the last upload's id and mint a fresh (empty) file_uploader widget.
    st.session_state.pop("_uploaded_id", None)
    st.session_state["uploader_key"] += 1


def _load_sample():
    st.session_state["df"] = pd.read_csv(SAMPLE_DATA_PATH)
    st.session_state["source_name"] = "mixed_sample"
    _reset_uploader()
    _clear_results()


def _clear_data():
    """Forget the loaded file and anything derived from it.

    The loaded-file keys and the results always die together -- results
    describe a specific `df`, so a survivor would be applied to the next file.
    One helper because three call sites need the rule: Reset, the failed-read
    arm, and the uploader's has-file -> no-file arm. It was written out
    longhand in all three, so adding a fourth loaded-file key meant remembering
    three unrelated places, which is the failure mode `_clear_results` already
    exists to prevent for `result_generated_cols`.
    """
    for key in ["df", "source_name"]:
        st.session_state.pop(key, None)
    _clear_results()


def _reset():
    _clear_data()
    _reset_uploader()


def _reset_button():
    """The one Reset button, rendered by whichever arm is live.

    Extracted so the two warning arms can offer it too. It used to exist only
    alongside Classify in the `else` arm, so a CSV with no rows or no text
    column showed a dead-end warning: the only ways out were Sample or the
    uploader's X, and the X did not clear state at all until that was fixed.
    One `key` across every call site is safe: every call sits in a different
    arm of the one if/elif chain over the loaded-file state at the foot of the
    script, so exactly one renders per run.
    """
    st.button("Reset", icon=":material/refresh:", key="reset", on_click=_reset)


def _download_button(result_df, source_name):
    # Serialize lazily: the callable runs only when Download is clicked, not on
    # every rerun that keeps results on screen. Built from the unstyled
    # result_df, so styling never reaches the file.
    #
    # on_click="ignore" then drops the rerun that the click itself used to
    # cause (the default is "rerun"), which re-executed the whole script to
    # redraw a page that had not changed: the notice pass, the metric
    # aggregations, the per-cell Styler over every row up to STYLE_ROW_CAP, and
    # _is_long_text's astype(str).str.len() sweep over every column, all to
    # hand over a file. The two settle different halves and do not conflict --
    # marshall_file routes a callable to the deferred-file path, which the
    # click resolves with one backend round trip rather than a rerun, so it
    # still runs with no rerun to attach to. Every rerun that does happen (a
    # widget, a manual Rerun, the file watcher) still re-renders from
    # session_state exactly as before, which is why the rename notice must stay
    # in _render_results rather than in process_dataframe. (A theme toggle is
    # not one: it restyles the page without rerunning the script.)
    st.download_button(
        label="Download",
        data=lambda: result_df.to_csv(index=False),
        file_name=f"{source_name}_sentiment.csv",
        mime="text/csv",
        icon=":material/download:",
        key="download",
        on_click="ignore",
    )


def _width_caps(frame: pd.DataFrame, exclude: Collection[object] = ()) -> dict:
    """column_config entries capping each long column of `frame` at TEXT_COL_WIDTH.

    Shared by the preview and the results table, so the two grids agree on
    which columns get the cap; the rule itself is explained at the results
    call site. `exclude` names columns whose config the caller assigns itself.
    """
    return {
        col: st.column_config.Column(width=TEXT_COL_WIDTH)
        for col in frame.columns
        if col not in exclude and _is_long_text(frame[col], col)
    }


def _render_results(result_df, source_name, generated_cols, announce=False) -> bool:
    # Unpacked once rather than read as generated_cols.sentiment at each of the
    # nine sites below: the lookups then read as plain column names, a near
    # literal swap of the hardcoded strings they replace, and any (sentiment,
    # confidence) 2-tuple works, which is what lets the flow tests hand-seed a
    # plain tuple. No default value -- a default would silently reinstate the
    # literals on exactly the input this parameter exists for.
    sentiment_col, confidence_col = generated_cols

    # Say so when the source CSV forced a rename. Here rather than in
    # process_dataframe because results re-render from session_state on every
    # rerun: a notice emitted during classify would vanish on the first one
    # (a manual Rerun, or the file watcher's after a code edit) -- while the
    # user still needs to know what the file's headers mean. Above the all-blank
    # split so it shows on that branch too, whose download carries the same
    # headers. st.info, not st.warning: nothing failed and no data was lost.
    #
    # Built from the columns that actually moved, not from the pair: the two
    # names resolve independently, so a CSV carrying only Sentiment renames
    # only Sentiment. Naming both there would be true but misread -- a bolded
    # Confidence next to a bolded Sentiment (model) reads as "both of these
    # are new names", when Confidence is exactly what it always was.
    renamed = [
        (base, resolved)
        for base, resolved in (
            (SENTIMENT_COL, sentiment_col),
            (CONFIDENCE_COL, confidence_col),
        )
        if resolved != base
    ]
    if renamed:
        # Name the user's columns that actually stood in the way, read off
        # result_df, rather than the base constants. They coincide in the common
        # case, but a file carrying only `Sentiment (model)` renames without
        # holding any column called `Sentiment` -- naming the constant there
        # would point at a column that is not in the file at all. Never empty
        # when `renamed` is: a rename means the name space was occupied, and
        # result_df is the source frame plus the generated columns.
        blocking = [
            col
            for base, resolved in renamed
            for col in _namespace_members(result_df.columns, base)
            if col != resolved
        ]
        noun = "a column" if len(blocking) == 1 else "columns"
        taken = " and ".join(blocking)
        added = " and ".join(f"**{resolved}**" for _, resolved in renamed)
        st.info(
            # "including in the download" rather than "on screen and in the
            # download": this notice also renders on the all-blank branch,
            # which skips the results table entirely, so there is no "on
            # screen" to be unchanged there. The download button renders in
            # both arms of that split, so that half stays true on both.
            f"This file already has {noun} named {taken}, so the model's "
            f"output was added as {added}. Your original columns are "
            "unchanged, including in the download.",
            icon=":material/info:",
        )

    # Returned so the caller can tell "results rendered" from "a results *table*
    # rendered". They are not the same thing on the branch below, and reading
    # the guard instead of the outcome is what briefly left an all-blank file
    # with no tabular data on screen at all: results existed, so the preview was
    # suppressed, while this arm draws an info callout and no table.
    if result_df[sentiment_col].eq("").all():
        st.info(
            "All values in this column are empty. No classification was performed.",
            icon=":material/info:",
        )
        # Standalone here: this arm has no results card to carry it, and the
        # file is still worth having -- the source plus the two columns.
        _download_button(result_df, source_name)
        drew_results_table = False
    else:
        drew_results_table = True
        # A toast on the run that classified, not a persistent st.success.
        # This function re-renders on every rerun that keeps results on screen,
        # so a callout here announced "Classification complete!" again after
        # any unrelated rerun -- switching the column away and back included,
        # with nothing classified -- and spent a 75px band above the table on
        # saying so. The metric row is the lasting confirmation. `announce` is
        # the caller's classify_clicked; the default is the re-render. Here in
        # the non-blank arm, so an all-blank run is never called complete.
        if announce:
            st.toast("Classification complete!", icon=":material/check_circle:")

        # total > 0 guaranteed: the df.empty and all-blank branches exit before
        # here. A horizontal container (not st.columns) lets the metric cards
        # wrap on narrow screens, per Streamlit dashboard guidance.
        total = len(result_df)
        classified = result_df[result_df[sentiment_col] != ""]
        pos_count = int((classified[sentiment_col] == "positive").sum())
        neg_count = int((classified[sentiment_col] == "negative").sum())
        avg_conf = classified[confidence_col].mean() if len(classified) else 0.0
        # Rows process_dataframe skipped: blank, whitespace-only or missing.
        # They are inside `total`, which is what makes the two percentages
        # below fall short of 100 -- on samples/blank_cells.csv they read 40%
        # and 30% with nothing on the page accounting for the other 30%. Named
        # rather than removed from the denominator: "Total rows" should keep
        # meaning rows in the file, and an average confidence over rows the
        # model never scored would be meaningless, so `classified` stays the
        # only defensible denominator for avg_conf.
        skipped = total - len(classified)

        # One summary band: the metric cards at their content width, then the
        # distribution chart in a card that takes the rest of the row. A
        # horizontal container rather than st.columns, so the band wraps on a
        # narrow main area -- the chart drops to its own row under the cards --
        # instead of squeezing five cards into fixed fractions. Content-width
        # cards because stretched ones split the row unevenly by label length
        # and, once wrapped, a lone card filled the whole row to show "3".
        with st.container(horizontal=True):
            st.metric("Total rows", total, border=True, width="content")
            st.metric(
                "Positive",
                f"{pos_count} ({pos_count / total * 100:.0f}%)",
                border=True,
                width="content",
            )
            st.metric(
                "Negative",
                f"{neg_count} ({neg_count / total * 100:.0f}%)",
                border=True,
                width="content",
            )
            st.metric("Avg confidence", f"{avg_conf:.1%}", border=True, width="content")
            # Conditional, so the common path is still exactly four cards and
            # no file grows a permanently-zero metric. A horizontal container
            # wraps, so the fifth card costs no layout surgery.
            if skipped:
                st.metric(
                    "Skipped",
                    skipped,
                    help="Empty, whitespace-only or missing text — not classified.",
                    border=True,
                    width="content",
                )

            # The chart card is the band's one stretching child, so it takes
            # whatever width the cards leave -- the row's spare width goes to
            # the two bars rather than to padding inside five metric cards.
            with st.container(border=True):
                st.markdown("**Sentiment distribution**")
                # This key is a locally built two-row chart frame, never the
                # user's CSV, so it cannot collide -- but it is the frame's key
                # *and* the x= binding below, so the two must stay matched or
                # the chart is built against a column that is not there. It
                # follows the resolved name for the same reason everything else
                # does: under a collision the literal named "Sentiment", which
                # in a user's file holds the user's own values. (It used to be
                # the rendered axis title too; x_label="" below retires that
                # job, not this one.)
                dist_df = pd.DataFrame(
                    {
                        sentiment_col: ["positive", "negative"],
                        "Count": [pos_count, neg_count],
                        # Literal hex, not a category to be mapped:
                        # st.bar_chart documents that a color column already
                        # holding hex strings is used verbatim rather than
                        # assigned from the palette.
                        #
                        # Row order is load-bearing and the coupling is
                        # invisible: the emitted spec carries `scale.range` in
                        # *row* order and no `domain` at all, so Vega-Lite
                        # derives the domain by sorting these hex strings
                        # ascending. The mapping is therefore correct only
                        # while the column is already ascending -- true today
                        # because "#21c354" < "#ff4b4b". Reorder these two rows
                        # and positive draws red, with nothing in the spec to
                        # say so.
                        CHART_COLOR_COL: [POSITIVE_COLOR, NEGATIVE_COLOR],
                    }
                )
                # color= names the hex column above. With no color at all both
                # bars drew in one palette color, so the single card whose
                # whole job is to compare positive against negative drew them
                # indistinguishably, and the metric cards beside it were the
                # only place the split was legible. Coloring *by category*
                # instead (color=sentiment_col) does adapt per mode, but picks
                # two shades of the same blue and adds a legend restating the
                # axis -- it has no notion that one of these is good news. The
                # literal hexes are the tint's, so green and red mean one thing
                # across the card and the table below it; the usual objection
                # to a pinned hex (it cannot flip lightness with the
                # background) is what the mid-lightness of these two answers,
                # and is why the tint could already hardcode them.
                #
                # x_label="" drops the axis title, which rendered rotated down
                # the left edge, restated the card heading directly above it,
                # and took that width from the bars. y_label is deliberately
                # left alone, so "Count" still says what the numbers are.
                #
                # sort=False, because the default (True) hands the categorical
                # axis to Vega-Lite's ascending sort and alphabetises it:
                # "negative" above "positive", reversing the Positive/Negative
                # cards beside it. False means "data order", so the chart
                # follows the frame built above rather than the spelling of
                # whatever id2label happens to return.
                #
                # height= because the default sizes the chart for a card of its
                # own; in the band it would make the row several times taller
                # than the metric cards beside it, and two bars do not need it.
                st.bar_chart(
                    dist_df,
                    x=sentiment_col,
                    y="Count",
                    color=CHART_COLOR_COL,
                    horizontal=True,
                    sort=False,
                    x_label="",
                    height=CHART_HEIGHT,
                )

        # Full width under the summary band rather than beside the chart:
        # side by side, the table's share of the main area overflowed its own
        # columns in any window narrower than about 1280px.
        with st.container(border=True):
            # Download in the card's header row, next to the thing it
            # downloads, rather than below a table that can be taller than the
            # screen. The source name beside the label because nothing else
            # names the data after Sample: the uploader is empty then, and the
            # name otherwise surfaces only in the downloaded file's name. The
            # stretch space pushes the button to the right edge.
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Results**")
                st.caption(source_name)
                st.space("stretch")
                _download_button(result_df, source_name)
            # Styler does value-based coloring; column_config does formatting
            # (per Streamlit guidance). The tint is a subtle, theme-safe rgba so
            # it reads on light and dark themes.
            sentiment_tint = {
                "positive": _tint(POSITIVE_COLOR),
                "negative": _tint(NEGATIVE_COLOR),
            }
            # The Styler builds a per-cell style for every row, which defeats
            # st.dataframe's virtualization; skip the (cosmetic) tint above
            # STYLE_ROW_CAP. The CSV download uses the unstyled result_df.
            #
            # The second clause is not a second opinion about cost -- it is a
            # hard limit. streamlit rejects a Styler on `styler.data.size >
            # pd.options.styler.render.max_elements` (262,144 by default), i.e.
            # on *cells*, while STYLE_ROW_CAP counts *rows*. A frame under 2000
            # rows but over the cell budget -- >131 columns at 2000 rows, >262
            # at 1000 -- passed this guard, got wrapped, and then raised
            # StreamlitAPIException inside st.dataframe below. Nothing catches
            # it, so the script aborted: no results table *and* no download
            # button, losing a classification the user had already waited for.
            # (Download has since moved into this card's header row, ahead of
            # the grid, so the same abort would now cost the table alone --
            # which is still the thing the user waited for.)
            # Note the tint is one column but the limit counts the whole frame.
            # `<=` because streamlit raises on `>`, so a frame sitting exactly
            # on max_elements is legal and should keep its tint.
            display_df = result_df
            if len(result_df) <= STYLE_ROW_CAP and result_df.size <= int(
                pd.options.styler.render.max_elements
            ):
                display_df = result_df.style.map(
                    lambda v: sentiment_tint.get(v, ""), subset=[sentiment_col]
                )
            # Without an explicit width a column is "sized to fit the cell
            # contents", and a long review auto-sizes to glide's 500px maximum.
            # The grid is the main area less the sidebar and padding (606px at
            # a 1100px window, 946px at 1440), so at the narrow end two such
            # columns, or one plus the generated pair, need a horizontal
            # scroll. Cap every *free-text* source column rather than only the
            # classified one: a second text column blows the same budget.
            # Narrow columns stay auto-sized: padding a 2-char `id` out to
            # TEXT_COL_WIDTH would spend the very budget this cap exists to
            # protect, and the same argument covers *short* text columns -- a
            # ground-truth Sentiment column of "positive"/"negative" auto-sizes
            # to 78px, so capping it would cost 222px of the same budget. `_is_long_text` is the whole test now; there is no dtype
            # clause, because the base `Column` sets a width WITHOUT declaring
            # a type. `TextColumn(width=...)` also emits
            # type_config={"type": "text"}, and it was that coupling -- not any
            # layout fact -- that forced numeric columns to be excluded
            # wholesale: capping a numeric column meant rendering its numbers
            # as text. Column decouples the two, so a numeric column under a
            # >50-char header (the one shape the dtype check used to catch, and
            # the one it could never fix) now gets capped like anything else.
            # The exclusion holds the *resolved* names, which is what makes it
            # collision-proof: a source column named Sentiment is a source
            # column like any other, while the model's renamed column keeps the
            # config assigned below.
            column_config = _width_caps(
                result_df, exclude=(sentiment_col, confidence_col)
            )
            # Both entries drop the positional label: st.column_config documents
            # label=None as "the column name is used", so the header self-syncs
            # with the frame and a renamed "Sentiment (model)" can never
            # disagree with the downloaded CSV.
            #
            # Both are pinned, which does two things at once. A stretched grid
            # hands its spare width to every *unpinned* column in equal shares,
            # so unpinned, these two took as much of it as the review text did
            # (text 599 / Sentiment 377 / Confidence 450px at a 1920px window)
            # and the text stayed truncated on the widest screens; pinned, they
            # keep their natural width and the free-text columns get all of it.
            # And a pinned column stays in view however the grid scrolls, so
            # when a narrow window does force a horizontal scroll, it is the
            # user's columns that scroll -- the prediction and its confidence
            # can no longer be the part pushed off the grid. The cost is
            # position: pinned columns render at the left edge, ahead of the
            # file's own columns (the download keeps the file's order).
            column_config[sentiment_col] = st.column_config.TextColumn(
                help="Predicted sentiment (blank for empty or missing text).",
                pinned=True,
            )
            # color="blue" rather than the default, which is the theme's
            # primary -- Streamlit red in light mode. A full red bar beside a
            # 99.9% label reads as an alarm on the one column that is
            # reporting the model's certainty, and it put red directly against
            # the red the sentiment column beside it uses to mean "negative",
            # so the same color meant two things in adjacent cells. (In dark
            # mode the primary is the violet accent .streamlit/config.toml
            # keeps for the Classify action, so it would be the wrong job
            # there too.) A named color, not a hex, so it still adapts per
            # mode; blue because it keeps green and red reserved for
            # sentiment. Not "auto" (green above half, red below): a binary
            # softmax maximum lives in [0.5, 1.0], so auto is green for every
            # scored row and its threshold reports nothing.
            column_config[confidence_col] = st.column_config.ProgressColumn(
                help="Model confidence in the predicted sentiment.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
                color="blue",
                pinned=True,
            )
            # placeholder="" because the default renders a missing cell as the
            # literal word "None". process_dataframe's fillna("") applies to the
            # local `texts` list, not to the frame, so `result = df.copy()`
            # keeps the source column's real NaN -- and the app then
            # contradicted itself on the same row, printing "None" for the
            # user's blank text next to a genuinely empty generated Sentiment
            # cell. samples/blank_cells.csv is exactly this shape.
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config=column_config,
                placeholder="",
            )
    return drew_results_table


# The sidebar holds the inputs, top to bottom in the order they are used:
# file, then (once a file is loaded) column and Classify/Reset. The main area
# is left to what the inputs produce.
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    st.button(
        "Sample",
        key="sample",
        icon=":material/dataset:",
        help="Load the built-in sample CSV instead of uploading a file.",
        on_click=_load_sample,
    )
    # The column picker and Classify/Reset (or Reset alone) join these at the
    # foot of the script, once the loaded file's state is settled. Nothing else
    # writes to the sidebar, so they land directly under Sample.

# Below the chrome, not above it. Nothing up to this point needs the model --
# only process_dataframe does -- but streamlit emits deltas as the script runs,
# so loading first meant the page body was *nothing but* a spinner until the
# weights were ready: no title, no uploader, no Sample button, nothing to read
# and nothing to click. That is minutes on a cold cache and still the fp16 load
# plus mx.eval on every fresh process, and it is not one-session-only either --
# st.cache_resource holds a compute lock, so every session that connects during
# the first load blocks on it and gets the same empty page. Kept eager (rather
# than moved into the classify branch) so the load still overlaps with the user
# choosing a file: the uploader posts to /_stcore/upload_file, which does not
# wait on the script thread. Still executed at import, which is what conftest's
# module-level patches exist to intercept.
model, tokenizer = load_model()

# Load a freshly uploaded file once. Guarding on file_id stops the persisted
# uploader value from being re-read on every rerun, which would otherwise undo
# Reset and clobber a Sample selection. _uploaded_id is advanced only after a
# successful read, so a failed upload keeps re-showing its error (instead of
# vanishing on the next rerun) and never leaves the previous file's data on screen.
read_failed = False
if uploaded_file is not None and uploaded_file.file_id != st.session_state.get(
    "_uploaded_id"
):
    try:
        # Exactly one statement in this block, and that is load-bearing.
        # ParserError, EmptyDataError and UnicodeDecodeError are all
        # ValueError subclasses, so the four-name tuple that used to sit
        # below read as a curated allowlist of parse failures while being
        # exactly `except ValueError`. Harmless while pd.read_csv is alone
        # here; add a second statement (an encoding sniff, a dtype coercion,
        # a size check) and its own ValueErrors get swallowed and reported to
        # the user as a malformed CSV, cause discarded -- the plausible-and-
        # wrong failure mode the collision rule below exists to avoid.
        new_df = pd.read_csv(uploaded_file)
    except ValueError:
        # Drop any previously loaded data so the failed upload can't keep
        # presenting the old file's preview/results as if it were this one.
        # The error itself renders in the state chain below.
        _clear_data()
        read_failed = True
    else:
        st.session_state["_uploaded_id"] = uploaded_file.file_id
        st.session_state["df"] = new_df
        st.session_state["source_name"] = uploaded_file.name.rsplit(".", 1)[0]
        _clear_results()
elif uploaded_file is None and "_uploaded_id" in st.session_state:
    # The one uploader transition nothing else observed. Clicking the widget's
    # X makes st.file_uploader return None, which fails the guard above, so the
    # branch was simply skipped and `df` was read straight out of session_state
    # a few lines down -- leaving the preview, metrics, chart, results table and
    # a Download button all describing a file the uploader now reports as
    # absent. Same rule the failed-read arm already enforces, applied to the
    # has-file -> no-file edge.
    #
    # Guarded on _uploaded_id rather than a bare `is None`: the uploader is
    # empty by construction on the rerun that _load_sample populates df, and
    # both _load_sample and _reset pop _uploaded_id via _reset_uploader, so
    # neither can reach here and wipe the data it just loaded. Popping the id
    # (rather than calling _reset_uploader) is deliberate too -- bumping
    # uploader_key would remount a widget that is already empty.
    st.session_state.pop("_uploaded_id", None)
    _clear_data()

df = st.session_state.get("df")
source_name = st.session_state.get("source_name", "")

# One chain over every state the loaded file can be in, so each arm owns both
# halves of its UI: the message in the main area and the controls in the
# sidebar. Every arm but the empty page renders Reset -- that is the rule
# "Reset is reachable from every state a loaded file can reach" stated as code.
if read_failed:
    # Reset belongs here for the same reason it belongs in the two warning
    # arms, and this arm needs it most: _clear_data() above popped df, so none
    # of the arms below run, while this error is deliberately sticky
    # (_uploaded_id is left stale so it survives benign reruns). Without this
    # the only exits are the uploader's X and Sample. _reset() is the right
    # escape rather than a bare _clear_data(): it also bumps uploader_key,
    # remounting an empty uploader, which is what actually retires the stale
    # id and the error with it.
    st.error(
        "Could not read this file. Please check it's a valid CSV.",
        icon=":material/error:",
    )
    with st.sidebar:
        _reset_button()
elif df is None:
    # With the uploader in the sidebar, the main area would otherwise be a
    # title over empty space -- and with the sidebar collapsed (a narrow
    # window, or the user closed it) nothing on screen would say where the
    # uploader went. A plain card rather than st.info: nothing has happened
    # yet, and callout weight is reserved for things that have.
    with st.container(border=True, width="content"):
        st.markdown("**Get started**")
        st.markdown(
            "1. Upload a CSV with a text column in the sidebar, or click "
            "**Sample** to try the built-in data.\n"
            "2. Check the text column — it is detected automatically.\n"
            "3. Click **Classify**, then download the results."
        )
elif df.empty:
    # Both warning arms offer Reset: the file is loaded but unusable, which is
    # exactly when the user needs a way back to an empty page.
    st.warning(
        "This CSV has no rows. Please upload a file with data.",
        icon=":material/warning:",
    )
    with st.sidebar:
        _reset_button()
elif (default_col := detect_text_column(df)) is None:
    st.warning(
        "No text columns detected. Please check your CSV.",
        icon=":material/warning:",
    )
    with st.sidebar:
        _reset_button()
else:
    columns = df.columns.tolist()
    with st.sidebar:
        text_column = st.selectbox(
            "Text column",
            options=columns,
            index=columns.index(default_col),
            # The 512-token truncation is said here, where the text is chosen:
            # nothing else in the UI mentions it since the page caption that
            # carried it was removed.
            help=(
                "Select the column containing English text for sentiment "
                "classification. Text longer than 512 tokens is truncated."
            ),
            # Scoped to the loaded dataset, not to its headers. Unkeyed, this
            # widget's identity is a hash of (label, options, index, ...), so
            # two files with the same header list and the same auto-detected
            # column shared one widget: a manual override on the first silently
            # carried into the second, and `index` -- the auto-detect -- was
            # ignored. That rule was an implementation detail of streamlit's id
            # computation rather than anything this app chose. Keying on the
            # load makes it explicit: every new upload, Sample or Reset mints a
            # fresh widget, so auto-detect applies to every file. Both parts are
            # available by now -- uploader_key from the setdefault above, and
            # _uploaded_id written before this branch renders.
            key=(
                f"text_column_{st.session_state['uploader_key']}_"
                f"{st.session_state.get('_uploaded_id', 'sample')}"
            ),
        )
        # Horizontal container (not fixed-width columns) so each button is as
        # wide as its label+icon needs and neither wraps to a second line.
        with st.container(horizontal=True):
            classify_clicked = st.button(
                "Classify",
                type="primary",
                icon=":material/play_arrow:",
                key="classify",
            )
            _reset_button()

    if classify_clicked:
        with st.spinner("Classifying..."):
            # Bound as classified_df so it does not shadow the result_df
            # read back from session_state a few lines below.
            classified_df, generated_cols = process_dataframe(
                df, text_column, model, tokenizer
            )
            st.session_state["result_df"] = classified_df
            st.session_state["result_col"] = text_column
            # Stored, not recomputed at render time. Recovering the pair
            # from result_df by *name* is undecidable (a source frame
            # carrying both Sentiment and "Sentiment (model)" is
            # indistinguishable from a renamed one), and recovering it
            # *positionally* would encode an append-order contract that
            # nothing enforces.
            st.session_state["result_generated_cols"] = generated_cols

    # Render persisted results so post-classify reruns (e.g. a manual Rerun
    # or a widget change) don't collapse the view or re-run inference.
    # Invalidate when the selected column no longer matches what was run.
    result_df = st.session_state.get("result_df")
    if (
        result_df is not None
        and st.session_state.get("result_col") == text_column
        # Part of the guard, not a lookup: the three result_* keys are
        # written in one branch and popped together by _clear_results, but
        # they can still come apart across a *code* change --  streamlit's
        # file watcher reruns the edited script against the session_state
        # that is already there, so a session holding results from before
        # this key existed would otherwise hit an uncaught KeyError and
        # render a traceback in place of the page. Treating the missing
        # pair as "no results yet" degrades to the pre-classify view.
        and "result_generated_cols" in st.session_state
    ):
        # Indexed, not .get() with a default, now that the guard above
        # settles presence: a default of the plain names would silently
        # tint, size and count the user's own Sentiment column on exactly
        # the input this indirection exists for.
        drew_results_table = _render_results(
            result_df,
            source_name,
            st.session_state["result_generated_cols"],
            announce=classify_clicked,
        )
    else:
        drew_results_table = False

    # Bound to whether a results *table* was actually drawn, not to the
    # guard above. The two come apart on the all-blank branch, which emits
    # an info callout and no table: keying off the guard suppressed the
    # preview there too and left the page with no tabular data anywhere,
    # on exactly the run where seeing the column is what tells the user
    # they picked the wrong one. Otherwise the results table is the whole
    # file plus the two generated columns, so leaving the preview up would
    # repeat the file directly below the frame containing it. Not an expander:
    # nothing is being tucked away for later, the element has simply
    # finished its job. It returns the moment it is useful again --
    # selecting a different column invalidates the results and puts the
    # user back to choosing, which is what the preview is for.
    #
    # No out-of-order slot is needed any more: Classify lives in the
    # sidebar, so nothing in the main area has to render above the preview
    # before the decision to draw it is made.
    if not drew_results_table:
        with st.container(border=True):
            st.markdown("**Preview**")
            # Names the source and the column, since nothing else on the page
            # names the data after Sample (the uploader is empty then).
            st.caption(
                f"{source_name} — {_count(len(df), 'row')}, "
                f"{_count(len(df.columns), 'column')}. "
                f"Classifying `{text_column}`, shown first."
            )
            # The whole file rather than the selected column alone: the
            # picker lists headers only, and when auto-detect picks the
            # wrong text column (an id, a date, a product name) the values
            # beside it are what show which header holds the reviews. The
            # selected column is moved to the left edge with column_order,
            # not pinned: a pinned column is left out when the grid spreads
            # its spare width across the columns, so a single-column file
            # rendered one 300px column beside an empty grid. Long columns
            # get the same cap as the results table.
            preview_config = _width_caps(df, exclude=(text_column,))
            preview_config[text_column] = st.column_config.Column(
                width=(
                    TEXT_COL_WIDTH
                    if _is_long_text(df[text_column], text_column)
                    else None
                ),
                help="The column that will be classified.",
            )
            # placeholder="" for the same reason as the results grid: a
            # missing cell otherwise reads as the word "None", and
            # blank_cells.csv has several in its first rows.
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                # str(): column_order is matched against the Arrow-serialized
                # headers, which are always strings. pd.read_csv only ever
                # produces string labels, but a frame with integer labels
                # otherwise raised TypeError in the proto assignment.
                column_order=[
                    str(c)
                    for c in (text_column, *(c for c in columns if c != text_column))
                ],
                column_config=preview_config,
                placeholder="",
            )
