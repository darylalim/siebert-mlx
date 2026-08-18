import os
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple, cast

# mlx 0.32.0 ships no .pyi stubs for its compiled `core` extension (0.31.2 did),
# and ty cannot resolve a binary module without one. Remove the suppression once
# upstream restores the stubs.
import mlx.core as mx  # ty: ignore[unresolved-import]
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
# the built-in light and dark backgrounds alike -- which is what lets them
# be hardcoded at all, and was already the standing argument for the tint.
# Verified on both themes. The built-in *categorical* palette is not an
# option here: it adapts per mode but carries no positive/negative meaning,
# so it drew the two bars in two shades of the same blue.
POSITIVE_COLOR = "#21c354"
NEGATIVE_COLOR = "#ff4b4b"
# The chart wants the solid hue; the table wants a wash behind text. Same
# two colors at two strengths, so this is the only thing that differs.
TINT_ALPHA = 0.12
# Pixel cap for free-text columns in the results table. Sized against the
# centered layout's 736px content cap: the two generated columns plus a narrow
# id take ~278px, so 300 leaves real headroom instead of the zero slack that
# 400px ("large") left. Past three-ish text columns nothing fixed can prevent
# a horizontal scroll, and that is the correct outcome.
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
        fd, tmp_name = tempfile.mkstemp(
            dir=local_dir, prefix="model.safetensors.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            save_file(pt_weights, tmp_path)
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
    # the loader thread's GPU stream, and a later rerun's mx.eval fails with
    # "There is no Stream(gpu, 0) in current thread."
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

            probs = mx.softmax(model(**inputs).logits, axis=-1)
            max_probs = mx.max(probs, axis=-1)
            preds = mx.argmax(probs, axis=-1)
            mx.eval(max_probs, preds)

            # preds/max_probs are 1-D, so .tolist() is always a list here;
            # cast narrows mlx's `int | float | list` return type for the checker.
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


st.set_page_config(page_title="SiEBERT MLX", page_icon=":material/sentiment_satisfied:")

st.title("SiEBERT MLX")
# The landing page otherwise said nothing about what the app does or what it
# wants, and every requirement surfaced only as a post-hoc rejection ("No text
# columns detected") after the user had already chosen a file. The 512-token
# truncation was invisible everywhere in the UI. st.caption, not st.info: this
# is orienting metadata, not an instruction, and the callout weight is reserved
# for things that actually happened.
st.caption(
    "Classify the sentiment of English text on Apple Silicon. "
    "Upload a CSV with a text column — text longer than 512 tokens is truncated."
)

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
    """The one Reset button, rendered by whichever arm of the df branch is live.

    Extracted so the two warning arms can offer it too. It used to exist only
    alongside Classify in the `else` arm, so a CSV with no rows or no text
    column showed a dead-end warning: the only ways out were Sample or the
    uploader's X, and the X did not clear state at all until that was fixed.
    One `key` for all three call sites is safe -- they are mutually exclusive
    branches, so exactly one renders per run.
    """
    st.button("Reset", icon=":material/refresh:", key="reset", on_click=_reset)


def _render_results(result_df, source_name, generated_cols):
    # Unpacked once rather than read as generated_cols.sentiment at each of the
    # nine sites below: the lookups then read as plain column names, a near
    # literal swap of the hardcoded strings they replace, and any (sentiment,
    # confidence) 2-tuple works, which is what lets the flow tests hand-seed a
    # plain tuple. No default value -- a default would silently reinstate the
    # literals on exactly the input this parameter exists for.
    sentiment_col, confidence_col = generated_cols

    # Say so when the source CSV forced a rename. Here rather than in
    # process_dataframe because results re-render from session_state on every
    # rerun: a notice emitted during classify would vanish on the first one,
    # including the rerun the Download click itself causes -- precisely when
    # the user needs to know what the file's headers mean. Above the all-blank
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
            # screen" to be unchanged there. The download button is outside
            # that split and always renders, so that half stays true on both.
            f"This file already has {noun} named {taken}, so the model's "
            f"output was added as {added}. Your original columns are "
            "unchanged, including in the download.",
            icon=":material/info:",
        )

    if result_df[sentiment_col].eq("").all():
        st.info(
            "All values in this column are empty. No classification was performed.",
            icon=":material/info:",
        )
    else:
        st.success("Classification complete!", icon=":material/check_circle:")

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

        with st.container(horizontal=True):
            st.metric("Total rows", total, border=True)
            st.metric(
                "Positive",
                f"{pos_count} ({pos_count / total * 100:.0f}%)",
                border=True,
            )
            st.metric(
                "Negative",
                f"{neg_count} ({neg_count / total * 100:.0f}%)",
                border=True,
            )
            st.metric("Avg confidence", f"{avg_conf:.1%}", border=True)
            # Conditional, so the common path is still exactly four cards and
            # no file grows a permanently-zero metric. A horizontal container
            # wraps, so the fifth card costs no layout surgery.
            if skipped:
                st.metric(
                    "Skipped",
                    skipped,
                    help="Empty, whitespace-only or missing text — not classified.",
                    border=True,
                )

        with st.container(border=True):
            st.markdown("**Sentiment distribution**")
            # This key is a locally built two-row chart frame, never the user's
            # CSV, so it cannot collide -- but it is the frame's key *and* the
            # x= binding below, so the two must stay matched or the chart is
            # built against a column that is not there. It follows the resolved
            # name for the same reason everything else does: under a collision
            # the literal named "Sentiment", which in a user's file holds the
            # user's own values. (It used to be the rendered axis title too;
            # x_label="" below retires that job, not this one.)
            dist_df = pd.DataFrame(
                {
                    sentiment_col: ["positive", "negative"],
                    "Count": [pos_count, neg_count],
                    # Literal hex, not a category to be mapped: st.bar_chart
                    # documents that a color column already holding hex strings
                    # is used verbatim rather than assigned from the palette.
                    "_color": [POSITIVE_COLOR, NEGATIVE_COLOR],
                }
            )
            # color= names the hex column above. With no color at all both
            # bars drew in one palette color, so the single card whose whole
            # job is to compare positive against negative drew them
            # indistinguishably, and the metric row above it was the only place
            # the split was legible. Coloring *by category* instead
            # (color=sentiment_col) does adapt per mode, but picks two shades
            # of the same blue and adds a legend restating the axis -- it has
            # no notion that one of these is good news. The literal hexes are
            # the tint's, so green and red mean one thing across the card and
            # the table below it; the usual objection to a pinned hex (it
            # cannot flip lightness with the background) is what the
            # mid-lightness of these two answers, and is why the tint could
            # already hardcode them.
            #
            # x_label="" drops the axis title, which rendered rotated down the
            # left edge, restated the card heading two lines above it, and took
            # that width from the bars. y_label is deliberately left alone, so
            # "Count" still says what the numbers are.
            # sort=False, because the default (True) hands the categorical axis
            # to Vega-Lite's ascending sort and alphabetises it: "negative"
            # above "positive", reversing the metric row directly above. False
            # means "data order", so the chart follows the frame built above
            # rather than the spelling of whatever id2label happens to return.
            st.bar_chart(
                dist_df,
                x=sentiment_col,
                y="Count",
                color="_color",
                horizontal=True,
                sort=False,
                x_label="",
            )

        with st.container(border=True):
            st.markdown("**Results**")
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
            # contents", so one long review takes the row and pushes
            # Confidence's percentage off the grid — the centered layout caps
            # content at 736px at any window size, so a wider browser cannot
            # rescue it. Cap every *free-text* source column rather than only
            # the classified one: a second text column blows the same budget.
            # Narrow columns stay auto-sized: padding a 2-char `id` out to
            # TEXT_COL_WIDTH would spend the very budget this cap exists to
            # protect, and the same argument covers *short* text columns -- a
            # ground-truth Sentiment column of "positive"/"negative" auto-sizes
            # to 67px, so capping it cost 233px and pushed Confidence off the
            # grid. `_is_long_text` is the whole test now; there is no dtype
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
            column_config = {
                col: st.column_config.Column(width=TEXT_COL_WIDTH)
                for col in result_df.columns
                if col not in (sentiment_col, confidence_col)
                and _is_long_text(result_df[col], col)
            }
            # Both entries drop the positional label: st.column_config documents
            # label=None as "the column name is used", so the header self-syncs
            # with the frame and a renamed "Sentiment (model)" can never
            # disagree with the downloaded CSV.
            column_config[sentiment_col] = st.column_config.TextColumn(
                help="Predicted sentiment (blank for empty or missing text).",
            )
            # color="blue" rather than the default, which is the theme's
            # primary -- Streamlit red on both built-in themes. A full red bar
            # beside a 99.9% label reads as an alarm on the one column that is
            # reporting the model's certainty, and it put red directly against
            # the red the sentiment column beside it uses to mean "negative",
            # so the same color meant two things in adjacent cells. A named
            # color, not a hex, so it still adapts per mode; blue because it
            # keeps green and red reserved for sentiment. Not "auto" (green
            # above half, red below): a binary softmax maximum lives in
            # [0.5, 1.0], so auto is green for every scored row and its
            # threshold reports nothing.
            column_config[confidence_col] = st.column_config.ProgressColumn(
                help="Model confidence in the predicted sentiment.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
                color="blue",
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
    # marshall_file routes a callable to the deferred-file path, which is
    # served on the download request itself and so still runs with no rerun to
    # attach to. Everything else that reruns (theme toggle, any widget) still
    # re-renders from session_state exactly as before, which is why the notice
    # above must stay in this function rather than in process_dataframe.
    st.download_button(
        label="Download",
        data=lambda: result_df.to_csv(index=False),
        file_name=f"{source_name}_sentiment.csv",
        mime="text/csv",
        icon=":material/download:",
        key="download",
        on_click="ignore",
    )


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
        _clear_data()
        st.error(
            "Could not read this file. Please check it's a valid CSV.",
            icon=":material/error:",
        )
        # Reset belongs here for the same reason it belongs in the two warning
        # arms, and this arm needs it most: clearing df means the `if df is not
        # None:` block below never runs, so none of *its* _reset_button() call
        # sites fire, while this error is deliberately sticky (_uploaded_id is
        # left stale so it survives benign reruns). Without this the only exits
        # are the uploader's X and Sample. _reset() is the right escape rather
        # than a bare _clear_data(): it also bumps uploader_key, remounting an
        # empty uploader, which is what actually retires the stale id and the
        # error with it.
        _reset_button()
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

if df is not None:
    # Both warning arms end in _reset_button(): the file is loaded but unusable,
    # which is exactly when the user needs a way back to an empty page.
    if df.empty:
        st.warning(
            "This CSV has no rows. Please upload a file with data.",
            icon=":material/warning:",
        )
        _reset_button()
    elif (default_col := detect_text_column(df)) is None:
        st.warning(
            "No text columns detected. Please check your CSV.",
            icon=":material/warning:",
        )
        _reset_button()
    else:
        columns = df.columns.tolist()
        text_column = st.selectbox(
            "Text column",
            options=columns,
            index=columns.index(default_col),
            help="Select the column containing English text for sentiment classification.",
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

        # Reserved here, filled at the foot of this branch. The preview has
        # to sit above Classify -- it is the "did I pick the right column"
        # check the user makes *before* paying for inference -- but whether to
        # draw it at all is only settled below, after the classify branch has
        # run. st.empty is the documented way to insert an element out of
        # order, and it is what keeps the swap on the same run: filled in
        # place, the preview could only react one rerun late, so the run that
        # classified would still draw the duplicate and some later, unrelated
        # rerun would drop it for no visible reason.
        preview_slot = st.empty()

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

        # Render persisted results so post-classify reruns (e.g. the Download
        # click or a theme toggle) don't collapse the view or re-run inference.
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
            _render_results(
                result_df, source_name, st.session_state["result_generated_cols"]
            )
        else:
            # Only while there are no results on screen. The results table is
            # this same column plus the two generated ones, so once it renders
            # the preview is five rows of text repeated directly above the full
            # frame that contains them -- ~230px of duplication between the
            # user and the thing they waited for. Not an expander: nothing is
            # being tucked away for later, the element has simply finished its
            # job. The `if` arm's guard is the whole condition, so the preview
            # comes back exactly when it is useful again -- selecting a
            # different column invalidates the results and returns the user to
            # choosing, which is what the preview is for.
            with preview_slot.container():
                st.caption("Preview of selected column")
                # placeholder="" for the same reason as the results grid: a
                # missing cell otherwise reads as the word "None", and
                # blank_cells.csv has one inside the first five rows .head()
                # shows.
                st.dataframe(
                    df[[text_column]].head(),
                    width="stretch",
                    hide_index=True,
                    placeholder="",
                )
