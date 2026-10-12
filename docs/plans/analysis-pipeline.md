# Plan: an analysis pipeline around SiEBERT

Status: **drafted 2026-10-12, not started.** Implement in a fresh session; read this file and
`CLAUDE.md` first.

Four phases, each merged before the next starts:

0. **Complementary light and dark themes (0A), an adaptive layout (0B), and pinned models with a
   private, offline-capable setup (0C).** Designed below; 0B starts with measurement spikes.
1. **Emotion detection.** Designed in full below, and ready to implement. Adds large-file limits
   (cross-cutting E).
2. **Topic clusters.** Outline only; see "Later phases".
3. **LLM explanations and aspects.** Outline only; see "Later phases".

Between Phases 1 and 2, **non-English detection** (cross-cutting F) ships as its own release.

---

# Phase 0: themes, adaptive layout, pinned and private

## Phase 0A: complementary light and dark themes (designed 2026-10-12)

**Ship Phase 0 before Phase 1, as its own release.** Phase 1 re-measures geometry, recaptures the
screenshot and adds a chart, so doing the theme first means all of that is measured once, against
the final colors.

### Why

Light mode is Streamlit's built-in theme, and it has **the exact defects the dark theme was made to
fix**:

- **The Classify button is `#ff4b4b`, which is `NEGATIVE_COLOR` itself**, so the one call to action
  reads as "negative". Its white label is **3.30:1**, under the 4.5 floor.
- **The positive bar `#21c354` is 2.33:1 on white**, under the 3:1 floor for chart marks
  (WCAG 1.4.11). Light mode's distribution chart fails on exactly the bar that means good news.
- **The built-in light `codeTextColor` follows `greenTextColor`**, so the preview caption's column
  name reads as the positive green (the same defect the dark theme fixed). Confirm by rendering.
- **The two modes don't feel like one app:** the bluish built-in light surfaces (`#f0f2f6`) and the
  red accent share nothing with the neutral graphite dark mode and its violet accent.

### Design: one identity, mirrored

Neutral surfaces, one violet accent, and the data supplying the only other saturated color, in
both modes. The values are chosen so each light role mirrors its dark role at the **same contrast
step**:

| Role (key) | Dark (shipped) | Light (new) | Light ratio | Dark ratio |
|---|---|---|---|---|
| `backgroundColor` | `#111214` | `#ffffff` | | |
| `secondaryBackgroundColor` (sidebar, header) | `#1f2024` | `#efeff1` | 1.15 vs main | 1.15 vs main |
| `textColor` | `#f2f2f4` | `#111214` (dark's background) | body 18.74, sidebar 16.32 | 16.76 |
| muted text (text at 0.6) | | | captions 4.89, sidebar 4.70 | 6.51 |
| `primaryColor` | `#7a58d8` | **`#7a58d8` (same)** | white label 4.98, 4.98 on main, 4.34 on sidebar | 4.98, 3.76, 3.27 |
| `blueColor` (confidence bars) | `#2b78cc` | **`#2b78cc` (same)** | 4.51 on grid | 4.16 |
| `borderColor` | `#48494e` | `#b4b4ba` | 2.06 main, ~1.79 sidebar | 2.09, 1.81 |
| `dataframeHeaderBackgroundColor` | `#1f2024` | `#efeff1` | muted header text 4.70 | 6.09 |
| `codeTextColor` | `#f2f2f4` (= text) | `#111214` (= text) | caption 4.70 on sidebar | 6.28 |

The light ratios are computed from the config arithmetic, the same way `tests/test_theme.py` does
it; they have not been measured on a rendered page yet (see Verify). The light primary is the dark
primary, so its hue clearance (117° from the positive green, ~104° from the negative red, 45° from
the confidence blue) carries over unchanged.

```toml
[theme.light]
primaryColor = "#7a58d8"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#efeff1"
textColor = "#111214"
blueColor = "#2b78cc"
codeTextColor = "#111214"
borderColor = "#b4b4ba"
dataframeHeaderBackgroundColor = "#efeff1"
```

Each key gets a comment carrying its numbers, matching the `[theme.dark]` style. Rewrite the
file's header comment too, which currently says light mode stays built-in.

**Left unset on purpose:**

- **`yellowColor`, `redTextColor`.** The dark theme needed them because the built-in washes went
  olive or too dim on graphite; light washes don't have that problem. Measure them on the rendered
  page, and set them only if a number fails.
- **`chartCategoricalColors`.** Nothing draws from it today. Phase 1's emotion chart will use the
  built-in light palette, which suits a light background.
- **No `[theme.light.sidebar]`.** The sidebar swaps surfaces on its own, exactly as in dark.

### One app change: `POSITIVE_COLOR` `#21c354` → `#1d9e45`

The sentiment colors are literal hexes, not theme keys, so they can't differ per mode. The
alternative, picking colors with `st.context.theme.type`, was **rejected**: a theme toggle does not
rerun the script (verified on 1.64.0, see CLAUDE.md), so the chart would show the previous mode's
colors until some unrelated rerun.

Instead, choose one green that reads in both modes. A color clears 3:1 on both `#ffffff` and
`#111214` only if its relative luminance is between ~0.12 and ~0.30.

| Constant | Old | New | On white | On `#111214` | Pinned text on its tint, light / dark |
|---|---|---|---|---|---|
| `POSITIVE_COLOR` | `#21c354` (2.33 / 8.03) | **`#1d9e45`** | 3.48 | 5.38 | 4.69 / 6.07 |
| `NEGATIVE_COLOR` | `#ff4b4b` | **unchanged** | 3.30 | 5.68 | 4.67 / 6.10 |

Ripples to handle in the same commit:

- **Tint literal:** `test_tints_the_generated_column` hardcodes `rgba(33, 195, 84, 0.12)`. It
  becomes `rgba(29, 158, 69, 0.12)`. Keep it a literal; CLAUDE.md explains why that test must not
  derive it.
- **Chart color order:** `test_chart_colors_ascend_because_vega_sorts_the_domain` still holds,
  since `#1d9e45` < `#ff4b4b`. Re-run it anyway; that ordering is exactly what the test exists to
  catch.
- **Docs:** update CLAUDE.md's chart-colors bullet, which calls these "Streamlit's own semantic
  green/red". The green is now the app's own.

### Tests (`tests/test_theme.py`)

- **`TestScope`:** `[theme]` holds exactly `{"dark", "light"}`, with no top-level keys (a key there
  reaches both modes and was never the plan). Both sections are colors-only and use six-digit hex.
- **Palettes:**
  - Build `LIGHT` / `LIGHT_SIDEBAR` the way `DARK` / `SIDEBAR` are built (sidebar = palette +
    sidebar section + swapped surfaces).
  - Parametrize every `TestContrast` floor over all four palettes.
  - Includes the pinned-prediction-on-tint, sentiment-bar, confidence-bar and grid-header tests,
    which currently read `DARK` only.
- **New tests: the modes stay complementary.**
  - `primaryColor` and `blueColor` are identical in both modes. That's the shared identity, and a
    one-sided edit should fail loudly.
  - The main-to-sidebar step stays within ±0.05 of each other across modes.
- **Mutation-test** each new assertion, as the dark suite was.

### Verify (headless, like the dark theme)

- **The menu:** confirm on 1.64.0 that with both `[theme.light]` and `[theme.dark]` defined, the
  main menu still offers **System / Light / Dark**, and that each mode renders its own section.
- **States to render in both modes:**
  - landing, preview, dropdown open, toast;
  - results, including a tinted row and the chart;
  - error, warning and the collision notice;
  - sidebar hover on Classify, and focus ring.
- **Measure** on the rendered page:
  - the alert washes and their text, which `test_theme.py` doesn't model;
  - Classify's hover fill, `darken(primary, 0.15)` = `#522cba` on `#efeff1`;
  - the focus ring, `primary@0.5`.
- **Record** the measured light ratios as comments in `config.toml`, replacing the computed ones in
  the table above.
- **Known residuals** to check, not fix: the focus ring and the dropdown highlight, as in dark.
- **Fonts, sizes and radii are untouched**, so `TEXT_COL_WIDTH`, `LONG_TEXT_CHARS` and the geometry
  numbers stay valid. Spot-check one grid width anyway.

### Docs and release

- **CLAUDE.md:**
  - project overview, and the three dark-theme bullets (now "themes": scope, palette,
    constraints);
  - the chart-colors bullet;
  - the Stop-hook note (`.streamlit/config.toml` is already fingerprinted, no change);
  - retire the "light mode is pixel-identical to built-in" claim;
  - the README Screenshots section: the capture stays dark, but say light is now custom.
- **Screenshot:** optionally add a light capture (`color_scheme="light"` in
  `docs/capture_screenshot.py`). CLAUDE.md records that a light shot was deliberately dropped on
  2026-08-15, so this is a choice to revisit, not an omission to fix.
- **Version:** Phase 0 (themes, the adaptive layout **and** 0C below) ships together as **0.8.0**.
  Phase 1 then becomes **0.9.0**; the commit-10 wording below is updated to match.
- **Commits:**
  - T1: `feat(ui): add a light theme that mirrors the dark one`. Covers the config, `POSITIVE_COLOR`
    and the tint literal, `test_theme.py`, and the measured comments.
  - T2: `docs: document the paired themes`.
  - The version bump moves to the end of Phase 0C (P4).

## Phase 0B: a layout that adapts to different displays (designed 2026-10-12)

### Constraint: one layout, not per-device versions

- **The server can't see the viewport.** `st.context` (1.64) exposes `cookies, headers,
  ip_address, is_embedded, locale, theme, timezone, timezone_offset, url`, and no width. Guessing
  from the User-Agent header was rejected: it gets wrong exactly the case that matters most here,
  a resized desktop window.
- **No custom CSS.** Streamlit's own theming guidance, and this repo's practice, rule out
  breakpoint CSS. CLAUDE.md records that `stDecoration`-style selectors silently rot across
  versions.
- **So adaptation comes only from three things:**
  1. Streamlit's own breakpoints: sidebar overlay at ≤ 767 px, main padding 16 → 80 px at 864.
  2. Wrapping flex rows: `st.container(horizontal=True)` wraps its children.
  3. Width budgets chosen so the *narrowest* supported display still works.

### Display tiers (CSS px; Retina scaling doesn't change these, browser zoom does)

| Tier | Width | Real devices | Target |
|---|---|---|---|
| Phone (LAN only, opt-in per 0C) | 320–767 | iPhone 375–430 | **Works**: no page-level horizontal scroll, everything reachable. Not optimized |
| Compact | 768–1099 | half-screen windows on a MacBook, iPad | **Good**: nothing clipped; the table may scroll horizontally |
| Laptop | 1100–1727 | MacBook built-in screens: 1512 (14"), 1728 (16") | **Primary design target**: Sample results fit with no table scroll, sidebar open |
| Desktop | 1728–2559 | 1920 / 2560 external monitors | **Good**, no dead space |
| Ultrawide | ≥ 2560 | 3008 Studio Display, 3440 ultrawide | **Readable**: no 2,500 px-wide bars or text lines |

The usable table width at each W is already recorded in CLAUDE.md's geometry bullet:
- sidebar open: `W − 494` for W ≥ 864, `W − 366` for 768–863;
- collapsed or phone: `W − 194` for W ≥ 864, `W − 66` below 864.

So a 360 px phone has **294 px**, a 390 px phone **324**, and the 1512 MacBook **1018** with the
sidebar open.

### Rules (apply to Phase 0B and every later phase)

1. **Pinned-width budget: pinned columns total ≤ 360 px.**
   - **Why:** pinned columns never scroll, so a pinned set wider than the table leaves nothing
     scrollable.
   - **Today:** `Sentiment` 78 + `Confidence` 150 = 228.
   - **Phase 1 as drafted:** 456 px, which is wider than a phone's whole table area and leaves
     110 px in a 900 px half-screen window with the sidebar open.
   - **Fix (Phase 0B, L2):** give every pinned `ProgressColumn` an explicit narrow width (target
     ≤ 100 px; try `width="small"` and an int, keep the narrowest whose "99.9%" label still reads
     in both themes). Then Phase 1's four pinned columns total about 78 + 100 + 70 + 100 ≈ 348 with plain headers.
   - **Why not unpin the scores instead:** unpinned columns get `grow: 1`, so on a 1920 screen two
     confidence columns would take spare width from the review text (text ≈ 576 px instead of
     ≈ 970). Pinning narrow is better at every tier.
   - **Budget the collision state, not just plain headers.** When the file already has a
     `Sentiment` or `Emotion` column, the generated headers become `Sentiment (model)` (126 px,
     measured in CLAUDE.md, against 78) and `Emotion (model)` (~118, estimated). Auto-sized text
     columns would then total about 126 + 100 + 118 + 100 ≈ 444, over budget on exactly the
     labeled-CSV path the rename exists for.
     - **Preferred fix:** give all four pinned columns explicit widths (labels ~90, scores ≤ 100),
       so the total doesn't depend on the names. A renamed header then truncates
       (`Sentiment (mo…`). Check in L1 screenshots that it's still distinguishable from the user's
       own `Sentiment`, which renders in full. The rename notice already names both columns, and a
       `help=` on each pinned column can carry the full name.
     - **Fallback**, if truncation reads badly: allow the collision state up to ~450 px, and rely on
       L2's spike result for what glide does when pinned columns are wider than the table.
     - Either way, L1 measures the collision state (its labeled `text,Sentiment` CSV, plus an
       `Emotion` column once Phase 1 lands), and the pinned-total test covers the renamed names too.
   - **Phase 2's `Topic` must fit the same budget or stay unpinned** (that was already an open
     question there; now there's a number for it).
2. **No page-level horizontal scroll at any width ≥ 320 px** (WCAG 1.4.10 Reflow).
   - The grid's *own* horizontal scroll is allowed: data tables are exempt from that rule as
     two-dimensional content, and pinning keeps the predictions in view.
   - Check: `document.documentElement.scrollWidth <= innerWidth` at every tier.
3. **The Results header holds at most two controls at any width:** `Download` (button) +
   `Exports` (`st.popover`, icon `:material/more_vert:` or `:material/download:`).
   - Secondary downloads live in the popover: the summary (D) and the conditional filtered
     download (B). This replaces "a second button in the header row" in B and D.
   - **Why:** three buttons beside the label and caption would wrap unpredictably in a compact
     window. A popover takes the same width at every tier and grows downward, not sideways.
   - Download stays a one-click button: it's the most common action and its tests pin it in the
     header row.
4. **Rows that wrap, wrap in reading order.** Metric cards and the sentiment chart stay one
   horizontal container (already true: the chart drops to its own row at ~1100). Phase 1's emotion
   chart is its own row, never a third child of the metric row. Check at every tier that cards
   never squeeze below their label width (`width="content"` already guarantees this).
5. **Sidebar content stays short enough for a 768 × 600 window.**
   - Uploader, Sample, column picker and Classify/Reset should be reachable without scrolling the
     sidebar.
   - Phase 1 filters go **below** Classify/Reset, so a scroll only ever hides filters, never the
     action.
   - Use `st.pills` for the 7 emotions (wraps in the 300 px sidebar) rather than a long
     `st.multiselect` list.

### Work items

- **L1. `scripts/layout_matrix.py`**, a manual tool like `docs/capture_screenshot.py`
  (`uv run --with playwright`, system Chrome, headless).
  - **Viewports:** 360×780, 390×844, 768×1024, 900×700 (half-screen), 1100×800, 1512×982,
    1728×1117, 1920×1080, 2560×1440, 3440×1440.
  - **Sidebar:** open and collapsed (≥ 768).
  - **States:** landing, Sample preview, Sample results, `blank_cells.csv` results, a labeled
    `text,Sentiment` CSV.
  - **Records:** each grid's `clientWidth` / `scrollWidth`, the pinned total, how many rows the
    metric row wraps into, the chart card width, whether the Results header row wrapped (control
    `top`s equal), page-level `scrollWidth` against `innerWidth`, and whether Classify is in view
    with no sidebar scroll.
  - **Output:** a CSV plus screenshots in a scratch directory.
  - **Baseline** the current app first, before any 0B change, the same way the benchmark (C)
    baselines Phase 1.
- **L2. Pinned budget:** narrow the `Confidence` `ProgressColumn` per rule 1, then re-run L1.
  - **Spike:** what glide does when the pinned set is wider than the table (390 px phone today
    leaves 96 px). Does the unpinned area still scroll, or does it collapse? This decides whether
    rule 1's 360 is enough or should be lower.
  - **Tests:** update `test_generated_columns_are_not_width_capped` (it asserts the
    `ProgressColumn` call whole) to the new width.
  - **Collision state:** measure the pinned total with a colliding `Sentiment` column too, and
    decide between rule 1's preferred fix and its fallback.
  - **CLAUDE.md geometry:** `Confidence`'s base width changes from 150, so every S and every
    overflow onset in CLAUDE.md's geometry bullets moves. That should be an improvement; re-measure
    all five shapes.
- **L3. Ultrawide readability cap (spike first).**
  - **Check:** whether `st.container(width=<int>)` acts as a *maximum*, i.e. falls back to the
    parent's width when the parent is narrower. Measure it; don't assume.
  - **If it does:** wrap the summary band (and Phase 1's emotion chart row) in a container capped
    around 1600 px, so a two-bar chart isn't 2,500 px wide on a 3440 display.
  - Decide whether to cap the results card too. The trade-off is long-review readability against
    showing more columns. Choose it from L1 screenshots at 2560 / 3440, not by reasoning.
  - **If it doesn't:** leave ultrawide uncapped and document it. No CSS.
- **L4. Table height.**
  - `st.dataframe`'s default height shows about 10 rows on every display, wasting most of a
    1117 px-tall MacBook screen or a portrait monitor.
  - Evaluate a larger fixed height (e.g. 15 rows), checking it still leaves the summary band and
    the table header on screen at 1100×800.
  - `height="stretch"` needs a bounded parent, which the main flow isn't. Confirm on 1.64.
  - Pick the value from L1 screenshots, and pin it as a constant like `CHART_HEIGHT`.
- **L5. Docs and release:**
  - CLAUDE.md: replace the geometry and overflow numbers with L1's matrix, add the display tiers
    and the five rules, and update the phone note in the layout bullet.
  - The `chore: bump version to 0.8.0` for all of Phase 0 comes after 0C (P4).

### What stays as it is

- Wide layout, `initial_sidebar_state="auto"`, inputs in the sidebar, `TEXT_COL_WIDTH` /
  `LONG_TEXT_CHARS`. CLAUDE.md records why each was chosen, and nothing here contradicts it.
- Phones keep the known cost that the sidebar stays open over the results until closed by hand.
  Streamlit 1.64 has no API to close it, and moving inputs into the main area for phones would need
  the viewport width the server doesn't have.

## Phase 0C: pinned models, privacy and offline use (designed 2026-10-12)

Two small changes that touch loading and config rather than the UI. They land in Phase 0 because
Phase 1 adds a second model, and both are cheaper to establish with one model than to retrofit
across two.

### P1. Pin every model to a commit SHA

- **Why:** `load_model` resolves `main` **three times**, independently: `AutoConfig`,
  `snapshot_download` (inside `_ensure_safetensors`) and `AutoTokenizer`. So:
  - an upstream push changes predictions on the next launch with no change in this repo, and
    nothing records which checkpoint produced a downloaded CSV;
  - a push landing between the three calls could even mix revisions in one load (unlikely, but
    nothing prevents it).
  - CLAUDE.md's CI note already pays for this: the hub cache key's `-v1` suffix has to be bumped
    by hand "if upstream moves `main`", and no one is told when it does.
- **Today's SHAs (checked 2026-10-12 via the Hub API, matching the cache's `refs/main`):**
  - `siebert/sentiment-roberta-large-english` → `74cea614e245b0832c770ec9aa51bd58df965b9c`
    (upstream last modified 2024-06-11);
  - `j-hartmann/emotion-english-distilroberta-base` → `0e1cd914e3d46199ed785853e12b57304e04178b`
    (2023-01-02), pinned in Phase 1 commit 3.
- **Design:**
  - A module-level `SENTIMENT_MODEL = ModelRef(repo, revision)` NamedTuple (Phase 1 adds
    `EMOTION_MODEL`; Phase 1's `_load_classifier` takes a `ModelRef`).
  - Pass `revision=` to **all three** calls. `_ensure_safetensors` takes it as a required
    argument, the same "required, so `ty` flags every missed call site" reasoning as
    `_render_results`' generated pair.
- **Spike results (2026-10-12, warm cache, `huggingface_hub` 1.32.0):**
  - With the SHA pinned, all three calls succeed **offline** and return the same snapshot folder.
  - A wrong SHA fails all three loudly offline (`OSError` / `OfflineModeIsEnabled`), never falling
    back to `main`.
  - Online, a pinned load still makes Hub requests: 0.68 s against 1.36 s for `main`. **Pinning is
    not an offline mode;** P3 is.
- **Tests:**
  - Unit: each revision is a 40-character lowercase hex SHA, never a branch name. Mutation-test it
    with `"main"`.
  - Unit: the patched `AutoConfig`, `snapshot_download` and `AutoTokenizer` each receive the same
    `revision=`. Mutation-test by dropping it from one call.
  - `conftest.py`'s module-level `snapshot_download` mock must accept the new keyword.
- **CI cache key:** derive it from the pins instead of `-v1`, so a pin bump busts the cache by
  itself and the manual suffix retires (Phase 1 commit 8 then needs no bump):
  ```yaml
  key: hf-hub-${{ runner.os }}-${{ steps.pins.outputs.hash }}
  ```
  Keying on `hashFiles('streamlit_app.py')` would bust it on every code edit. **Where
  `steps.pins` gets the hash (decide in implementation):**
  - a step that greps the `ModelRef(...)` lines into `$GITHUB_OUTPUT` and keys on that (no new
    file; `actionlint` checks the step), **or**
  - a `models.lock` text file that the app reads (one source, but a second file the app depends
    on).
  - Recommendation: the grep step, plus a unit test that the pattern it greps still matches every
    `ModelRef`, so a reformatted line can't silently empty the key.
- **Updating a pin** (document in CLAUDE.md as a procedure): change the SHA, run
  `pytest --integration` (the pinned confidences will move if the weights did), re-run
  `scripts/eval_quality.py` (A) and record the numbers, then commit as `build(models): ...`.
- **Knock-on edits in this plan:**
  - S1's "still resolves the current `main` revision, so a new upstream revision is still picked
    up" becomes "resolves the pinned revision".
  - S2's stale revisions now arise only from a deliberate pin bump.
  - The last Risks bullet (upstream adding `model.safetensors` skips the conversion automatically)
    now needs a pin bump to take effect.

### P2. Private by default

- **Streamlit's usage statistics.** `browser.gatherUsageStats` defaults to **true** (checked on
  1.64.0), so the browser reports usage events to Streamlit. Set `[browser] gatherUsageStats =
  false` in `.streamlit/config.toml`.
- **Network exposure.** `server.address` defaults to unset, which **binds every interface**: anyone
  on the same Wi-Fi can open the app, upload files and use this Mac's GPU. Each session's data
  stays separate, but the exposure is still surprising for a local tool. **Decision for the user
  at implementation:**
  - **Recommended:** `[server] address = "localhost"`. Phone testing (0B's phone tier) then opts in
    per run with `--server.address 0.0.0.0`, documented in the README and in 0B's L1 notes.
  - Alternative: keep the default and document it.
- **Hugging Face telemetry.** Check what `huggingface_hub` / `transformers` send beyond the
  downloads themselves (the user-agent string and any `send_telemetry` calls) on the locked
  versions. If anything, set `HF_HUB_DISABLE_TELEMETRY=1` in `load_model` before the first call.
  Don't assume either way.
- **Config and tests:**
  - The config file's header comment ("Dark mode only…") is already rewritten by 0A. Give
    `[browser]` / `[server]` their own comments.
  - `test_every_key_is_a_streamlit_option` already validates every section, not only the theme.
  - Add `TestPrivacy` to `tests/test_theme.py`: `gatherUsageStats` is false, and (if chosen)
    `server.address` is `localhost`. Renaming the file to `test_config.py` would be more accurate,
    but CLAUDE.md cites `test_theme.py` throughout; leave it.

### P3. Offline mode, and the claim that nothing leaves the machine

- **It already works:** with a warm cache, `HF_HUB_OFFLINE=1 uv run streamlit run
  streamlit_app.py` loads with no network (spike above: 0.06 s for the three calls). Document it
  rather than build it.
- **Cold cache under offline mode** fails in `load_model` with an `OSError` and a traceback. Catch
  it at the call site and show an `st.error` naming the fix ("run once online to download the
  model, about 1.4 GB"), so the page body still renders. Pin it with a flow test that raises from
  the patched loader.
- **Verify the README claim before writing it.** Run the app under Playwright with
  `HF_HUB_OFFLINE=1` and `gatherUsageStats=false`, go through upload → Classify → Download, and
  record **every** browser request. The claim holds only if every host is `localhost`. Fonts,
  icons and Vega are expected to be bundled, but check. Reuse L1's harness.
- **README wording** (only after that check): "After the first run downloads the model, the app
  works offline (`HF_HUB_OFFLINE=1`). Your text is processed on this Mac and is never sent
  anywhere."

### P4. Docs and release

- CLAUDE.md: the `load_model` architecture item (revisions), the CI cache-key note (it's derived
  now), a new privacy bullet (the two config keys and why), and the pin-update procedure.
- README: offline use, the privacy sentence, and how to open it from a phone.
- `chore: bump version to 0.8.0` for all of Phase 0.
- **Commits:** P1 `feat: pin model revisions` (with the CI key), P2 `feat: private by default`,
  P3 `feat: offline mode` (the cold-cache error), then docs and the bump.

---

# Phase 1: emotion detection

## Goal

Every Classify run adds two columns next to `Sentiment` / `Confidence`: **`Emotion`** (one of
`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) and **`Emotion score`**.
Results show an emotion distribution, and the download carries all four generated columns.

## Decisions (made with the user, 2026-10-12)

| Decision | Choice | Why |
|---|---|---|
| Model | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | Same `RobertaForSequenceClassification` class SiEBERT uses, so it reuses the whole MLX load path; 82M params, ~160 MB in fp16 |
| Score column | **`Emotion score`**, not `Emotion confidence` (2026-10-12) | Phase 0B narrows pinned score columns to ≤ 100 px. `Emotion confidence` (18 characters) needs ~130 px and would truncate, and CLAUDE.md rules out a short `label=` because headers must match the download. `Emotion score` is ~13 characters, estimated at ~100 px from CLAUDE.md's header widths; **measure it in L2/§5**. SiEBERT's `Confidence` keeps its name |
| Activation | **Always on**, no toggle | No new state to invalidate. A toggle would need result invalidation like a column change, plus its own flow tests |
| Rejected | LLM stage (`mlx-lm`), embedding clusters, RoBERTa-large emotion model | LLM: new dependency, answers vary run to run, seconds per row. Clusters: needs a choice of cluster count and a new UI. Large model: ~4× the per-row cost, not spiked |

## Spike evidence (2026-10-12, mlx-transformers as locked)

Loaded via the app's exact path (`snapshot_download` → `.bin`→safetensors → `RobertaForSequenceClassification(config)` → `from_pretrained(float16=True)` → `mx.eval(model.parameters())`), compared against `transformers.pipeline` on CPU:

- Loads cleanly: 6 layers, 105 weight tensors (the `.bin` holds 106, and the extra `position_ids` buffer is ignored). 0.12 s warm.
- Logits come back **float32** (same fp32 attention-mask promotion as SiEBERT).
- Max |MLX − torch| probability difference over 7 texts: **0.00013**. Top label was the same for all 7.
- Speed: **70 rows in 0.28 s** (batches of 8). Expect roughly +10% on top of SiEBERT's run time.
- Its tokenizer produces **identical `input_ids`** to SiEBERT's (both are RoBERTa BPE, vocab 50265, 512-token limit).
- Spike top labels, useful later as integration pins: `"I love this!"` → joy 0.977, `"Oh wow. I didn't know that."` → surprise 0.975, `"I'm terrified the update will wipe my data."` → fear 0.993. Recompute to 4 decimals in the integration test; don't copy these numbers.
- The cache now holds a converted `model.safetensors` for this repo (mode fixed to 0644 by hand). So the **conversion path is untested for this model on a cold cache**. Exercise it once by hand in a scratch `HF_HOME`.

## Design

### 1. Loading (`streamlit_app.py`)

- Add `EMOTION_MODEL = ModelRef("j-hartmann/emotion-english-distilroberta-base", "0e1cd914e3d46199ed785853e12b57304e04178b")` beside 0C's `SENTIMENT_MODEL` (0C, P1).
- Extract the body of `load_model` into an **uncached** `_load_classifier(ref: ModelRef)` that returns `(model, tokenizer)` and passes `ref.revision` to all three loading calls. Keep the `mx.eval` and its comment there; it applies to every model.
- `load_model()` keeps its name, signature and decorator, so conftest, the `real_model` fixture and the CLAUDE.md references still hold. Add a sibling `load_emotion_model()` with `@st.cache_resource(show_spinner="Loading emotion model...", show_time=True)`.
- Call site: `emotion_model, emotion_tokenizer = load_emotion_model()` directly after `model, tokenizer = load_model()`. Same eager-below-the-chrome placement, for the same reasons (see CLAUDE.md's UI bullet).
- `_ensure_safetensors` is already generic in the repo. By this point it takes 0C's required `revision` and has been split in two by S1 (commit 2), so Phase 1 adds nothing to it. Reword its "1.4 GB" comment so it doesn't read as SiEBERT-only.
- Use a **separate tokenizer** per model, even though the ids are identical today. Tokenizing is negligible next to the encoder, and sharing one would silently break if either model were swapped.

### 2. Inference

Keep `process_dataframe`'s 2-tuple contract **unchanged**, so the existing unit tests stay a pure safety net for the refactor:

- Extract `_valid_sorted(texts) -> list[(idx, text)]`, the blank filter plus the length sort.
- Extract `_classify_texts(valid, model, tokenizer, progress_bar) -> (labels, confs)`, the batch loop (tokenize, `return_dict=True`, softmax/max/argmax, `mx.eval`, `cast` on `.tolist()`, write-back by `idx`).
- `process_dataframe` calls both. Its output must be byte-identical; the existing suite proves it.
- New `add_emotions(result, source_df, text_column, model, tokenizer) -> (DataFrame, EmotionColumns)`:
  - Returns a copy of `result` with the two emotion columns appended.
  - Blanks get `""` / `0.0`, like sentiment.
  - Shows its own `st.progress(0, text="Detecting emotions...")` and empties it when done. Also give `process_dataframe`'s bar a `text=` so the two bars can be told apart. **Commit 5 (E) replaces the two bars with one run-wide bar** and an overall ETA.
- New `EmotionColumns(emotion, confidence)` NamedTuple. Resolve names with `_unique_column_name` against `taken = set(source_df.columns) | set(generated_cols)`:
  - Constants `EMOTION_COL = "Emotion"` and `EMOTION_SCORE_COL = "Emotion score"`.
  - A source `Emotion` column is preserved and the model's becomes `Emotion (model)`, same rule and direction as sentiment.
  - Check that `Emotion score` doesn't fall inside `Emotion`'s name space (it doesn't; the prefix test is `Emotion (model)`), and add a unit test for it.
- Use a separate NamedTuple rather than widening `GeneratedColumns`. `_render_results` unpacks the pair positionally and the flow tests hand-seed 2-tuples, so widening it would ripple through every one of those for no gain.

### 3. State

- The classify branch runs `add_emotions` right after `process_dataframe`, under the same spinner. It stores the resulting frame as `result_df` and the `EmotionColumns` as **`result_emotion_cols`**.
- `_clear_results` pops `result_emotion_cols` with the other three `result_*` keys.
- Make `"result_emotion_cols" in st.session_state` part of the render guard, for the same file-watcher-across-a-code-change reason that `result_generated_cols` is. This is the exact scenario that ships with this change: a running dev session holding pre-emotion results.
- `_render_results(result_df, source_name, generated_cols, emotion_cols, announce=False)`. `emotion_cols` is **required**, with no default, so `ty` flags every missed call site.

### 4. UI (`_render_results`)

- **Rename notice:** add `(EMOTION_COL, emotion_col)` and `(EMOTION_SCORE_COL, emotion_score_col)` to the `renamed` filter. The wording ("the model's output was added as …") still reads correctly.
- **Results table:**
  - `Emotion` gets a `TextColumn`; `Emotion score` gets a `ProgressColumn(format="percent", color="blue")` at the **same narrow width Phase 0B chose for `Confidence`**. Both are `pinned=True` and excluded from `_width_caps`. The four pinned columns must total ≤ 360 px (Phase 0B, rule 1), **including when both sentiment and emotion names are renamed**; assert both totals in a test.
  - **No tint on `Emotion`.** Green and red mean positive and negative in this app, and 7 extra hues would clash with that and with the contrast checks in `tests/test_theme.py`.
- **Second summary row:** an **`Emotion distribution`** bordered card under the existing band.
  - `st.bar_chart` with `horizontal=True`, `sort=False`, `x_label=""` and no `color=` (the palette is fine here, since emotion carries no good/bad meaning).
  - Rows in **`id2label` order**, not sorted by count, so the axis is stable between runs.
  - Height around 7 × 28 px. Pick it in the browser, then pin it as a constant (`EMOTION_CHART_HEIGHT`) the way `CHART_HEIGHT` is pinned.
  - Its frame's key is `emotion_col`, not a literal (same lesson as `test_chart_axis_follows_the_generated_column`).
- **No new metric card.** The first row already squeezes the chart at 1280. Revisit a "Top emotion" card only after measuring.
- **Copy:**
  - Text-column `help=`: "…for sentiment and emotion classification…". Keep the 512-token note; it's the only place the UI states it.
  - Get started card: mention emotions.
  - The title stays "SiEBERT MLX".
- **All-blank arm:** no change. Emotion columns are blank on exactly the same rows, because both passes use the same `_valid_sorted`.

### 5. Geometry: must re-measure

The original estimate was **+228 px to S** (≈78 for `Emotion`, 150 for a default-width progress column), which would move the Sample's overflow onset from 1022 to ≈1250 with the sidebar open. **Phase 0B's narrow progress columns replace that.** With both confidence columns at ≤ 100 px, the addition is roughly +170 px on top of Phase 0B's new baseline. Measure rather than trust this.

- Re-run `scripts/layout_matrix.py` (Phase 0B, L1) across all its viewports and states, plus a results state with emotions.
- Rules 1–5 from Phase 0B must still hold, in particular the 360 px pinned budget and no page-level horizontal scroll at 320 px.
- Rewrite CLAUDE.md's geometry bullets from the new matrix.

## Tests

Order matters. Land the refactor **first, with the existing suite unchanged and green**, before any emotion code exists.

- **Unit (`tests/test_streamlit_app.py`):**
  - Generalize `_make_mock_model` to take an `id2label` (currently 2-class). Add a 7-class pair.
  - `add_emotions`: labels follow the text, not the row (sorted batching); blanks get `""`/`0.0`; input not mutated; returns a copy.
  - `EmotionColumns` naming: plain names, a source `Emotion` column, a source `Emotion (model)` column, and both sentiment and emotion colliding at once.
  - `load_emotion_model` mirrors the existing `TestLoadModel` tests (repo, the same `revision=` on all three loading calls, `float16=True`, `mx.eval`, token).
- **Flow (`tests/test_app_flow.py`):**
  - `_FakeModel` is used for *both* models, since conftest patches the class globally. It answers 2-wide logits, which `id2label` maps to anger/disgust. Either is fine, but make that explicit in its docstring, or make its width follow `len(config.id2label)`.
  - New tests:
    - the emotion card renders above the Results card;
    - both emotion columns are pinned, in the table and in the download;
    - a colliding `Emotion` column triggers the notice;
    - a `session_state` holding results but missing `result_emotion_cols` degrades to the preview instead of crashing;
    - the hand-seeded result tests get an `emotion_cols` tuple.
  - `TestRenderResultsRenderingKwargs`: pin `sort=False`, `x_label=""` and the height on the new chart.
- **Integration (`tests/test_inference_integration.py`):** a `real_emotion_model` fixture (or extend `real_model`), the float16 weights / float32 logits check, the 7-label `id2label`, and three 4-decimal pins.
- **Network note:** `AutoConfig`/`AutoTokenizer` are unpatched, and `streamlit_app` calls both loaders at import, so every suite now also resolves the emotion model's config and tokenizer files (~2.6 MB) from the hub or a warm cache. Update CLAUDE.md's ~1.2 MB figure.

## CI / release

- **No manual cache-key bump.** Since 0C (P1), the `integration` job's hub cache key is derived from the pinned revisions, so adding `EMOTION_MODEL` in commit 3 changes the key by itself. Before 0C, the exact static key would have hit, never saved, and re-downloaded and re-converted the emotion checkpoint on every run. Confirm on the first green run that the new key saves a cache containing both models.
- Bump `[project].version` to `0.9.0` (Phase 0 ships as 0.8.0) **only in the final commit, once everything is green**. The `release` job publishes a GitHub Release as soon as a bumped version lands green on `main`.

## Docs

- CLAUDE.md:
  - the overview (four generated columns), Architecture items 3–4 (two loaders, `add_emotions`), and Key Patterns (emotion names, `result_emotion_cols`, the new chart, re-measured geometry);
  - the Tests section;
  - the `torch` bullet: now **two** `.bin`-only checkpoints depend on the conversion.
- README: feature list, and recapture `docs/screenshot-dark.png` with `docs/capture_screenshot.py`. Re-measure `stMain.scrollHeight`, since the page gets taller.

## Suggested commit sequence

0. `chore: add scripts/benchmark.py` (C) and record the SiEBERT-only baseline **before anything else changes**.
1. `refactor: extract _load_classifier, _valid_sorted, _classify_texts`: no behaviour change; the existing suite passes unmodified.
2. `perf: fetch pytorch_model.bin only when no converted safetensors exists` (S1) + tests. Before the emotion model, so both checkpoints benefit. Record the disk saved.
3. `feat: load the emotion model and add add_emotions` + unit tests. Adds `EMOTION_MODEL` with its pinned SHA (0C, P1).
4. `feat(ui): render emotion columns and distribution` + state, guard, flow tests.
5. `feat: large-file limits` (E): the live ETA, the row ceiling, the character pre-slice, + tests. After commit 4, so the limits are set against the two-model cost.
6. `feat(ui): sidebar filters for sentiment and emotion` (B) + flow tests.
7. `feat(ui): summary export` (D) + the shared counts helper.
8. `test: add the labeled eval set and pin emotion inference against the real checkpoint` (A). Lands in three steps the user gates: the labeling guide is agreed, then the user reviews every drafted label, then the κ re-label. The gate's counts are pinned only after that.
9. `docs: emotion detection; large-file limits; disk usage (S2, S5); re-measure grid geometry; recapture screenshot; record eval, benchmark and per-tab memory numbers (S3)`.
10. `chore: bump version to 0.9.0` + the repo description and topics below, **in the same commit**, so no copy advertises emotion detection before it's on `main`.

### Repo description and topics (at release)

The description lives in **three places** that must match: `pyproject.toml` `description`,
README line 8 (the sentence under the badges), and GitHub's About.

- **All three were synced on 2026-10-12** to the pre-emotion text:
  - description: "Classify the sentiment of English text in CSV files, locally on Apple Silicon. A
    Streamlit app running SiEBERT (RoBERTa-large) with MLX that adds sentiment and confidence
    columns to your data." The README's copy links SiEBERT to its model card.
  - GitHub topics: added `nlp`, `huggingface`, `csv`, `macos`, `on-device-ai`, `python`, for 14 in
    total.
- **In commit 10, set all three to:**
  > Classify the sentiment and emotion of English text in CSV files, locally on Apple Silicon. A
  > Streamlit app running SiEBERT and an emotion model with MLX that adds labels and confidence
  > scores to your data.
- **Update GitHub** only *after* commit 10 is merged and green, since it's outward-facing:
  ```bash
  gh repo edit darylalim/siebert-mlx --description "<text above>" \
    --add-topic emotion-detection,emotion-recognition
  ```
  That gives 16 topics, under GitHub's 20.
- **Later phases:** add `topic-modeling` with Phase 2 and `llm` with Phase 3, each in that phase's
  release commit. Don't add them before the phase ships.

---

# Beyond Phase 1

Everything below applies to the later phases or to all of them, not just emotion detection.

## Later phases (proposed 2026-10-12, not yet designed in detail)

Order: **emotions (above) → topic clusters → LLM explanations.** Each phase reuses the patterns the
one before it established: cached loader, `_valid_sorted` batching, collision-safe column names, a
required render argument, a `session_state` key in the render guard. So don't start a phase until
the previous one has merged.

### Phase 2: topic clusters

- **What it adds:** a `Topic` column, plus a "Topics" card showing, for each cluster, its size and
  its positive/negative split. This answers *what* the negative reviews are about, which today's
  metrics can't.
- **How:**
  - Compute an embedding per non-blank row, cluster the embeddings, and label each cluster by its
    most distinctive keywords (c-TF-IDF over the cluster's texts).
  - Same input always gives the same output, and no LLM is involved.
  - Fixed seed for clustering; blanks get `""`.
- **Candidate embedding model:** `Qwen/Qwen3-Embedding-0.6B`, or
  `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (0.34 GB).
- **Spike first:**
  - **Embeddings:** check whether the installed `mlx-transformers` (which ships `qwen3.py`) can load
    the base checkpoint and produce last-token-pooled embeddings with no new dependency. Otherwise
    evaluate the `mlx-embeddings` package. The 4-bit DWQ build likely needs mlx-lm-style quantized
    loading, so check that and don't assume either way.
  - **Clustering:** k-means on `mx` arrays is about 20 lines; `scikit-learn` would be a new
    dependency (prefer not to add it).
- **Open question:** how many clusters. Options: a fixed small k (e.g. 6–8) adjusted for row count,
  a sidebar number input (which adds a state that invalidates results), or a silhouette sweep over
  k. Decide after seeing real clusters on the sample CSVs.
- **Watch:** this is a third pinned-or-not column. Re-measure grid geometry again, and decide
  whether `Topic` should be pinned at all; a long keyword label is free text and may want the
  width cap instead.

### Phase 3: LLM explanations and aspects (opt-in, on a subset)

- **Not part of Classify.** A separate sidebar button, e.g. "Explain flagged rows", runs it after
  results exist. The core run stays fast, and its tests stay deterministic.
- **Which rows:** only rows flagged by the earlier stages:
  - low SiEBERT confidence (e.g. < 0.9);
  - sentiment/emotion disagreements (`negative` + `joy` often means sarcasm);
  - optionally, rows the user selects in the table (`st.dataframe` selection).
  - **Cap the count** (e.g. 50), and show what was skipped.
- **Outputs:** an `Aspect` column (what the text is about: "battery", "delivery") and an
  `Explanation` column (one short sentence on why it reads that way).
- **Also:** one call per topic cluster to replace its keyword label with a short name. That's about
  10 calls, cheap enough to run automatically once phase 3 exists.
- **Model:** `mlx-community/Qwen3.5-9B-MLX-4bit` (5.95 GB), with
  `mlx-community/Qwen3.5-4B-MLX-4bit` (3.03 GB) as the fast fallback. The budget on a 32 GB M2 Max
  leaves ample room next to SiEBERT + emotion + embeddings (≈1.2 GB together).
- **New dependency:** `mlx-lm`. Spike first:
  - whether it loads these `image-text-to-text`-tagged builds text-only, or needs `mlx-vlm`;
  - the same thread-local stream rule as `load_model` (`mx.eval` the weights on the loading thread),
    and whether generation on a later rerun thread is safe;
  - per-row latency on the M2 Max.
- **Output handling:**
  - temperature 0;
  - ask for JSON with a fixed schema, validate it, and treat a parse failure as a blank cell
    rather than an error;
  - never let a bad generation abort the run.
- **Tests:** mock the generator in unit and flow tests. In integration, assert only the shape of
  the output (valid JSON, non-empty aspect), never exact text.
- **Lazy-load** the LLM on first click, unlike the eager classifiers: most sessions will never use
  it, and 6 GB is a heavy eager load.

## Cross-cutting additions (agreed 2026-10-12)

Each lands in phase 1 for sentiment + emotion and is extended by phase 2 for topics, unless noted.

### A. Quality check on real data

- **Why:** today the only evidence for any prediction is each model card's own claim. SiEBERT's
  93% is an average over 15 datasets, and the emotion model's 66% was measured on tweets, Reddit
  and dialogue, not on reviews like the ones this app is fed. **Both** models are evaluated: the
  sentiment column is the app's main feature, and it shouldn't be the one left on the card's word.
- **Data:** `samples/labeled_eval.csv`, with columns `text, gold_sentiment, gold_emotion` (and
  `gold_topic` in phase 2).
  - **All 200 unique domain rows, not a sample of 60.** The five domain files hold 200 unique
    texts, and `mixed_sample.csv`'s 20 are all among them (checked 2026-10-12). Labeling the whole
    pool avoids choosing a sample, and gives seven emotion classes ~29 rows each on average rather
    than ~9. One row then moves accuracy 0.5 points instead of 1.7. Draft in five per-domain
    batches, so review can stop and resume.
  - **The `gold_` prefix** keeps the labels apart from the generated `Sentiment` / `Emotion`
    columns, so the eval run never hits the collision path by accident. (A separate flow test
    already covers that path on purpose.) Lowercase `emotion` wouldn't collide with `Emotion`
    either, but the prefix makes it impossible to read the two as the same column.
  - **The labels must come from a human.** Claude may draft them, but a model's labels used to
    grade a model measure agreement between them, not accuracy. The user reviews every row before
    it counts. Record in the file's header comment (or a sibling `.md`) who labeled it and when.
- **Labeling guide: `samples/labeled_eval.md`, written and agreed before any row is labeled.**
  - **`gold_sentiment`:** `positive`, `negative` or `ambiguous`. SiEBERT is binary, so it has
    no right answer for a genuinely neutral or mixed review ("Arrived on time. Does what it says.",
    "Great food, rude staff"). Forcing one would grade the model on a coin flip.
  - **`gold_emotion`:** one of the model's seven labels (`anger, disgust, fear, joy, neutral,
    sadness, surprise`), or `ambiguous`.
  - **`ambiguous` rows are excluded from scoring** and counted separately. The report prints how
    many there are per column, so a guide that excludes too much is visible.
  - **Rules for the known hard boundaries.** The quoted examples here are illustrative; the guide
    itself gives each rule one worked example taken from the pool:
    - `neutral` vs mild `joy`: "Works fine" is `neutral`; "Works great, really happy" is `joy`.
    - `anger` vs `disgust`: anger is aimed at someone or something that did wrong; disgust is
      revulsion (food, hygiene, content).
    - Sarcasm: label the **intended** meaning ("Oh great, it crashed again" is `negative` /
      `anger`), since that's what a user of the app wants back.
    - Mixed reviews: label the overall verdict if one is clear (a closing "would not return"),
      otherwise `ambiguous`.
  - **A self-consistency check stands in for a second annotator.** After labeling, re-label a
    random 20 rows (fixed seed) blind, a day later, and report the agreement (Cohen's κ). With one
    annotator that's the only ceiling on how much the scores can mean. If κ is low for a class,
    tighten its rule rather than trusting its F1.
- **Tool:** `scripts/eval_quality.py`, run by hand like `docs/capture_screenshot.py`. It runs the
  real models over the file through `process_dataframe` / `add_emotions` (the app's own path,
  including the sort and batching) and prints, per model:
  - accuracy and macro-F1 over the scored (non-`ambiguous`) rows;
  - **95% bootstrap confidence intervals** for both (1,000 resamples, fixed seed, stdlib
    `random`; no new dependency). With a few hundred rows the interval is the honest headline, and
    a later run's change can be read against it;
  - **per-class support, precision, recall and F1**, with a class supported by fewer than 10
    scored rows marked `low support`. Macro-F1 is printed twice, over all classes and over
    well-supported ones only, so one rare class (likely `disgust` or `fear` in reviews) can't
    swing the headline unseen;
  - a confusion matrix, and the count of `ambiguous` rows per column;
  - for sentiment, the same metrics **per domain**, since the card's 93% hides domain spread.
- **Record the numbers** in CLAUDE.md, with the date and the versions used: the point estimate
  *and* the interval, per-class support, and κ.
- **Gate:** the models are deterministic, so an integration test can pin the **exact** number of
  correct rows, **one count per model**. That catches a changed checkpoint, tokenizer or label
  mapping without thresholds that can't decide whether a 1-point move is noise. The intervals are
  for humans reading CLAUDE.md, not for the gate.
  - **Pin from CI, not only from the M2 Max.** The `integration` job runs on GitHub's macOS
    runner, a different GPU. The existing 4-decimal pins already pass there, but across 200 rows a
    borderline prediction flipping is likelier than a 4th-decimal move. Before committing the
    counts, run the test both locally and in CI. If they differ, pin CI's value, and record both
    and the rows that flip.
- `samples/` is already in the Stop hook's fingerprint, so edits to the eval set re-run the gates.

### B. Filter the results table

- **Placement:** filters live **in the sidebar**, under Classify/Reset: a `st.multiselect` (or
  `st.pills`) per generated label column. Sentiment and emotion in phase 1, topic in phase 2.
  - This keeps the rule that the main area holds no inputs. Re-check what
    `test_inputs_live_in_the_sidebar_and_data_in_the_main_area` asserts before relying on that.
- **Behaviour:**
  - Filtering is a rerun that re-renders from `session_state`, never re-runs inference. Pin this
    with the `_FakeModel.calls` count.
  - **Metrics and charts stay whole-file. Only the table filters.** The summary describes the run,
    and a table caption says "Showing N of M rows".
- **Widget keys:** scope them to the loaded dataset the same way the text-column selectbox is
  keyed, so a filter set on one file doesn't silently carry into the next, **and to the result**.
  - **Why the result too:** changing the text column invalidates the results without loading a new
    file, and re-classifying produces new labels. Dataset-keyed filters would survive with the old
    selections, hiding every row or keeping emotion values that no longer appear.
  - **How:** a `result_run` counter in `session_state`, bumped every time the classify branch
    writes the `result_*` keys, and part of every filter key. It's never popped (a fresh count is
    what makes a fresh widget), so it stays out of `_clear_results`.
  - **Pin:** set a filter, change the column and back, re-classify, and assert the filter is reset.
- **Download stays whole-file, always.** Its meaning never changes: the source plus the generated
  columns. Making it follow the filter was rejected: a user who forgets a filter is on gets 30 rows
  of a 2,000-row file, silently.
- **Filtered download (decided 2026-10-12): a conditional entry in the `Exports` popover.**
  Phase 0B, rule 3 keeps the Results header to `Download` + `Exports` at any width, so this is not
  a header button.
  - **When:** it renders inside the popover only while a filter actually narrows the table
    (filtered rows < total).
  - **Label and file:** labeled with the count (`Download 30 rows`) and named
    `<source>_filtered.csv`, so the file stays honest after it's forwarded.
  - **Content:** the same column order as the full download, and the same lazy `data=` callable
    over the **unstyled** filtered frame, with `on_click="ignore"` and its own key
    (`download_filtered`).
  - **Pin with tests:** it's absent with no filter; it's absent when a filter selects every value;
    its row count matches the label; the full Download's content is unchanged while a filter is on.
  - **Check:** whether `st.dataframe`'s own toolbar already exports the frame it's given, and so
    the filtered one too, and in what column order. If so, document it as a second, less visible
    route out. Don't treat it as a replacement: it's easy to miss, and its column order may differ.
- **Watch:**
  - The Styler cell-cap guard must be evaluated on the **filtered** frame it actually styles.
  - An empty filter result needs its own message rather than an empty grid.

### C. Performance budget

- **Tool:** `scripts/benchmark.py`, run by hand rather than in CI, since runner timing varies too
  much to gate on.
  - It builds an N-row frame by repeating the sample CSVs (default 10,000 rows, with a fixed
    length mix) and times each stage against the real models: SiEBERT, emotion, embeddings,
    clustering.
  - It reports rows/s and peak memory. Check the current mlx API name for the peak-memory query
    (`mx.get_peak_memory` in recent releases) rather than assuming it.
- **Baseline first:** record the SiEBERT-only number **before** phase 1's first commit.
- **Budgets:** emotion adds ≤ 15% to the baseline; phase 2 adds ≤ 25% on top of phase 1.
  - Phase 3 is opt-in and excluded from the total, but record its seconds per row.
- Record results in CLAUDE.md, with the machine (M2 Max, 32 GB) and the date.

### D. Summary export

- **What:** a second, small CSV for reports, alongside the row-level download.
- **Format:** tidy/long, with columns `dimension, value, rows, share, positive, negative`:
  - one row per sentiment label;
  - one row per emotion label;
  - in phase 2, one row per topic.
  - Use the resolved column names as `dimension`.
- **Mechanics:** an `st.download_button` inside the Results header's `Exports` popover (Phase 0B,
  rule 3), with the same lazy `data=` callable and `on_click="ignore"`, and its own key (e.g.
  `download_summary`). Name the file `<source>_summary.csv`. This commit (7) is where the popover
  first appears; until then the header holds `Download` alone.
- **Skipped rows, and what `share` divides by:** use the metric cards' rule, so the export can't
  repeat the problem the `Skipped` card fixed (shares falling short of 100% with nothing explaining
  the gap).
  - `share` divides by **total rows in the file**, the same denominator as the Positive/Negative
    cards.
  - When any rows were skipped, each dimension gets a `(skipped)` value row, so each dimension's
    shares sum to 100%.
  - Pin it with a test on `samples/blank_cells.csv`: per dimension, `rows` sums to the file's row
    count and `share` sums to 1.
- **Not in the all-blank arm:** there is nothing to summarize there.
- **Single source of truth:** compute the counts in one helper shared with the metric cards and the
  charts, so the export can never disagree with what's on screen. Pin that with a test comparing
  the two.

### E. Large-file limits (Phase 1, commit 5)

- **Why:** nothing caps a file today. Each phase multiplies the per-row cost (SiEBERT is 24 layers,
  emotion adds 6, Phase 2 adds embeddings), and the only feedback during a long run is a progress
  bar with no time on it. A file too big for this Mac fails late: after the upload, or partway
  through a run whose results are then lost.
- **Limits that exist already (checked on 1.64.0):**
  - `server.maxUploadSize` = 200 MB: the uploader refuses larger files itself.
  - `server.maxMessageSize` = 200 MB: one websocket message. The whole-file preview and the
    results table are each sent as one Arrow payload, measured at 40.2 MB for 150k rows (CLAUDE.md,
    preview bullet). Above 150k rows Streamlit switches to lazy loading by itself. **Check** that
    lazy loading keeps a 200 MB CSV's grid under the message limit; don't assume it.
- **Design, in order of value:**
  1. **One run-wide progress bar with a live ETA.** `st.progress(value, text=...)`, with
     "Step 1 of 2: sentiment · about 3 min left" estimated from elapsed time so far. That's
     machine-independent, unlike a rows/s constant from the M2 Max.
     - **One bar for the whole run, not one per model.** With a bar per pass (sentiment, then
       emotion, then F's language), each bar's own ETA claims to be nearly done before the next
       pass starts. Replace them with a single bar: `_classify_texts` takes a progress callback,
       and each pass's share of the bar is weighted by its measured cost from C (a per-model
       constant, e.g. emotion ≈ 0.1–0.25× SiEBERT; record where it came from).
     - **Count characters, not rows.** The length sort runs short rows first, so a rows-based
       early estimate would be optimistic.
     - Pin with a unit test using a fake clock, including the hand-off between passes.
  2. **A pre-run estimate for big files.** Above `WARN_ROWS` (set from C's benchmark: roughly
     where a run passes ~2 minutes), the sidebar shows an `st.caption` under Classify: "~N min
     for this file", from C's measured chars/s. Only above the threshold, since small files need
     no warning. It's labeled approximate and quotes the benchmark machine.
  3. **A hard row ceiling, `MAX_ROWS`, only where something actually breaks.** Set it from
     measurement: the row count where per-tab memory (S3: `df` + `result_df` + the models)
     or the message limit fails on a 16 GB Mac, which is the smallest the README should promise.
     Above it, the `else` arm shows an `st.warning` with the count and the limit, and Classify is
     disabled (Reset stays). **Don't** set a ceiling for time alone: a slow run the user chose is
     their call, which is what the ETA is for.
  4. **Pre-slice very long cells to 40,000 characters** before tokenizing. A fast tokenizer
     tokenizes the **whole** string and only then truncates to 512 tokens, so a 5 MB cell costs
     5 MB of tokenization to keep ~2 KB.
     - **Why 40,000 is safe:** the longest token in both vocabularies is 66 characters (measured
       2026-10-12), so 510 content tokens span at most 33,660 characters. A cut beyond that can't
       change the first 510 tokens. The margin covers the pre-tokenizer's whitespace lookahead at
       the cut.
     - **Prove it:** a unit test with the real tokenizer encodes long texts (including one with
       a cut inside a word and one inside a whitespace run) both ways, and compares `input_ids`
       bitwise. The constant and the 66 go in a comment.
     - **Per tokenizer, not one constant.** The 66 holds only for the two RoBERTa BPE vocabularies.
       F's XLM-R uses a SentencePiece vocabulary, and Phase 2's Qwen3 embedder reads up to 32k
       tokens, so a 40,000-character cut could change its input. Store each model's cut beside
       its `ModelRef`, as `(max_length − 2) × longest token + margin`. A test recomputes the
       longest token from each real tokenizer, so a tokenizer change fails it.
     - Measure the saving on one 1 MB cell first; if it's negligible, drop this item.
- **Cancel:** Streamlit already stops a running script when the user clicks any widget, or Stop in
  the main menu. The interruption lands at the next `st` call, i.e. the next `progress_bar.progress`.
  **Spike:** confirm a sidebar click mid-run stops within one batch, and that no `result_*` key is
  half-written (they're written together after the loop, so it should hold). If both hold, document
  it and add no button. If they don't, `_clear_results()` before the run starts.
- **Tests:** the ETA text (fake clock), the warning-and-disabled-Classify arm above `MAX_ROWS`
  (hand-seeded frame, no real classify), the caption appearing only above `WARN_ROWS`, the
  pre-slice parity test.
- **Docs:** CLAUDE.md records the measured `WARN_ROWS` / `MAX_ROWS` and how they were set; the
  README states the limit.

### F. Non-English input (its own release between Phase 1 and Phase 2)

- **Why:** everything that reads the text (SiEBERT, the emotion model and, in Phase 2, the
  English keyword labels) is English-only, and none of it fails on other languages: each returns a label with a confidence,
  often a high one. A mixed-language file gets silently plausible, meaningless rows.
- **Behavior: flag, never skip.**
  - Skipping would blank the prediction, which already means "this cell was empty". The
    `Skipped` card and the blank-text rules would then mean two things.
  - Instead, add a `Language` column (ISO 639-1 code: `en`, `es`, …), plus a conditional
    `Not English` metric card that renders only when the count is non-zero, the same pattern as
    `Skipped`.
  - Predictions on flagged rows are kept. The filters (B) gain a language filter, and the summary
    export (D) a `language` dimension, so a user can exclude them in one click.
- **Naming and layout:**
  - `Language` goes through `_generated_columns`' collision rule like every generated column
    (`Language (model)` if the file has one).
  - It's **not pinned** (it isn't a prediction the user came for), so it costs nothing against
    0B's 360 px pinned budget.
- **State, loading and layout:** the same plumbing as every generated column, which Phase 1
  established:
  - a `result_language_cols` key, popped by `_clear_results` and part of the render guard (the
    file-watcher-across-a-code-change case again);
  - a **required** `_render_results` argument, so `ty` flags every missed call site;
  - **the model class travels with the `ModelRef`.** `_load_classifier` builds a
    `RobertaForSequenceClassification`, and XLM-R needs `XLMRobertaForSequenceClassification`.
    Add a `model_class` field (defaulting to the RoBERTa class) rather than a second loader. Check
    that the CI cache-key grep (0C, P1) still matches the longer `ModelRef(...)` lines.
  - **The metric row's worst case.** Phase 1 §4 kept the first row to its existing cards because
    the chart is squeezed at 1280. With `Skipped` *and* `Not English` both showing, the row holds
    six cards plus the chart. Add a file with both blank and non-English rows to L1's states, and
    apply rule 4. If the chart drops under ~215 px at 1512, give it its own row in that state
    rather than dropping a card.
- **Detector (spike decides, three candidates):**

| Option | Runs on | Size | Notes |
|---|---|---|---|
| `papluca/xlm-roberta-base-language-detection` (pinned `9865598…`) | **MLX**, via `mlx-transformers`' `xlm_roberta.py` | 278M params, ~0.56 GB fp16 | MIT license; ships `model.safetensors`, so **no `.bin` conversion**; 20 languages, so anything else is forced into one of them (still "not English") |
| `lingua-language-detector` | Rust, CPU | No download; memory depends on mode | Strong on short text; not MLX, so it would go in the non-MLX table |
| A script heuristic (share of non-Latin letters) | Python | None | Catches Chinese/Arabic/Cyrillic and misses Spanish, French, German. Too weak alone |

  - **Recommendation:** the XLM-R model, for consistency (all inference on MLX, one loader path
    via `_load_classifier` and a `ModelRef`), unless the spike shows a false-positive problem.
  - **The metric that matters is false positives on English:** run it over the sample rows (all
    English: 200 unique domain rows, including slang, emoji and app-store shorthand). Each English
    row it flags is a wrong warning on the user's data. Also measure it on a small non-English set.
  - **Tune and report on different rows.** Pick the threshold on two domains (e.g. product and
    movie reviews) and report the false-positive rate on the other three. Tuning and scoring on
    the same rows would make the rate optimistic. `mixed_sample.csv` is a subset of the domain
    files, so it isn't a separate set.
  - **Short texts** ("ok", "lol", "10/10") are where every detector is weakest. Flag a row only
    when the top language isn't `en` **and** its probability clears a threshold picked from the
    spike; below the threshold, write `en`. Record the threshold and why.
- **Data:** `samples/non_english.csv`, ~20 rows across 5–6 languages plus 5 English controls, for
  the flow tests and the spike. Hand-written, and kept apart from `labeled_eval.csv` (A).
- **Cost:** another full pass (12 layers, base size): benchmark it (C) and give it its own budget
  (≤ +20% over Phase 1). Disk +1.11 GB (fp32 on disk), memory ~0.56 GB in fp16.
- **Release:** its own minor version after Phase 1's 0.9.0 (0.10.0), with the README feature list
  and the description left unchanged (it's a guard, not a headline feature).

## Storage (agreed 2026-10-12)

Measured on the dev machine on 2026-10-12, with `huggingface_hub` 1.32.0.

### Where things are stored

| Layer | What | Managed by | Lifetime |
|---|---|---|---|
| Disk: `~/.cache/huggingface/hub` | Checkpoints, plus the `model.safetensors` the app converts and writes into the cached snapshot folder | `huggingface_hub`; the app only adds the converted file | Forever. Nothing in this project deletes from it |
| Memory: model weights | One fp16 copy per model | `@st.cache_resource`, one per server process, shared by all tabs | Until the server process stops |
| Memory: session data | `df`, `result_df`, `result_*`, widget keys | `st.session_state`, one set per browser tab | Until the tab closes or the server restarts. No size limit, no eviction |
| User files | Uploads and downloads | Streamlit, in memory; downloads are generated on click | Never written to disk on the server |
| CI | Model cache | `actions/cache`, key derived from the pinned revisions (0C, P1) | Until a pin changes |

### Measured today

- **SiEBERT uses 4.0 GB, not the ~1.4 GB the docs imply.**
  - `pytorch_model.bin` 1.42 GB, kept after conversion.
  - The converted `model.safetensors`, 1.42 GB.
  - A stale older revision (`a22773d…`) holding another 1.42 GB `model.safetensors`.
- **The emotion model** is the same pattern: 313 MB `.bin` + 313 MB converted copy.
- **`huggingface_hub` 1.32 keeps file contents in a shared store** (`hub/blobs/<xx>/…`), with each
  model's own `blobs/` entry being a symlink into it. So `du` on a single `models--*` folder
  **undercounts**. Measure with `du -sh ~/.cache/huggingface/hub` or `hf cache scan`.

### Footprint by phase (estimates; replace with measurements as each phase lands)

| Phase | New disk | Memory |
|---|---|---|
| Today | 2.84 GB (SiEBERT `.bin` + converted copy) | ~0.7 GB weights |
| 1. Emotion | +0.63 GB (`.bin` + converted copy). 0.31 GB only if S1's **optional** `.bin` deletion lands: S1 alone stops re-downloads, not the first download | +0.16 GB |
| F. Language | +1.11 GB (XLM-R's `model.safetensors` is fp32 on disk; checked 2026-10-12). Already safetensors, so no second copy | +0.56 GB (fp16) |
| 2. Topics | +0.34 GB (4-bit) to +1.2 GB (bf16) | similar |
| 3. LLM | +6 GB (9B 4-bit); +3 GB more if the 4B fallback is also cached | +6 GB, only after the first click |

### Work items

- **S1. Don't keep the `.bin` once a converted copy exists (Phase 1, commit 2 below).**
  - **Today:** `snapshot_download` asks for `model.safetensors`, `pytorch_model.bin` *and*
    `config.json` in one call, so the 1.4 GB `.bin` must stay in the cache or it is re-downloaded
    on every launch.
  - **Change:** split `_ensure_safetensors` into two downloads.
    1. First, `allow_patterns=["config.json", "model.safetensors"]`. This is cheap, and it
       resolves the **pinned** revision (0C, P1), so a pin bump is still picked up.
    2. Only if `model.safetensors` is then missing from that snapshot folder, download
       `pytorch_model.bin`, convert it, and atomically replace as today.
  - The keep-the-file-permissions logic and the `exists()` guard are unchanged.
  - **Then, optionally, delete the `.bin` after a successful conversion.** That means removing the
    snapshot symlink *and* the shared-store file it points to. Spike this first:
    - Content-addressed storage means another repo could reference the same file, so only delete
      it when nothing else links to it. If that can't be decided safely, skip deletion and
      document `hf cache` cleanup instead.
    - Check how `hf cache scan` and `hf cache verify` report a snapshot with a deleted file.
  - **Tests:**
    - `TestEnsureSafetensors` currently assumes one `snapshot_download` call. Update it to assert
      the two-step order, including that `.bin` is **never requested** when the converted file
      exists. That's the real fix, so mutation-test it.
    - `conftest.py`'s module-level `snapshot_download` mock must handle both calls.
    - Re-check that the CI `integration` cache (its key now derived from the pins, 0C P1) still covers this.
  - **Behaviour across a pin bump (0C, P1):** the new snapshot folder has no converted file,
    so step 2 downloads the `.bin` and converts once, and the previous revision's folder is left
    behind. That's how the stale `a22773d…` copy got there (back when `main` moved upstream).
    Covered by S2.
- **S2. Stale revisions:** **document, don't automate.** The cache is the user's and is shared with
  other tools.
  - The README gets a "Disk usage" note: how to measure (`hf cache scan`) and how to delete
    (`hf cache delete`, which lets you pick revisions interactively).
  - It also gives the real per-phase total from the table above.
- **S3. Per-tab memory:**
  - **The cost:** `result_df` is a full copy of `df` plus the generated columns, so each tab holds
    the uploaded file at least twice. Filters (B) must slice per rerun and **never** store the
    filtered frame in `session_state`.
  - **Measure:** extend `scripts/benchmark.py` (C) to report the memory of the frames a tab holds
    (e.g. `df.memory_usage(deep=True).sum()` across `df` and `result_df`) at 10k rows.
  - **Don't optimize yet:** storing only the generated columns and re-joining them at render time
    is the obvious optimization, but it touches every persisted-results invariant in CLAUDE.md.
    Only consider it if the measurement says it's needed.
- **S4. First LLM click (Phase 3):** it's a ~6 GB download.
  - Use `show_spinner=..., show_time=True` on its cached loader, as `load_model` does.
  - Check free disk space **before** starting (`shutil.disk_usage` on the cache folder), and show
    a clear `st.error` instead of failing partway.
  - Downloads that are interrupted resume from `huggingface_hub`'s `.incomplete` files, so no
    extra handling is needed. Confirm that during the Phase 3 spike.
- **S5. README / CLAUDE.md numbers:** replace "~1.4 GB" everywhere with the measured per-phase
  totals, and update CLAUDE.md's `--integration` "~1.4 GB on a cold cache" note.

## No database (yet) (decided 2026-10-12)

**None of the phases above needs a database. Don't add one as part of them.**

- **Today nothing persists:** results live in `st.session_state`, downloads are generated on click,
  and the app never writes user text to disk. Keep that as a deliberate property, not an accident.
- **Add a database only for a feature that genuinely needs persistence.** If one is proposed:

| Feature | Use | Why |
|---|---|---|
| Run history (results survive a restart), summary/filter queries | **DuckDB** (embedded, one file) | Reads and writes pandas frames directly; fast `GROUP BY` for D and B; one dependency, no server |
| Prediction cache (skip texts seen before) | **SQLite** (stdlib `sqlite3`) | Zero new dependencies; a simple lookup by key |
| Embedding search (Phase 2) | DuckDB `vss` or SQLite `sqlite-vec` | Stays embedded; a dedicated vector database is overkill at CSV scale |
| Multi-user / hosted deployment | Postgres (+ `pgvector`) | Only if the app stops being local and single-user |

- **Obligations that come with any of them:**
  1. **Opt-in, visible storage.** It's the first time user text touches disk, so signpost it in the
     UI, and add a "clear history" control that `_reset` does not replace.
  2. **Record the model revision on every stored prediction** (and the repo id). Otherwise
     results from different checkpoints mix without anyone noticing. A prediction cache's key is
     `(text hash, model repo, revision)`, never the text alone.
  3. **A connection or cursor per thread.** Streamlit runs each rerun on a fresh thread (the same
     reason `load_model` needs `mx.eval`). A connection shared via `st.cache_resource` must hand
     out a per-thread cursor (DuckDB: `con.cursor()`; SQLite: `check_same_thread` plus a lock, or
     one connection per thread).
  - Plus: keep the database file out of the repo (`.gitignore`), out of the Stop-hook fingerprint,
    and out of the HF cache folder.

## Non-MLX components (for orientation)

**All model *inference* runs on MLX. The project is not MLX-only.** Don't remove or "MLX-ify" any
of these without a plan of its own:

| Component | Runs on | Notes |
|---|---|---|
| SiEBERT, emotion inference | **MLX** (`mlx-transformers`) | The checkpoints are the authors' **PyTorch** checkpoints, converted at load time; they are not `mlx-community` builds |
| `.bin` → safetensors conversion | **PyTorch**, CPU | One-time, at load. It's why `torch` (~550 MB) is a required dependency (see CLAUDE.md) |
| Tokenizers | `transformers` / Rust `tokenizers`, CPU | Output converted to `mx.array` |
| Downloads | `huggingface_hub` | |
| Phase 2 embeddings | **MLX** | Base checkpoint via `mlx-transformers`, or an **MLX-native** `mlx-community` build (spike decides) |
| Phase 2 clustering | **MLX** if k-means on `mx` arrays (preferred); CPU if `scikit-learn` | |
| Phase 2 keyword labels (c-TF-IDF) | pandas / numpy, CPU | Small; no need to move to MLX |
| Phase 3 LLM | **MLX** (`mlx-lm`) | **MLX-native** `mlx-community` 4-bit builds |
| Spike reference (`transformers.pipeline`) | PyTorch | Used only to check MLX output during planning, 2026-10-12. Not part of the app or its tests |

- **Going MLX-only is possible but not planned.** It would mean pre-converting SiEBERT and the
  emotion model and loading third-party copies of the weights instead of the authors' repos. The
  ~550 MB `torch` dependency is the price of reading the original checkpoints directly.

## Risks and open questions

- **Accuracy is modest:** 66% on the card's own balanced 7-class eval (chance is 14%). The training data is Twitter, Reddit, self-reports and TV dialogue, so product and app reviews are out of domain. Show the confidence rather than hiding it, and consider wording the help text as "likely emotion".
- **Disagreements are expected, not bugs.** SiEBERT is positive/negative only, so `positive` + `neutral` or `negative` + `surprise` will be common. Do not "reconcile" the two outputs. A later phase could use those disagreements as a "review these rows" filter.
- **No license declared** on the model card (`cardData.license` is absent, which is also true of SiEBERT). The app downloads it at runtime rather than redistributing it, so its status is the same as today's. Note it in the README.
- Upstream ships only `pytorch_model.bin` (plus `tf_model.h5`, which `allow_patterns` already excludes). If upstream ever adds `model.safetensors`, the conversion is skipped once the pin (0C, P1) is bumped to that revision; until then nothing changes.
