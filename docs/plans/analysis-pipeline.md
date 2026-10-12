# Plan: an analysis pipeline around SiEBERT

Status: **drafted 2026-10-12, not started.** Implement in a fresh session; read this file and
`CLAUDE.md` first.

Three phases, each merged before the next starts:

1. **Emotion detection.** Designed in full below, and ready to implement.
2. **Topic clusters.** Outline only; see "Later phases".
3. **LLM explanations and aspects.** Outline only; see "Later phases".

---

# Phase 1: emotion detection

## Goal

Every Classify run adds two columns next to `Sentiment` / `Confidence`: **`Emotion`** (one of
`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) and **`Emotion confidence`**.
Results show an emotion distribution, and the download carries all four generated columns.

## Decisions (made with the user, 2026-10-12)

| Decision | Choice | Why |
|---|---|---|
| Model | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | Same `RobertaForSequenceClassification` class SiEBERT uses, so it reuses the whole MLX load path; 82M params, ~160 MB in fp16 |
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

- Add `EMOTION_MODEL_PATH = "j-hartmann/emotion-english-distilroberta-base"`, and move SiEBERT's path to a `SENTIMENT_MODEL_PATH` constant.
- Extract the body of `load_model` into an **uncached** `_load_classifier(model_path)` that returns `(model, tokenizer)`. Keep the `mx.eval` and its comment there; it applies to every model.
- `load_model()` keeps its name, signature and decorator, so conftest, the `real_model` fixture and the CLAUDE.md references still hold. Add a sibling `load_emotion_model()` with `@st.cache_resource(show_spinner="Loading emotion model...", show_time=True)`.
- Call site: `emotion_model, emotion_tokenizer = load_emotion_model()` directly after `model, tokenizer = load_model()`. Same eager-below-the-chrome placement, for the same reasons (see CLAUDE.md's UI bullet).
- `_ensure_safetensors` is already generic in `model_path`, so no change. Reword its "1.4 GB" comment so it doesn't read as SiEBERT-only.
- Use a **separate tokenizer** per model, even though the ids are identical today. Tokenizing is negligible next to the encoder, and sharing one would silently break if either model were swapped.

### 2. Inference

Keep `process_dataframe`'s 2-tuple contract **unchanged**, so the existing unit tests stay a pure safety net for the refactor:

- Extract `_valid_sorted(texts) -> list[(idx, text)]`, the blank filter plus the length sort.
- Extract `_classify_texts(valid, model, tokenizer, progress_bar) -> (labels, confs)`, the batch loop (tokenize, `return_dict=True`, softmax/max/argmax, `mx.eval`, `cast` on `.tolist()`, write-back by `idx`).
- `process_dataframe` calls both. Its output must be byte-identical; the existing suite proves it.
- New `add_emotions(result, source_df, text_column, model, tokenizer) -> (DataFrame, EmotionColumns)`:
  - Returns a copy of `result` with the two emotion columns appended.
  - Blanks get `""` / `0.0`, like sentiment.
  - Shows its own `st.progress(0, text="Detecting emotions...")` and empties it when done. Also give `process_dataframe`'s bar a `text=` so the two bars can be told apart.
- New `EmotionColumns(emotion, confidence)` NamedTuple. Resolve names with `_unique_column_name` against `taken = set(source_df.columns) | set(generated_cols)`:
  - Constants `EMOTION_COL = "Emotion"` and `EMOTION_CONFIDENCE_COL = "Emotion confidence"`.
  - A source `Emotion` column is preserved and the model's becomes `Emotion (model)`, same rule and direction as sentiment.
  - Check that `Emotion confidence` doesn't fall inside `Emotion`'s name space (it doesn't; the prefix test is `Emotion (model)`), and add a unit test for it.
- Use a separate NamedTuple rather than widening `GeneratedColumns`. `_render_results` unpacks the pair positionally and the flow tests hand-seed 2-tuples, so widening it would ripple through every one of those for no gain.

### 3. State

- The classify branch runs `add_emotions` right after `process_dataframe`, under the same spinner. It stores the resulting frame as `result_df` and the `EmotionColumns` as **`result_emotion_cols`**.
- `_clear_results` pops `result_emotion_cols` with the other three `result_*` keys.
- Make `"result_emotion_cols" in st.session_state` part of the render guard, for the same file-watcher-across-a-code-change reason that `result_generated_cols` is. This is the exact scenario that ships with this change: a running dev session holding pre-emotion results.
- `_render_results(result_df, source_name, generated_cols, emotion_cols, announce=False)`. `emotion_cols` is **required**, with no default, so `ty` flags every missed call site.

### 4. UI (`_render_results`)

- **Rename notice:** add `(EMOTION_COL, emotion_col)` and `(EMOTION_CONFIDENCE_COL, emotion_conf_col)` to the `renamed` filter. The wording ("the model's output was added as …") still reads correctly.
- **Results table:**
  - `Emotion` gets a `TextColumn`; `Emotion confidence` gets a `ProgressColumn(format="percent", color="blue")`. Both are `pinned=True` and excluded from `_width_caps`.
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

The two new pinned columns add about **+228 px to S**, the sum of the grid's base column widths (≈78 for `Emotion`, 150 for the progress column; measure, don't trust this). With the sidebar open:

- The Sample results' overflow onset moves from **1022 to ≈1250**, and the labeled `text,Sentiment` shape's from 1148 to ≈1376.
- At 1440 the Sample still fits (S ≈ 756 against a 946 grid). At 1100 every results shape overflows. The scroll moves only the user's columns, because all four generated columns are pinned.
- Mitigation to try first: `width="small"` on the emotion `ProgressColumn` (and maybe the sentiment one). Measure both.
- Re-measure all five shapes listed in CLAUDE.md's "Why the length gate was needed" bullet with headless Playwright (`scrollWidth` against `clientWidth`). Then rewrite that bullet and the geometry bullet with the new numbers.

## Tests

Order matters. Land the refactor **first, with the existing suite unchanged and green**, before any emotion code exists.

- **Unit (`tests/test_streamlit_app.py`):**
  - Generalize `_make_mock_model` to take an `id2label` (currently 2-class). Add a 7-class pair.
  - `add_emotions`: labels follow the text, not the row (sorted batching); blanks get `""`/`0.0`; input not mutated; returns a copy.
  - `EmotionColumns` naming: plain names, a source `Emotion` column, a source `Emotion (model)` column, and both sentiment and emotion colliding at once.
  - `load_emotion_model` mirrors the existing `TestLoadModel` tests (path, `float16=True`, `mx.eval`, token).
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

- **Bump the `integration` job's hub cache key suffix `-v1` → `-v2`.** The key is exact and static, so the existing cache hits, never saves, and without the bump every run would re-download and re-convert the emotion checkpoint.
- Bump `[project].version` to `0.8.0` **only in the final commit, once everything is green**. The `release` job publishes a GitHub Release as soon as a bumped version lands green on `main`.

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
3. `feat: load the emotion model and add add_emotions` + unit tests.
4. `feat(ui): render emotion columns and distribution` + state, guard, flow tests.
5. `feat(ui): sidebar filters for sentiment and emotion` (B) + flow tests.
6. `feat(ui): summary export` (D) + the shared counts helper.
7. `test: add the labeled eval set and pin emotion inference against the real checkpoint` (A) + CI cache-key bump. The user reviews the labels before this lands.
8. `docs: emotion detection; disk usage (S2, S5); re-measure grid geometry; recapture screenshot; record eval, benchmark and per-tab memory numbers (S3)`.
9. `chore: bump version to 0.8.0` + the repo description and topics below, **in the same commit**, so no copy advertises emotion detection before it's on `main`.

### Repo description and topics (at release)

The description lives in **three places** that must match: `pyproject.toml` `description`,
README line 8 (the sentence under the badges), and GitHub's About.

- **All three were synced on 2026-10-12** to the pre-emotion text:
  - description: "Classify the sentiment of English text in CSV files, locally on Apple Silicon. A
    Streamlit app running SiEBERT (RoBERTa-large) with MLX that adds sentiment and confidence
    columns to your data." The README's copy links SiEBERT to its model card.
  - GitHub topics: added `nlp`, `huggingface`, `csv`, `macos`, `on-device-ai`, `python`, for 14 in
    total.
- **In commit 9, set all three to:**
  > Classify the sentiment and emotion of English text in CSV files, locally on Apple Silicon. A
  > Streamlit app running SiEBERT and an emotion model with MLX that adds labels and confidence
  > scores to your data.
- **Update GitHub** only *after* commit 9 is merged and green, since it's outward-facing:
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

- **Why:** today the only evidence for any prediction is each model card's own claim, and the
  emotion model's 66% was measured on tweets, Reddit and dialogue, not on reviews like the ones
  this app is fed.
- **Data:** `samples/labeled_eval.csv`, about 60 rows drawn from the existing domain samples (12
  per domain), with columns `text, emotion` (and `topic` in phase 2).
  - **The labels must come from a human.** Claude may draft them, but a model's labels used to
    grade a model measure agreement between them, not accuracy. The user reviews every row before
    it counts.
  - It does **not** need a `Sentiment` column. If one is added, it exercises the collision path on
    purpose.
- **Tool:** `scripts/eval_quality.py`, run by hand like `docs/capture_screenshot.py`. It runs the
  real models and prints accuracy, macro-F1 and a confusion matrix.
- **Record the numbers** in CLAUDE.md, with the date and the versions used.
- **Gate:** the models are deterministic, so an integration test can pin the **exact** number of
  correct rows. That catches a changed checkpoint, tokenizer or label mapping without thresholds
  that can't decide whether a 1-point move is noise.
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
  keyed, so a filter set on one file doesn't silently carry into the next.
- **Download stays whole-file, always.** Its meaning never changes: the source plus the generated
  columns. Making it follow the filter was rejected: a user who forgets a filter is on gets 30 rows
  of a 2,000-row file, silently.
- **Filtered download (decided 2026-10-12): a conditional second button.**
  - **When:** it renders only while a filter actually narrows the table (filtered rows < total).
    With no filter set, the header row is identical to today's, so it costs width only when it's
    useful.
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
- **Mechanics:** a second `st.download_button` in the Results header row, with the same lazy
  `data=` callable and `on_click="ignore"`, and its own key (e.g. `download_summary`). Name the
  file `<source>_summary.csv`.
- **Not in the all-blank arm:** there is nothing to summarize there.
- **Single source of truth:** compute the counts in one helper shared with the metric cards and the
  charts, so the export can never disagree with what's on screen. Pin that with a test comparing
  the two.

## Storage (agreed 2026-10-12)

Measured on the dev machine on 2026-10-12, with `huggingface_hub` 1.32.0.

### Where things are stored

| Layer | What | Managed by | Lifetime |
|---|---|---|---|
| Disk: `~/.cache/huggingface/hub` | Checkpoints, plus the `model.safetensors` the app converts and writes into the cached snapshot folder | `huggingface_hub`; the app only adds the converted file | Forever. Nothing in this project deletes from it |
| Memory: model weights | One fp16 copy per model | `@st.cache_resource`, one per server process, shared by all tabs | Until the server process stops |
| Memory: session data | `df`, `result_df`, `result_*`, widget keys | `st.session_state`, one set per browser tab | Until the tab closes or the server restarts. No size limit, no eviction |
| User files | Uploads and downloads | Streamlit, in memory; downloads are generated on click | Never written to disk on the server |
| CI | Model cache | `actions/cache`, exact static key | Until the `-vN` suffix is bumped |

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
| 1. Emotion | +0.63 GB → 0.31 GB once S1 lands | +0.16 GB |
| 2. Topics | +0.34 GB (4-bit) to +1.2 GB (bf16) | similar |
| 3. LLM | +6 GB (9B 4-bit); +3 GB more if the 4B fallback is also cached | +6 GB, only after the first click |

### Work items

- **S1. Don't keep the `.bin` once a converted copy exists (Phase 1, commit 2 below).**
  - **Today:** `snapshot_download` asks for `model.safetensors`, `pytorch_model.bin` *and*
    `config.json` in one call, so the 1.4 GB `.bin` must stay in the cache or it is re-downloaded
    on every launch.
  - **Change:** split `_ensure_safetensors` into two downloads.
    1. First, `allow_patterns=["config.json", "model.safetensors"]`. This is cheap, and it still
       resolves the *current* `main` revision, so a new upstream revision is still picked up.
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
    - Re-check that the CI `integration` cache (key bumped in commit 7) still covers this.
  - **Behaviour across a new upstream revision:** the new snapshot folder has no converted file,
    so step 2 downloads the `.bin` and converts once, and the previous revision's folder is left
    behind. That's how the stale `a22773d…` copy got there. Covered by S2.
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
- Upstream ships only `pytorch_model.bin` (plus `tf_model.h5`, which `allow_patterns` already excludes). If upstream ever adds `model.safetensors`, the conversion is skipped automatically.
