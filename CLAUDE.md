# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SiEBERT MLX** (`siebert-mlx`) is a Streamlit application for sentiment classification in English text using SiEBERT (`siebert/sentiment-roberta-large-english`) on Apple Silicon with MLX. Users upload a CSV (or try the built-in sample data), select the text column, classify, and download results with "Sentiment" and "Confidence" columns. Guided step-by-step UI with auto-detected text columns, summary metrics, and a Reset button to start over.

## Commands

```bash
# Setup — torch/safetensors are direct production dependencies (mlx-transformers
# 0.3.0 dropped torch), so the lazy-import conversion path and tests/conftest.py's
# torch/safetensors patches both work after a plain sync.
uv sync

# Auth (model download) — read from env or a gitignored .env (via python-dotenv)
export HF_TOKEN=hf_...

# Run
uv run streamlit run streamlit_app.py

# Lint
uv run ruff check .

# Format
uv run ruff format .
uv run ruff format --check .   # CI uses --check (fails instead of rewriting)

# Type check
uv run ty check .

# Test
uv run pytest                                          # integration tests skipped
uv run pytest tests/test_streamlit_app.py              # unit tests
uv run pytest tests/test_app_flow.py                   # AppTest flow tests
uv run pytest tests/test_streamlit_app.py::test_name   # single test
uv run pytest --integration                            # + real checkpoint, ~1.4 GB on a cold cache
```

Use `ruff` for all linting and formatting. Run `uv run ruff check --fix .` to auto-fix lint issues. Use `ty` for type checking. Use `pytest` for unit testing.

When working with Python, invoke the relevant `/astral:<skill>` (`/astral:uv`, `/astral:ty`, `/astral:ruff`) to ensure best practices are followed.

## Architecture

Single-file application (`streamlit_app.py`, ~331 lines):

1. **`detect_text_column`** — returns first string- or object-dtype column name via `next()` generator (`pd.api.types.is_string_dtype` / `is_object_dtype`, matching the pandas 3.0 default `str` dtype)
2. **`_ensure_safetensors`** — downloads model via `snapshot_download` (prefers `model.safetensors`, falls back to `pytorch_model.bin`), converts to safetensors if needed; `torch` and `safetensors` are lazy-imported only when conversion is required. The conversion writes to a sibling `model.safetensors.*.tmp` and `os.replace`s it into place — the `exists()` guard means a truncated destination would be accepted forever, so the write has to be all-or-nothing; the `finally` unlink covers every failure path and is a no-op once `replace` has consumed the temp file
3. **`load_model`** — loads config via `AutoConfig`, constructs `RobertaForSequenceClassification`, loads weights via `from_pretrained` with `float16=True` (weight **memory** only — since mlx-transformers 0.3.0, `get_extended_attention_mask` hardcodes a `mx.float32` additive mask, which promotes every encoder activation, so `.logits` come back `float32` and compute is fp32 regardless; there is no input-side lever, `roberta.py` never passes `dtype`), then `mx.eval(model.parameters())` to materialize the lazy weights on the loading thread; cached with `@st.cache_resource`; authenticates with `HF_TOKEN`
4. **`process_dataframe`** — pre-filters blanks, batches valid texts (`BATCH_SIZE=8`), tokenizes with `return_tensors="np"` and converts to `mx.array`, classifies via softmax over logits; uses `.tolist()` for batch conversion; clears its `st.progress` bar (`progress_bar.empty()`) when done
5. **UI** — guided step-by-step flow: file upload or sample data → column auto-detect and preview → classify → metric cards + sentiment-distribution bar chart + styled results table → CSV download. Results are stored in `st.session_state` and re-rendered from there, so post-classify reruns don't re-run inference

## Key Patterns

- MLX for all inference on Apple Silicon (no device management needed)
- MLX streams are thread-local; Streamlit runs each rerun on a fresh thread, so `load_model` must `mx.eval(model.parameters())` to materialize the weights before caching. Skipping this leaves the float16 weights as pending ops bound to the loader thread's GPU stream, and the first classify (a later rerun thread) fails with `RuntimeError: There is no Stream(gpu, 0) in current thread.`
- `hf_logging.set_verbosity_error()` suppresses expected checkpoint warnings
- Confidence via `mx.softmax(logits, axis=-1)` with `mx.max` and `mx.argmax`; `mx.eval()` before `.tolist()`; labels from `model.config.id2label`
- Empty, whitespace-only, and missing (`NaN`) texts skipped; get sentiment `""` and confidence `0.0` (`fillna("")` before `astype(str)` coerces pandas 3.0 missing cells, which no longer stringify to `"nan"`)
- Tokenizer uses `return_tensors="np"` converted to `mx.array`, with `truncation=True` (512 token limit) and `padding=True`
- `process_dataframe` returns a copy; input DataFrame is not mutated
- `st.session_state` persists the loaded DataFrame (`df`/`source_name`) and the classification result (`result_df`/`result_col`) across reruns, so a post-classify rerun (Download click, theme toggle) re-renders from state instead of re-running inference; the result is invalidated when the selected column no longer matches `result_col`
- `Sample`/`Reset` are `on_click` callbacks (mutate `session_state` before the rerun, no `st.rerun()`); `Classify` still uses the `st.button`-returns-`True`-only-once-after-click pattern (`classify_clicked`)
- Uploaded files load once, guarded on `uploaded_file.file_id`, and the uploader has a dynamic `key=f"uploader_{...}"` that `Reset` increments — without both, the persisted uploader value re-reads on every rerun, undoing `Reset` and clobbering a `Sample` pick
- Walrus operator (`:=`) in UI guards to combine detect + check into one `elif`
- `SAMPLE_DATA_PATH` points to `samples/mixed_sample.csv` for the "Sample" button
- KPI metrics use `st.metric(..., border=True)` inside `st.container(horizontal=True)` (content-sized, wraps on narrow screens — prefer over `st.columns` for metric/button rows, which forces fixed ratios and can wrap a button label onto two lines); the distribution `st.bar_chart` and results `st.dataframe` sit in bordered `st.container(border=True)` cards with bold `st.markdown` labels; Material Symbols icons (`:material/...:`) on buttons/callouts and a `page_icon`
- Results `st.dataframe` uses `column_config` (`ProgressColumn` for `Confidence` as `format="percent"`, `TextColumn` for `Sentiment`) with `hide_index=True`; per-value `Sentiment` coloring is a Pandas `Styler.map` tint (formatting via `column_config`, coloring via Styler), skipped above `STYLE_ROW_CAP` rows so `st.dataframe` virtualization stays fast; the CSV download is built from the unstyled `result_df` first, so styling never reaches the file
- Custom `shadcn`-inspired theme in `.streamlit/config.toml`: shared typography/shape/semantic palette in `[theme]`, surface colors split across `[theme.light]` / `[theme.dark]` (+ matching `.sidebar` sub-tables) so the settings-menu light/dark toggle stays available (a single `[theme]` block would lock the app to one mode). Dark mode uses a blue `primaryColor` (`#3B82F6`) so white primary-button text stays readable; green/red sentiment tints are theme-neutral rgba so they read in either mode
- Dependencies managed by `uv` with lockfile (`uv.lock`); `streamlit>=1.58.0,<2.0` is a plain `[project.dependencies]` constraint (modern theming keys + `st.container(horizontal=True)` need the floor). It used to need `[tool.uv] override-dependencies` to break `mlx-transformers`'s exact streamlit pin, but 0.3.0 dropped the streamlit dependency entirely, so the override is gone and the direct constraint is the single lever. `[dependency-groups]` has only `dev` (`pytest`/`ruff`/`ty`, installed by default)
- `torch`/`safetensors` are **declared directly** because `_ensure_safetensors` imports both. Their transitive status differs and the distinction matters: `safetensors` still arrives via both `mlx-transformers` and `transformers`, while `torch` no longer arrives at all — `mlx-transformers` 0.3.0 dropped it, and its loader now rejects `.bin` checkpoints outright (`_discover_safetensor_files` raises "supports safetensors only"). Since `siebert/sentiment-roberta-large-english` ships only `pytorch_model.bin` on `main`, `_ensure_safetensors` must convert, so `torch` (~515 MB) is a hard production requirement until upstream ships safetensors. Check the real state with `uv tree`; never assume a transitive path holds, and never assume one that holds is a contract
- Ruff lint config (`[tool.ruff.lint]`): `select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]`, `ignore = ["E501"]` (line length owned by `ruff format`), `combine-as-imports = true` (keeps the multi-name `transformers` import in one block); `zip()` calls pass `strict=True` (B905)
- As of ruff 0.16, `ruff format` also formats Python code fences inside Markdown — `CLAUDE.md` and `README.md` are now gated by CI's `ruff format --check` alongside the 5 `.py` files (7 total). Currently a no-op (every fence is ```bash or ```bibtex), but a ```python snippet added to either file must be ruff-formatted or the docs-only change fails CI

## Tests

- `tests/conftest.py` — module-level patches for `RobertaForSequenceClassification`, `snapshot_download`, `torch.load`, and `safetensors.torch.save_file` to prevent model downloads and weight conversion during test collection. `snapshot_download` returns a module-scoped `TemporaryDirectory`, **not** `tests/`: `_ensure_safetensors` materializes a real `model.safetensors` there via `os.replace` even with `save_file` mocked, so pointing it at the repo would litter `tests/` on every collection and then make `exists()` skip the conversion on later runs. Also exposes the module-scoped `real_model` fixture, which lifts those patches *and* re-points `streamlit_app`'s own globals (bound to the mocks at its import) at the real objects, clearing `load_model`'s `st.cache_resource` entry on both sides so mocks cannot leak either way
- `tests/test_inference_integration.py` — the only non-mocked tests: pins `float16` weights + `float32` logits, the `id2label` mapping, three 4-decimal confidences (one deliberately long, so batching it with the short ones exercises heavy padding), and stability across batch boundaries. Marked `integration` and **skipped**, not deselected, unless `--integration` is passed — a `-m 'not integration'` in `addopts` would make `pytest tests/test_inference_integration.py` collect nothing and exit 5, breaking the per-file pattern the other suites use. `addopts = "--strict-markers"` turns a typo'd marker into a collection error rather than a live download inside the offline Stop hook. The dtype assertion is the real tripwire — fp16's spacing below 1.0 (2⁻¹¹ ≈ 4.9e-4) means rounded confidences can never reliably distinguish fp16 from fp32, so the pinned values instead guard against a changed checkpoint, tokenizer, softmax axis, or label mapping
- `tests/test_streamlit_app.py` — unit tests for `detect_text_column`, `_ensure_safetensors`, `load_model`, `process_dataframe` (incl. progress-bar clearing), `BATCH_SIZE`, `STYLE_ROW_CAP`, and `SAMPLE_DATA_PATH`; uses class-scoped `autouse` fixture for Streamlit mock in `TestProcessDataframe` and per-test decorator mocks for model loading
- `tests/test_app_flow.py` — end-to-end flow tests via `streamlit.testing.v1.AppTest`: initial render, Sample/upload load into `session_state`, selectbox label/help, Classify+Reset visibility, Reset clears state, results persist across reruns and invalidate on column change; relies on `conftest.py` patches so no network access. `APP_PATH` must stay **absolute** — as of streamlit 1.61 `AppTest.from_file` resolves a relative path against the calling file (`tests/`), not the working directory
- Simulate real uploads in `AppTest` with `at.file_uploader[0].upload(filename, content, mime_type)` — the value persists across reruns, so it can regression-test the reset-after-upload fix (upload → Reset → still cleared) and stale-upload-doesn't-clobber-`Sample`

## Hooks & CI

- `.claude/settings.json` wires local Claude Code hooks (scripts in `.claude/hooks/`) plus a `permissions.deny` rule for `Read(./.env)` — `protect-env.sh` only guards *writes*, and the likelier leak is reading `HF_TOKEN` into the transcript. The rule is scoped to exactly `./.env`, so `.env.example` stays readable; a future secret-bearing `.env.local` would need its own entry. Known and accepted limit: the rule binds the **Read tool**, so `Bash(cat .env)` still prints the token, and Grep coverage is unverified — pattern-matching a shell command cannot close that path honestly, and a bypassable guard invites more confidence than it earns:
  - **PreToolUse** `protect-env.sh` — denies tool edits to `.env`/`.env.*` (holds `HF_TOKEN`, gitignored); `.env.example`/`.sample`/`.template` stay editable
  - **PostToolUse** `ruff-fix.sh` — a formatter, not a gate: it reports nothing back and swallows every failure. `ruff format` + `ruff check --fix` on `*.py`, `ruff format` **only** on `*.md`. Markdown is in scope because ruff ≥ 0.16 formats Python fences inside `.md`, so CI's `ruff format --check .` gates CLAUDE.md/README.md too (7 files); `ruff check` is Python-only and reports "No Python files found" on `.md`, so the `.md` arm skips that second subprocess rather than spawning a provable no-op. Scoped to files under `$CLAUDE_PROJECT_DIR` — an edit to a scratch script elsewhere is not this repo's to reformat under this repo's ruff config
  - **Stop** `verify-on-stop.sh` — runs all four checks CI's `check` job runs, in CI's order (`ruff check .`, `ruff format --check .`, `ty check .`, `pytest -q`), each exiting 2 (with a `stop_hook_active` loop guard) so any of them blocks finishing. `ruff check --fix` in the PostToolUse hook **cannot** stand in for `ruff check .` here: it applies only *safe* fixes and swallows the exit code, so unfixable `B`/`SIM`/`RUF` violations survive it silently and only CI would catch them
  - The Stop gate triggers on a **content fingerprint** (`shasum` over tracked + untracked-not-ignored `*.py`, `pyproject.toml`, `uv.lock`, `samples`), compared against `.claude/.verified` (gitignored) and rewritten only after all four gates pass — so a red turn is re-checked on the next Stop even if the follow-up edits never touch a verifiable file. Costs ~0.02s, so an unchanged turn skips the whole gate in ~0.05s instead of ~9s. Two rejected alternatives, both tried: a marker file `touch`ed by `ruff-fix.sh` is blind to everything a PostToolUse formatter never sees (a `.py` rewritten by Bash/`sed`, `git checkout`/`revert`/`stash pop`, and dependency edits all skip the gate) while over-firing on out-of-tree scratch files; and `git status` compares against HEAD, so an uncommitted work-in-progress tree reads dirty all session and every conversational turn pays full price. `pyproject.toml`/`uv.lock` are in scope because a dependency bump is exactly how this repo last shipped a silent behavior change (`dba72ff`), and `samples` because the suite reads `mixed_sample.csv` and `blank_cells.csv`
  - The whole-project checks live on **Stop, not PostToolUse**, deliberately. `ty check .` used to run per-edit, which meant N-1 redundant full-project runs per multi-file turn *and* an exit 2 on the transient cross-file breakage that is normal mid-refactor. Per-edit hooks should assert invariants that hold after every single edit (formatting is idempotent and file-local); cross-file consistency is false by construction until a refactor finishes
- `.github/workflows/ci.yml` mirrors the hooks on `macos-latest` (mlx ships arm64-only wheels): the `check` job runs `uv sync --frozen` (`--frozen` fails on a stale `uv.lock`), then `ruff check`, `ruff format --check`, `ty check`, `pytest -q` — all offline. A second `integration` job (`needs: check`, `timeout-minutes: 30`) runs `pytest --integration -m integration` against the real checkpoint, with `~/.cache/huggingface/hub` cached (covers both the download and the safetensors conversion). No `HF_TOKEN` secret is *required* — the model is public — but one is passed through if present, since anonymous downloads from shared Actions IPs are what get rate-limited; `load_model`'s `or None` turns the empty value of a missing secret back into an anonymous request. The cache key is exact and static, so it never self-refreshes: bump its `-v1` suffix if upstream moves `main` to a new revision
  - Workflow-level `permissions: contents: read` in both workflows; only the job that publishes re-grants itself `contents: write`. `concurrency` keeps a plain `cancel-in-progress: true`: making it PR-only does **not** guarantee every `main` commit gets a run, because [GitHub cancels any previously *pending* run in the group regardless of that setting](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency) (only `queue: max` changes that), and it would serialize every `main` push behind the 30-minute `integration` job. The `release` job's tag check makes a superseded run harmless anyway
  - Action versions are deliberately non-uniform: `actions/checkout@v7` and `actions/cache@v6` track floating major tags, but `astral-sh/setup-uv` is pinned to the **exact** `@v10.0.1` because upstream stopped publishing major/minor tags at v8 (citing the tj-actions supply-chain attack), so `@v10` does not exist. That pin needs manual bumping. `prune-cache` is left at v10's `false` default rather than restored to the pre-v9 `true`: `uv cache prune --ci` strips pre-built wheels and keeps only wheels built from source, and every dependency here ships a pre-built wheel — so `true` persisted an almost-empty cache and re-downloaded the set every run, which is exactly why upstream flipped the default
  - `cache-dependency-glob` is deliberately **not** set. The action's default is a 7-pattern list that already includes `**/pyproject.toml` and `**/uv.lock` (true since v6), so the `cache-dependency-glob: uv.lock` this repo used to carry *narrowed* the key by dropping `pyproject.toml` — a pyproject-only change (a new dependency group, a changed `requires-python`) could then restore a stale cache. `enable-cache: true` **is** still explicit, because the `auto` default disables caching on tag-push and release events
- A third `ci.yml` job, `release` (`needs: [check, integration]`, `ubuntu-latest`), publishes a GitHub Release the moment a bumped `[project].version` lands green on `main` — gated on both jobs, so a release is never cut from unverified code. Its `if:` asserts `github.ref == 'refs/heads/main'` explicitly rather than inheriting main-only from `on.push.branches`, so widening that trigger later cannot leak a public release out of an unreviewed branch. It reads the version with `uv version --short` (no sync needed; it only parses `pyproject.toml`, verified against a project with no `.venv` and no lockfile)
  - Idempotent on **tag existence**, not on a `git diff HEAD^`: a diff misses a bump arriving behind a merge commit's second parent and re-fires on a workflow re-run. The *tag* rather than the release, because deleting a release to re-cut it by hand leaves the tag behind, and that tag is the signal for the job to stay out of the way
  - The tag list is assigned to a variable **before** being tested, never piped straight into `grep`: a transient API error must abort the job under `set -e` rather than read as "no tag found" and fall through to an unconditional `gh release create`. Fail closed, not open
  - `--target "$GITHUB_SHA"` deliberately tags the commit *this run verified*, which need not be the commit that bumped the version — if the bump's own run was superseded, the release is cut from the newer tested tree rather than an untested one. A PEP 440 pre-release (`0.8.0rc1`, `0.9.0.dev1`) gets `--prerelease` so it cannot take the "Latest" slot or the README's `sort=semver` tag badge
  - `gh release create --target` creates the tag itself and `--generate-notes` builds notes server-side, so a shallow checkout and no tag push are enough
- `.github/workflows/release.yml` (`on: push: tags: v*`) is now only the **manual escape hatch** — re-cutting a deleted release, or tagging an older commit. It cannot double-publish against the `release` job: that job's tag is created with the `GITHUB_TOKEN`, and [GitHub does not start a workflow run from a `GITHUB_TOKEN` event](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow) (only `workflow_dispatch`/`repository_dispatch` are exempt). That suppression is the *only* thing keeping the two paths disjoint — swapping in a PAT or App token to chain them would need one of the two paths removed first

## Sample Data

- `samples/mixed_sample.csv` — 20-row sample (4 from each domain), loaded by the "Sample" button via `SAMPLE_DATA_PATH`
- `samples/product_reviews.csv` — 40 e-commerce product reviews
- `samples/movie_reviews.csv` — 40 film and TV opinions
- `samples/social_media.csv` — 40 tweets and social media posts
- `samples/restaurant_reviews.csv` — 40 dining and food service reviews
- `samples/app_reviews.csv` — 40 mobile/web app store reviews
- `samples/blank_cells.csv` — 10-row edge-case sample with an `id` column and missing/whitespace text cells, to exercise blank-skipping (including the pandas 3.0 `NaN` path that `fillna("")` guards against)
