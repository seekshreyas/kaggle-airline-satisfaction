# Plan 01: Feature Engineering & Modeling Pipeline

| | |
|---|---|
| **Status** | Phases 0–4 complete · first pipeline submission scored 0.94845 (public LB) |
| **Revised** | 2026-10-17 |
| **Competition** | [playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10/overview), Predicting Airline Satisfaction |
| **Metric** | ROC AUC on predicted probabilities |
| **Deadline** | 2026-10-31 |
| **Working notebook** | `notebooks/bruce_v1.py` |

---

## 1. WHY (Objective)

**Problem.** Predict whether a passenger is satisfied (`satisfaction`, 44.4% positive) from
21 features: demographics, trip details, 13 service ratings (0–5) and delays.

**Goals**
1. **Learn what matters.** Find out which feature engineering moves the needle, while
   keeping the model simple and fixed (logistic regression), so every gain can be
   explained.
2. **Build once, reuse.** End up with an end-to-end pipeline (raw CSV → CV score →
   submission) whose code lives in `src/` and can be cloned for the next competition.
3. **Spend the test set sparingly.** Kaggle's test set is the final exam, not a tuning tool.

**Principles**
- **Change one thing at a time.** First features (model fixed), then the model (features
  fixed), then the full data.
- **Small data first.** Iterate on a subset so each loop takes seconds, not minutes.
- **Keep a private holdout.** Set aside a slice of `train` that's never used while
  iterating. It stands in for the test set.
- **Notebook first, then promote.** Prototype a step in `bruce_v1.py`. Once it works on
  real data, move it into `src/` with a test, and the cell becomes a one-line call.

**Success looks like**
- A leaderboard showing which feature ideas helped, and by how much.
- A validated submission CSV from the final pipeline, submitted by 2026-10-31.
- `pixi run test` passes.

---

## 2. WHAT (Requirements)

Tick a box when the item is done **and** its check has actually run.

### Phase 0 · Setup
- [x] Use `notebooks/bruce_v1.py` as the pipeline notebook
- [x] Extend `normalize_columns()` with reusable naming styles (`simple`, `snake_case`),
      overrides, collision checks and a CLI → `src/datahelpers/datawrangle.py`
- [x] Track `reports/experiments/leaderboard.csv` in git (`.gitignore` exception)
- [x] `pixi run test` task, plus tests for column naming
- [x] Test inputs are declarative YAML (`tests/cases/*.yaml`); test scripts are generic runners
- [x] Add `pyyaml` to `pixi.toml` explicitly (manifest + lockfile updated)
- [x] Draft config YAMLs for review: `configs/competition.yaml` + one file per
      iteration step in `configs/experiments/` (k-fold CV with `n_splits: 4`)
- [x] Salted-hash randomization for holdout, dev subset and folds (`randomization.salt`
      in `competition.yaml`) → `src/datahelpers/splits.py` + YAML test cases
- [x] Review and approve the config drafts

### Phase 1 · Data and splits
- [x] Load train, test and sample submission with DuckDB, and snake_case the columns
      → `load_competition_data()`, CLI `python -m src.datahelpers.dataloader profile`
- [x] Split off a fixed **holdout** (20% of train, stratified, salted hash). Ids aren't
      saved: the hash recomputes the same split in ~2 s
- [x] Draw a fixed **dev subset** (25% of the rest, stratified) for fast iteration
- [x] Sanity check: row counts, target rate in each split, nulls (`bruce_v1.py` Phase 1)

### Phase 2 · Feature engineering (model fixed, dev subset)
- [x] `lr_00` baseline: impute + scale numerics, ratings as numbers, one-hot categoricals → **0.92028**
- [x] `lr_01` ratings as one-hot categories → **0.94773, keep (current winner)**
- [x] `lr_02` ratings as numbers + `is0` / `ge4` flags (alternative to 01) → 0.94203, loses to 01
- [x] `lr_03` delays: missing flag, `log1p`, delay difference → +0.00010, revert
- [x] `lr_04` splines on age and flight distance → +0.00019, revert
- [x] `lr_05` crosses: class × type of travel, class × customer type → +0.00000, revert
- [x] `lr_06` rating aggregates (mean, min, max, counts, group means; zeros excluded) →
      **0.94825, +0.00052, 4/4 folds better → keep (current winner)**
- [x] `lr_07` crosses: type of travel × {wifi, online boarding} → +0.00032 (4/4 folds), below
      min_gain → revert. Personal Travel AUC only 0.800 → 0.802
- [x] **Keep rule decided:** gain ≥ 0.0005 and > std of per-fold deltas vs the parent
      (`keep_rule.noise: paired_std`; fold std is 10–80× larger because folds are shared)
- [x] **Diagnostics for every run** (notebook section + `python alfred.py diagnose <name>`
      → PNGs in `reports/plots/<run_id>/`): ROC, calibration, score distribution, confusion, per-fold
      AUC, AUC by segment, coefficients, each overlaid on a comparison run
- [x] **Executive deck for Phase 2** (Deckset markdown): `pixi run report
      configs/reports/phase2_feature_engineering.yaml` → `reports/phase2_feature_engineering.md`
      + PNG charts in `reports/plots/`. All narrative numbers come from the saved runs
- [x] Pick the winning feature set → **`lr_06_rating_aggregates` (97 features, 0.94825)**

### Phase 2b · Visualization library (`src/viz/`)
- [x] Split `src/modeling/plots.py` into themed modules under `src/viz/` (see §3.9)
- [x] One shared theme (validated palette, base styling) and one export module (slide PNGs)
- [x] Plot functions take data (tidy DataFrames or `RunDiagnostics`) and return a
      `go.Figure`; no file I/O, no dependency on `src.reporting` (`drivers_figure` now takes
      a `label` function)
- [x] Catalog of every public plot (`src.viz.catalog()`, 12 plots), so new plots are discoverable
- [x] Smoke test that builds every catalogued plot from synthetic data (`tests/test_viz.py`,
      inputs in `tests/cases/viz.yaml`); it fails if a `*_figure` function isn't catalogued
- [x] Move callers (notebook, `alfred.py`, report builder) to `src.viz`; remove
      `src/modeling/plots.py`; regenerate the Phase 2 deck and check the PNGs are unchanged
      (all 7 PNGs and the markdown are byte-identical)

### Phase 3 · Model iteration (features fixed, dev subset)
- [x] Regularization grid: `C ∈ {0.01, 0.1, 1, 10}` × {L2 (`lbfgs`), L1 (`liblinear`)}.
      Elastic net (`saga`, `l1_ratio` 0.5) was dropped: one grid point took >10 min on the dev
      subset. `model.grid` now accepts a list of dicts so each penalty gets a compatible solver
- [x] Diagnostics: grid chart (`viz.grid_figure`), plus the per-run diagnostics. Result: the
      model is saturated. `C ≥ 1` changes nothing, `C = 0.01` costs 0.0005–0.001, L1 at best
      ties L2 (+0.00001) and is 5–7× slower
- [x] Pick the final model settings → **`C = 1`, L2, `lbfgs`** (simplest, fastest)

### Phase 4 · Full data and ship
- [x] Re-run the winner with CV on the full training data (holdout still excluded) →
      **0.94925 ± 0.00039** on 559,708 rows (`fold_full`, own salt)
- [x] Score the holdout **once** and compare it with CV → **0.94880** (−0.00045 vs CV, about one
      fold std: no sign of overfitting the folds). A guard refuses to score it again
- [x] Refit on all labelled data, predict test, and validate the submission file →
      `data/submissions/20261017-231328_lr_final_full.csv` (299,844 rows, valid)
- [x] Submit to Kaggle (manual, after your go-ahead) and record the public LB score →
      **0.94845** public (CV 0.94925 · holdout 0.94880: consistent)

### Reusable code (promoted from the notebook as each phase completes)
- [x] `src/utils`: project root, YAML config loading (`extends`, `@refs`, hash), logging
- [x] `src/datahelpers`: data loader, profile, holdout / dev / fold splits
- [x] `src/features`: registry of feature blocks + pipeline builder
- [x] `src/modeling`: model registry, metrics, CV with OOF, experiment runner, diagnostics
      (incl. paired per-fold deltas)
- [x] `src/viz`: themed plotting library (Phase 2b)
- [x] `src/reporting`: Deckset markdown builder, plain-language labels, Phase 2 deck
- [x] `src/submission.py`: build and validate a submission (+ CLI `validate`)
- [x] `alfred.py` CLI: `cv`, `leaderboard`, `ladder`, `features` (+ pixi tasks)
- [x] `alfred.py` CLI: `train`, `submit` (submit validates, then asks y/N)
- [x] Tests: config merging, features (no leakage, same columns on train and test),
      CV (folds are reproducible)
- [x] Tests: submission validation, grid expansion, leaderboard column alignment

### Out of scope (for now)
Tree and boosting models, target encoding, Optuna, blending or stacking, MLflow, and
adding the original 129k dataset. The design leaves room for each.

---

## 3. HOW (Execution)

### 3.1 Data splits

```mermaid
flowchart LR
    T["train.csv<br/>699,635 rows"] --> H["🔒 holdout 20%<br/>~140k · scored once in Phase 4"]
    T --> R["rest 80%<br/>~560k"]
    R --> D["dev subset 25%<br/>~140k · Phases 2–3"]
    R --> F["full 80%<br/>Phase 4 CV"]
    K["test.csv<br/>299,844 rows"] --> S["Kaggle submission<br/>Phase 4 only"]
```

- **Randomization is a salted hash, not a random seed.** Each row gets
  `BLAKE2b("<salt>/<purpose>:<id>")`, with purpose = holdout, dev or cv. Rows are ranked by
  that hash within each target class, which gives exact stratification. The same salt
  gives identical splits whatever the row order; a new salt reshuffles everything. The salt
  lives in `randomization.salt` in `competition.yaml`. `seed` is only the model's
  `random_state`.
- Every iteration step uses stratified **k-fold CV, k = 4** (`cv.n_splits`; an experiment
  can override it). **Same salt means same folds** for every experiment, so scores are
  comparable.
- Check it with `pixi run python -m src.datahelpers.splits summary configs/competition.yaml`
  (optionally `--salt <new>`).
- The final model is refit on all 699,635 labelled rows before predicting test.

### 3.2 Notebook-first workflow

For each step: prototype a cell in `bruce_v1.py`, check it on real data, then move the
code into `src/` with a test and replace the cell with a call to it. The notebook stays a
short storytelling document (slides: problem → data → features → CV → diagnostics →
leaderboard → submission). Expensive cells sit behind `mo.ui.run_button`.

### 3.3 Configuration

Each experiment is a YAML file that `extends` its parent and only states what changed.
Dicts merge, lists replace, and the child wins. `derive` blocks and `encode` groups are
**keyed by name**, so a child can add one block or override one group without restating
the rest. The fully merged config is saved with each run.

```
competition.yaml            shared: data, schema, splits, cv (k=4), keep rule
└─ lr_00_baseline           Phase 2: features (model fixed, dev subset)
   ├─ lr_01_ratings_onehot
   │  └─ lr_03_delays        (extends the 01/02 winner)
   │     └─ lr_04_splines → lr_05_crosses → lr_06_rating_aggregates → lr_07_rating_crosses
   │                           └─ lr_08_regularization   Phase 3: model grid (dev subset)
   │                              └─ lr_final_full       Phase 4: full data + holdout + submit
   └─ lr_02_rating_flags
```

```yaml
# configs/experiments/lr_04_splines.yaml: only the continuous group changes
name: lr_04_splines
extends: lr_03_delays.yaml
encode:
  continuous:
    steps:
      - impute: {strategy: median}
      - log1p:  {cols: [departure_delay_in_minutes, arrival_delay_in_minutes]}
      - spline: {cols: [age, flight_distance], n_knots: 6, degree: 3}
      - scale:  {method: standard}
```

`configs/competition.yaml` holds what's competition-specific: data paths, id and target
columns, metric, salt and seed, column naming (`snake_case`) and the column schema (continuous,
ordinal, categorical), rating groups, splits, CV settings and the keep rule. It's the
only file you rewrite for a new competition. `tests/test_config_files.py` checks that every
config parses, every `extends` target exists, the schema matches the real headers, and
every `@ref` resolves.

### 3.4 Features

Two kinds of feature blocks, kept apart on purpose:

| Kind | What it does | Leakage | Examples |
|---|---|---|---|
| **derive** | Stateless, row by row; adds columns | Can't leak (learns nothing) | `missing_flag`, `zero_flag`, `ge_flag`, `cross`, `delay_features`, `rating_aggregates` |
| **encode** | Learns from data (medians, scalers, knots, categories) | Fit inside each CV fold | `impute`, `log1p`, `clip`, `spline`, `scale`, `onehot` |

Blocks register themselves by name (`@register_derive("zero_flag")`), so a YAML file can
use them without any code changes. The builder turns a config into an sklearn
`Pipeline(derive → ColumnTransformer(encode) → model)` with pandas output, which keeps
feature names readable in the coefficient tables.

### 3.5 Feature ladder: evidence and keep rule

Each rung comes from an EDA observation (satisfaction rate on train):

| Rung | EDA evidence | Why a linear model needs help |
|---|---|---|
| 01 / 02 | Wifi 0→**89%**, 1→39%, 3→28%, 5→95%. Online boarding 0→62%, 3→13%, 4→64%. | Non-monotonic; 0 means N/A; step between 3 and 4 |
| 03 | Departure delay 0→44.8% vs >0→40%; heavy right skew; 292 nulls in arrival delay | Skew and missing values |
| 04 | Age is an inverted U (40s–50s ≈ 60%, 70s ≈ 10%). Distance jumps at ~1,500 mi (≈33% → 62%). | A straight line can't follow either |
| 05 | Business class + business travel 73%; business class + personal travel 11% | Interaction effect |
| 06 | 13 weak, correlated ratings | An overall-experience summary is easier for LR to use |

**Keep rule:** a rung stays if CV AUC improves by **≥ 0.0005** and the gain is larger
than the std of its per-fold differences from the parent (same folds, so fold difficulty
cancels). Otherwise it's reverted. Rungs 01 and 02 compete; the
next rung builds on the winner (or on both, if they combine well).

### 3.6 Model

`LogisticRegression` with `C` and `l1_ratio`. In sklearn 1.9, `penalty` is deprecated:
use `l1_ratio=0` for L2 (`lbfgs`, ~6 s per CV on the dev subset), `l1_ratio=1` for L1
(`liblinear`, ~47 s), and `C=np.inf` for no penalty. Elastic net needs `saga`, which is
too slow here (>10 min per CV). Metrics logged: ROC AUC (primary), log loss, accuracy at 0.5.

### 3.7 Run artifacts

Each run writes `reports/experiments/<YYYYMMDD-HHMMSS>_<name>/` containing the resolved
config, `meta.json` (git commit, config hash, timings), fold scores, OOF and test
predictions (parquet, gitignored) and coefficients. It also appends one row to
`reports/experiments/leaderboard.csv` (tracked in git):
`run_id, name, data_subset, n_features, cv_mean, cv_std, oof_auc, oof_logloss, holdout_auc, fit_seconds, config_hash, git_commit, public_lb`.

### 3.8 Code layout

```
configs/competition.yaml, configs/experiments/*.yaml
src/utils/        paths · config · logging
src/datahelpers/  dataloader · datawrangle (normalize_columns, prepare_frames, splits)
src/features/     registry · derive · encode · domain · build
src/modeling/     models · metrics · validate · train · experiment · diagnostics
src/viz/          theme · export · classification · experiments · explain · segments · dashboards
src/reporting/    deckset (markdown builder) · labels · feature_report
src/submission.py
alfred.py         typer CLI (cv · ladder · leaderboard · diagnose · report)
tests/            synthetic-data tests (no private data)
```

Every module is importable from the notebook and runnable from the command line with
typer, for example `pixi run python -m src.datahelpers.datawrangle columns <csv>`.

### 3.9 Visualization library (`src/viz/`)

Diagnostics (`src/modeling/diagnostics.py`) **compute** tables; `src/viz/` **draws** them.
Reports and notebooks only compose figures.

| Module | Contents |
|---|---|
| `theme.py` | Validated palette (blue / orange / red / gray, blue ramp), class and verdict colors, `style()` base layout, run-vs-comparison line styles |
| `export.py` | `for_slides()` (white surface, large type, no title), `save_png()` |
| `classification.py` | Single-run model quality: ROC, calibration, score distribution, confusion |
| `experiments.py` | Across runs: ladder, paired fold deltas, per-fold scores, gains per idea |
| `explain.py` | Coefficients (technical) and drivers (plain language) |
| `segments.py` | AUC within segments: grouped (notebook) and ranked (deck) |
| `dashboards.py` | `diagnostic_figures()`: the standard set for one run |

**Adding a plot:**
1. Put the function in the module that matches its job, or start a new module.
2. Take data in, return a `go.Figure`, and finish with `theme.style(fig, title)`.
   Use the theme's colors only. Keep file I/O out.
3. Add it to `CATALOG` in `src/viz/__init__.py`. The smoke test then builds it from
   synthetic data.

### 3.10 Risks

| Risk | Mitigation |
|---|---|
| LR is slow with many one-hot and spline columns on 700k rows | Dev subset in Phases 2–3; sparse output with `saga` as a fallback; `fit_seconds` on the leaderboard |
| Overfitting CV by climbing many rungs | Fixed folds, a minimum-gain rule, and the holdout check in Phase 4 |
| Config typos fail silently | Validate block names, params and column references up front |

### 3.11 Decisions log

| # | Decision |
|---|---|
| D1 | Configs are YAML; experiments `extends` a parent |
| D2 | Code is split by theme (`datahelpers`, `features`, `modeling`, `utils`); the whole repo is the reusable template |
| D3 | Tracking is a CSV leaderboard plus one folder per run (no MLflow); the leaderboard is in git |
| D4 | Known `eda.py` issues are left alone (unparsable sweetviz line; path needs a trailing slash) |
| D5 | `notebooks/bruce_v1.py` is the pipeline notebook; it's built notebook-first |
| D6 | `scratch*` files are ignored |
| D7 | Logistic regression only in this plan; the effort goes into features |
| D8 | Columns are renamed with `normalize_columns(df, style="snake_case")`; the default `simple` style keeps `eda.py` unchanged |
| D9 | Staged iteration: features on a dev subset → model on a dev subset → full data. A 20% holdout is scored once; Kaggle test is used only in Phase 4 |
| D10 | Splits and folds are assigned by a salted hash of `id` (salt configurable in YAML), not a random seed |
| D11 | Keep rule uses paired noise (std of per-fold deltas vs the parent), configurable via `keep_rule.noise` |
| D12 | Plots live in `src/viz/`, split by job; diagnostics compute, viz draws, reports compose |
