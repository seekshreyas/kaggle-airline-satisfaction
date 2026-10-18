import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    import pandas as pd
    import numpy as np
    import logging
    import marimo as mo
    import duckdb as ddb
    import icecream as ic

    from pathlib import Path

    # plotting
    import plotly.io as pio
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # eda
    import sweetviz as sv
    from skrub import TableReport

    from dotenv import load_dotenv

    load_dotenv()
    con = ddb.connect()

    def _get_required_env(var_name: str) -> str:
        value = os.getenv(var_name)
        if value is None or value == "":
            raise ValueError(
                f"Missing required environment variable: {var_name}"
            )
        return value

    # preferences
    pio.get_chrome()
    pd.options.plotting.backend = "plotly"

    # project defaults
    DIR_PROJECT = _get_required_env("DIR_PROJECT").rstrip("/\\")
    DIR_KAGGLE_REL = _get_required_env("DIR_KAGGLE_REL").lstrip("/\\")
    DIR_KAGGLE = os.path.join(DIR_PROJECT, DIR_KAGGLE_REL)

    print(f"Working directory: {DIR_PROJECT}")
    print(f"Kaggle directory: {DIR_KAGGLE}")

    sys.path.insert(1, DIR_PROJECT)
    sv.config_parser.read(DIR_PROJECT + "/configs/sweetviz_config.ini")

    # Notebook status
    NOTEBOOKSTATUS = "dev"


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Project Details
    ===============

    This is my notebook for the [kaggle competition]


    [kaggle competition]: https://www.kaggle.com/competitions/playground-series-s6e10/overview
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phase 1 · Data and splits

    Before any modeling, the labelled training data is split **once** into three parts:

    | Part | Share | Used for |
    |---|---|---|
    | 🔒 **Holdout** | 20% of train | Scored **once**, in Phase 4. Stands in for Kaggle's test set. |
    | 🧪 **Dev subset** | 25% of the rest (≈ 20% of train) | Fast iteration in Phases 2–3, with **k = 4** fold CV |
    | **Rest** | the remainder | Joins the dev subset for full-data CV in Phase 4 |

    **How rows are assigned.** No random seed is involved: each row's `id` is hashed with a
    configurable **salt** (`randomization.salt` in `configs/competition.yaml`). Within each
    class, rows are ranked by that hash, which makes every split and fold *exactly*
    stratified. The same salt always gives the same assignment; a new salt reshuffles everything.
    """)
    return


@app.cell
def _():
    from src.utils.config import load_config, config_hash
    from src.datahelpers.dataloader import load_competition_data, profile
    from src.datahelpers.splits import assign_splits, split_summary

    return (
        assign_splits,
        load_competition_data,
        load_config,
        profile,
        split_summary,
    )


@app.cell
def _():
    from src.modeling.experiment import run_experiment
    from src.modeling.diagnostics import (
        leaderboard,
        ladder,
        fold_scores,
        paired_deltas,
        latest_runs,
        diagnose_run,
    )
    from src.viz import diagnostic_figures, ladder_figure, paired_deltas_figure

    return (
        diagnose_run,
        diagnostic_figures,
        fold_scores,
        ladder,
        ladder_figure,
        latest_runs,
        leaderboard,
        paired_deltas,
        paired_deltas_figure,
        run_experiment,
    )


@app.cell
def _(load_config):
    comp_cfg = load_config(Path(DIR_PROJECT) / "configs" / "competition.yaml")
    mo.md(
        f"Config **{comp_cfg['competition']}** · target `{comp_cfg['target']}` · "
        f"metric `{comp_cfg['metric']}` · salt `{comp_cfg['randomization']['salt']}` · "
        f"k = {comp_cfg['cv']['n_splits']} folds"
    )
    return (comp_cfg,)


@app.cell
def _(comp_cfg, load_competition_data, profile):
    data = load_competition_data(comp_cfg)
    profile(data, comp_cfg)
    return (data,)


@app.cell
def _(assign_splits, comp_cfg, data, split_summary):
    train = assign_splits(data.train, comp_cfg)
    splits_table = split_summary(train, comp_cfg["target"])
    return splits_table, train


@app.cell
def _(splits_table):
    _fig = px.bar(
        splits_table,
        x="split",
        y="rows",
        text=splits_table["target_rate"].map("{:.2%} satisfied".format),
        title="Rows per split (label = target rate: identical everywhere ⇒ stratified)",
    )
    _fig.update_traces(textposition="outside").update_layout(
        yaxis_title="rows", xaxis_title=""
    )
    mo.vstack([_fig, mo.ui.table(splits_table, selection=None)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phase 2 · Feature engineering

    **Setup.** The model is held fixed (logistic regression, `C = 1`), and so is the data
    (dev subset, ≈140k rows). Each rung of the ladder adds **one** feature idea and is scored with
    **4-fold CV ROC AUC**. ROC AUC is the probability that a randomly chosen satisfied passenger
    gets a higher score than a randomly chosen unsatisfied one (0.5 = coin flip, 1 = perfect).
    All rungs share the same hashed folds, so their scores are directly comparable.

    **Keep rule** (`keep_rule` in `competition.yaml`): a rung is kept if it beats its parent by
    ≥ 0.0005 AUC **and** the gain exceeds the std of its **per-fold differences** from the parent
    (`noise: paired_std`). When a rung is reverted, the next rung builds on the previous winner.
    """)
    return


@app.cell
def _(comp_cfg, fold_scores, ladder, last_run, leaderboard):
    exp_dir = Path(DIR_PROJECT) / comp_cfg["paths"]["experiments_dir"]
    _ = last_run  # refresh after a run from the panel below
    board = leaderboard(exp_dir)
    folds_long = fold_scores(exp_dir, comp_cfg["metric"])
    ladder_table = ladder(
        board, comp_cfg["keep_rule"], folds_long, comp_cfg["metric"]
    )
    return board, exp_dir, folds_long, ladder_table


@app.cell
def _(ladder_figure, ladder_table):
    mo.vstack(
        [
            ladder_figure(ladder_table),
            mo.md(
                "**One-hot ratings** is the big win (+0.027). **Rating aggregates** add a small but "
                "consistent +0.0005. Delays, splines and crosses add ≤ 0.0002: once the model sees "
                "every rating, they carry little extra signal."
            ),
            mo.ui.table(ladder_table.round(5), selection=None),
        ]
    )
    return


@app.cell
def _(board, comp_cfg, folds_long, paired_deltas, paired_deltas_figure):
    _d = paired_deltas(board, folds_long, comp_cfg["metric"])
    mo.vstack(
        [
            paired_deltas_figure(
                _d[_d["parent"] != "lr_00_baseline"],
                comp_cfg["keep_rule"]["min_gain"],
            ),
            mo.callout(
                mo.md(
                    "**Why the rule compares against paired noise.** Every rung uses the same folds, so fold "
                    "difficulty (fold std ≈ 0.0022) is shared and cancels when comparing a rung with its "
                    "parent. The per-fold differences vary 10–80× less, so they are the right yardstick: "
                    "`lr_06` gains on all 4 folds and is kept."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Diagnostics · read any run

    Pick a run and a comparison (its parent by default). All charts use the run's
    **out-of-fold (OOF)** predictions, so every row is scored by a model that never saw it.

    | Chart | What to look for |
    |---|---|
    | **ROC** | Curve hugging the top-left; the gap between the two curves is the AUC gain |
    | **Calibration** | Points on the diagonal mean "70% predicted" really is 70% satisfied |
    | **Score distribution** | Little overlap between classes; the overlap is where errors live |
    | **Confusion** | Errors at the 0.5 threshold, as a share of each actual class |
    | **Folds** | Paired lines: a real gain sits above the parent on *every* fold |
    | **Segments** | Where the model ranks well or poorly (AUC within each group) |
    | **Coefficients** | Which features push towards or away from "satisfied" |
    """)
    return


@app.cell
def _(ladder_table):
    _names = sorted(ladder_table["name"])
    _labels = dict(
        sorted(
            {
                f"{n} · {a:.4f}": n
                for n, a in zip(ladder_table["name"], ladder_table["cv_mean"])
            }.items()
        )
    )
    diag_pick = mo.ui.dropdown(
        options=_labels,
        value=next(
            k for k, v in _labels.items() if v == "lr_06_rating_aggregates"
        ),
        label="Run",
    )
    diag_compare = mo.ui.dropdown(
        options=["parent", "none", *_names],
        value="parent",
        label="Compare with",
    )
    mo.hstack([diag_pick, diag_compare], justify="start", gap=1)
    return diag_compare, diag_pick


@app.cell
def _(
    board,
    comp_cfg,
    diag_compare,
    diag_pick,
    diagnose_run,
    exp_dir,
    latest_runs,
    train,
):
    _latest = latest_runs(board).set_index("name")
    _segments = train[
        [comp_cfg["id_col"], "class", "type_of_travel", "customer_type"]
    ]
    _other = (
        _latest.loc[diag_pick.value, "parent"]
        if diag_compare.value == "parent"
        else diag_compare.value
    )
    _other = (
        None
        if (_other is None or _other == "none" or pd.isna(_other))
        else _other
    )

    def _diagnose(name):
        return diagnose_run(
            exp_dir,
            _latest.loc[name, "run_id"],
            _segments,
            comp_cfg["id_col"],
            comp_cfg["metric"],
        )

    diag_run = _diagnose(diag_pick.value)
    diag_other = _diagnose(_other) if _other else None
    return diag_other, diag_run


@app.cell
def _(diag_other, diag_run, diagnostic_figures):
    def _stat(key, label, fmt, better="increase"):
        v = diag_run.scores[key]
        if diag_other is None:
            return mo.stat(fmt.format(v), label=label, bordered=True)
        d = v - diag_other.scores[key]
        return mo.stat(
            fmt.format(v),
            label=label,
            bordered=True,
            caption=f"{d:+.4f} vs {diag_other.name}",
            direction="increase" if d > 0 else "decrease",
            target_direction=better,
        )

    _f = diagnostic_figures(diag_run, diag_other)
    mo.vstack(
        [
            mo.hstack(
                [
                    _stat("roc_auc", "OOF ROC AUC", "{:.4f}"),
                    _stat("log_loss", "Log loss", "{:.4f}", "decrease"),
                    _stat("brier", "Brier score", "{:.4f}", "decrease"),
                    _stat("accuracy", "Accuracy @0.5", "{:.2%}"),
                    mo.stat(
                        f"{diag_run.scores['n_features']}",
                        label="Features",
                        bordered=True,
                        caption=f"{diag_run.scores['n_rows']:,} OOF rows",
                    ),
                ],
                widths="equal",
                gap=1,
            ),
            mo.hstack([_f["ROC"], _f["Calibration"]], widths="equal"),
            mo.hstack(
                [_f["Score distribution"], _f["Confusion"]], widths="equal"
            ),
            mo.hstack([_f["Folds"], _f["Segments"]], widths="equal"),
            _f["Coefficients"],
        ],
        gap=1,
    )
    return


@app.cell
def _():
    exp_picker = mo.ui.dropdown(
        options={
            p.stem: str(p)
            for p in sorted(
                (Path(DIR_PROJECT) / "configs" / "experiments").glob(
                    "lr_0[0-7]_*.yaml"
                )
            )
        },
        value="lr_01_ratings_onehot",
        label="Experiment",
    )
    run_btn = mo.ui.run_button(label="▶ Run 4-fold CV")
    mo.vstack(
        [
            mo.md("### Try an experiment"),
            mo.hstack([exp_picker, run_btn], justify="start"),
        ]
    )
    return exp_picker, run_btn


@app.cell
def _(exp_picker, load_config, run_btn, run_experiment, train):
    last_run = (
        run_experiment(load_config(exp_picker.value), train)
        if run_btn.value
        else None
    )
    (
        mo.vstack(
            [
                mo.md(
                    f"**{last_run.cfg['name']}**: AUC {last_run.row['cv_mean']:.5f} ± {last_run.row['cv_std']:.5f} "
                    f"· {last_run.row['n_features']} features · saved as `{last_run.run_id}`"
                ),
                mo.ui.table(last_run.cv.fold_scores.round(5), selection=None),
            ]
        )
        if last_run
        else mo.md(
            "_Pick an experiment and press run; the ladder above refreshes._"
        )
    )
    return (last_run,)


@app.cell
def _():
    from src.modeling.diagnostics import grid_table
    from src.viz import (
        grid_figure,
        roc_figure,
        calibration_figure,
        segment_auc_figure,
    )

    return (
        calibration_figure,
        grid_figure,
        grid_table,
        roc_figure,
        segment_auc_figure,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phase 3 · Model iteration

    **Setup.** Features are now fixed at the Phase 2 winner (`lr_06`, 97 features); only the
    model's regularization changes. `C` controls how strongly coefficients are shrunk (small `C`
    = strong shrinkage); the penalty is either **L2** (shrinks all coefficients) or **L1** (can
    zero some out). Same dev subset and same 4 folds as Phase 2.

    Elastic net was dropped: its solver (`saga`) needed > 10 minutes per grid point.
    """)
    return


@app.cell
def _(board, grid_figure, grid_table):
    _g = grid_table(board, "lr_08_regularization")
    mo.vstack(
        [
            grid_figure(_g),
            mo.md(
                "**The model is saturated.** Weakening regularization (`C` ≥ 1) changes nothing, "
                "strengthening it (`C` = 0.01) costs 0.0005–0.001 AUC, and L1 at best ties L2 while "
                "being 5–7× slower. **Final setting: `C = 1`, L2 (`lbfgs`)**: simplest and fastest."
            ),
            mo.ui.table(_g.round(5), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Phase 4 · Full data and the holdout

    The final model (`lr_final_full`: `lr_06` features, `C = 1`, L2) is trained three ways:

    1. **4-fold CV on all 559,708 non-holdout rows**, with its own hashed folds (`fold_full`)
    2. **Fit on those rows, scored once on the 20% holdout** (139,927 rows never used before)
    3. **Refit on all 699,635 labelled rows**, then predict the Kaggle test set (299,844 rows)

    If the holdout score is close to the CV score, the many experiments did not overfit the folds.
    """)
    return


@app.cell
def _(board):
    _final = (
        board[board["name"] == "lr_final_full"].sort_values("run_id").iloc[-1]
    )
    _dev = (
        board[board["name"] == "lr_06_rating_aggregates"]
        .sort_values("run_id")
        .iloc[-1]
    )
    mo.hstack(
        [
            mo.stat(
                f"{_dev['cv_mean']:.4f}",
                label="Dev CV AUC",
                caption="140k rows · Phase 2",
                bordered=True,
            ),
            mo.stat(
                f"{_final['cv_mean']:.4f}",
                label="Full-data CV AUC",
                caption=f"± {_final['cv_std']:.4f} · 560k rows",
                bordered=True,
            ),
            mo.stat(
                f"{_final['holdout_auc']:.4f}",
                label="Holdout AUC (once)",
                bordered=True,
                caption=f"{_final['holdout_auc'] - _final['cv_mean']:+.4f} vs full CV",
                direction="increase"
                if _final["holdout_auc"] >= _final["cv_mean"]
                else "decrease",
            ),
            mo.stat(
                f"{_final['oof_accuracy']:.1%}",
                label="Accuracy @0.5",
                caption="full-data OOF",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(
    board,
    calibration_figure,
    comp_cfg,
    diagnose_run,
    exp_dir,
    roc_figure,
    segment_auc_figure,
    train,
):
    final_run = diagnose_run(
        exp_dir,
        board[board["name"] == "lr_final_full"]
        .sort_values("run_id")
        .iloc[-1]["run_id"],
        train[
            [comp_cfg["id_col"], "class", "type_of_travel", "customer_type"]
        ],
        comp_cfg["id_col"],
        comp_cfg["metric"],
    )
    mo.vstack(
        [
            mo.hstack(
                [roc_figure(final_run), calibration_figure(final_run)],
                widths="equal",
            ),
            segment_auc_figure(final_run, comp_cfg.get("display_names")),
        ]
    )
    return


@app.cell
def _(board):
    _final = (
        board[board["name"] == "lr_final_full"].sort_values("run_id").iloc[-1]
    )
    mo.callout(
        mo.md(
            f"**Submission ready:** `{_final['submission_path']}` (validated: 299,844 rows, probabilities in [0, 1], "
            "ids in sample-submission order).\n\n"
            "Uploading is a manual, confirmed step: "
            f'`pixi run python alfred.py submit {_final["submission_path"]} -m "lr_final_full"`'
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
