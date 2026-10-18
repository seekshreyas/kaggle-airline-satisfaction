#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Alfred module for handling various utility functions and integrations.

    pixi run cv configs/experiments/lr_00_baseline.yaml [more.yaml ...]
    pixi run train configs/experiments/lr_final_full.yaml
    pixi run python alfred.py submit data/submissions/<run_id>.csv -m "message"
    pixi run ladder
    pixi run report configs/reports/phase2_feature_engineering.yaml
    pixi run python alfred.py features
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

app = typer.Typer(help=__doc__, no_args_is_help=True)
pd.set_option("display.width", 200)


def _comp_cfg() -> dict:
    from src.utils.config import load_config
    from src.utils.paths import project_root

    return load_config(project_root() / "configs" / "competition.yaml")


@app.command()
def features() -> None:
    """List the registered feature blocks."""
    import src.features  # noqa: F401
    from src.features.registry import list_blocks

    typer.echo(list_blocks().to_string(index=False))


def _run_configs(configs: list[Path], phase4: bool) -> None:
    """Load data once, then run every config (and grid point) in order."""
    from src.datahelpers.dataloader import load_competition_data
    from src.datahelpers.splits import assign_splits
    from src.modeling.experiment import expand_grid, run_experiment
    from src.utils.config import load_config
    from src.utils.logs import get_logger

    log = get_logger()
    base = _comp_cfg()
    data = load_competition_data(base)
    train = assign_splits(data.train, base)  # loaded once for all configs
    for path in configs:
        for cfg in expand_grid(load_config(path)):
            if not phase4:  # CV only: never touch the holdout or the test set
                cfg["score_holdout"] = cfg["predict_test"] = False
            r = run_experiment(cfg, train, test=data.test, sample_submission=data.sample_submission)
            extra = ""
            if r.row["holdout_auc"] is not None:
                extra += f" · holdout AUC {r.row['holdout_auc']:.5f}"
            if r.row["submission_path"]:
                extra += f" · submission {r.row['submission_path']}"
            log.info(
                f"{cfg['name']}: AUC {r.row['cv_mean']:.5f} ± {r.row['cv_std']:.5f} "
                f"({r.row['n_features']} features, {r.row['fit_seconds']}s){extra} -> {r.run_id}"
            )


@app.command()
def cv(configs: list[Path] = typer.Argument(..., help="One or more experiment YAMLs, run in order.")) -> None:
    """K-fold CV only; saves a run folder and a leaderboard row per config (no holdout, no test)."""
    _run_configs(configs, phase4=False)


@app.command()
def train(configs: list[Path] = typer.Argument(..., help="Experiment YAML(s) with Phase 4 flags.")) -> None:
    """CV, then the config's Phase 4 steps: score_holdout (once) and predict_test (submission CSV)."""
    _run_configs(configs, phase4=True)


@app.command()
def submit(
    path: Path = typer.Argument(..., help="Submission CSV, e.g. data/submissions/<run_id>.csv."),
    message: str = typer.Option(..., "-m", "--message", help="Kaggle submission message."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt (your own use only)."),
) -> None:
    """Validate a submission, show a summary, ask y/N, then upload with the Kaggle CLI."""
    import subprocess

    from src.datahelpers.dataloader import read_table
    from src.submission import validation_errors
    from src.utils.config import data_dir

    cfg = _comp_cfg()
    sample = read_table(data_dir(cfg) / cfg["data"]["files"]["sample_submission"])
    sub = read_table(path)
    errors = validation_errors(sub, sample, cfg.get("task", "binary"))
    if errors:
        for e in errors:
            typer.echo(f"✗ {e}")
        raise typer.Exit(1)
    target = sample.columns[1]
    typer.echo(f"✓ {path}: {len(sub):,} rows, mean {sub[target].mean():.4f}")
    typer.echo(f"  competition: {cfg['competition']}\n  message:     {message}")
    if not yes and not typer.confirm("Upload this submission to Kaggle?", default=False):
        typer.echo("Aborted; nothing was uploaded.")
        raise typer.Exit(0)
    subprocess.run(["kaggle", "competitions", "submit", "-c", cfg["competition"], "-f", str(path), "-m", message], check=True)


@app.command()
def leaderboard(top: int = typer.Option(20, help="Rows to show.")) -> None:
    """All runs, best CV AUC first."""
    from src.modeling.diagnostics import leaderboard as lb
    from src.utils.paths import project_root

    board = lb(project_root() / _comp_cfg()["paths"]["experiments_dir"])
    cols = ["run_id", "n_features", "cv_mean", "cv_std", "oof_log_loss", "fit_seconds"]
    typer.echo(board[cols].head(top).to_string(index=False) if not board.empty else "No runs yet.")


@app.command()
def ladder() -> None:
    """Latest run per experiment vs its parent, with the keep/revert verdict."""
    from src.modeling.diagnostics import ladder as ld
    from src.modeling.diagnostics import leaderboard as lb
    from src.utils.paths import project_root

    cfg = _comp_cfg()
    from src.modeling.diagnostics import fold_scores

    exp_dir = project_root() / cfg["paths"]["experiments_dir"]
    table = ld(lb(exp_dir), cfg["keep_rule"], fold_scores(exp_dir, cfg["metric"]), cfg["metric"])
    typer.echo(table.to_string(index=False, float_format="{:.5f}".format) if not table.empty else "No runs yet.")


@app.command()
def diagnose(
    name: str = typer.Argument(..., help="Experiment name; its latest run is used."),
    compare: str | None = typer.Option("parent", help="Experiment to overlay ('parent', a name, or 'none')."),
) -> None:
    """Save every diagnostic chart for a run as PNG in reports/plots/<run_id>/."""
    from src.datahelpers.dataloader import load_competition_data
    from src.modeling.diagnostics import diagnose_run, latest_runs
    from src.modeling.diagnostics import leaderboard as lb
    from src.viz import diagnostic_figures, save_png
    from src.utils.paths import project_root

    cfg = _comp_cfg()
    exp_dir = project_root() / cfg["paths"]["experiments_dir"]
    latest = latest_runs(lb(exp_dir)).set_index("name")
    other = latest.loc[name, "parent"] if compare == "parent" else (None if compare == "none" else compare)
    segs = load_competition_data(cfg).train[[cfg["id_col"], "class", "type_of_travel", "customer_type"]]

    def _diag(n):
        return diagnose_run(exp_dir, latest.loc[n, "run_id"], segs, cfg["id_col"], cfg["metric"])

    run = _diag(name)
    figs = diagnostic_figures(run, _diag(other) if isinstance(other, str) else None)
    out_dir = project_root() / "reports" / "plots" / run.run_id
    for i, (title, fig) in enumerate(figs.items(), start=1):
        save_png(fig, out_dir / f"{i:02d}_{title.lower().replace(' ', '_')}.png")
    typer.echo(f"Wrote {len(figs)} PNGs to {out_dir}")


@app.command()
def report(config: Path = typer.Argument(..., help="Report YAML, e.g. configs/reports/phase2_feature_engineering.yaml.")) -> None:
    """Executive deck: PNG charts in reports/plots/ + Deckset markdown."""
    from src.datahelpers.dataloader import load_competition_data
    from src.datahelpers.splits import assign_splits
    from src.reporting.feature_report import generate

    cfg = _comp_cfg()
    train = assign_splits(load_competition_data(cfg).train, cfg)
    deck, charts = generate(config, train)
    typer.echo(f"Wrote {len(charts)} charts and {deck}")


if __name__ == "__main__":
    load_dotenv()
    app()
