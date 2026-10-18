#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build, validate and write Kaggle submission files.

    pixi run python -m src.submission validate data/submissions/<run_id>.csv
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from pathlib import Path

import numpy as np
import pandas as pd
import typer


def make_submission(
    test_ids: pd.Series, preds: np.ndarray, sample_submission: pd.DataFrame, id_col: str, target: str
) -> pd.DataFrame:
    """Predictions in the sample submission's row order and column layout."""
    by_id = pd.Series(np.asarray(preds, dtype=float), index=pd.Index(test_ids, name=id_col), name=target)
    if by_id.index.has_duplicates:
        raise ValueError("Duplicate ids in test predictions")
    missing = set(sample_submission[id_col]) - set(by_id.index)
    if missing:
        raise ValueError(f"{len(missing)} sample-submission ids have no prediction")
    return pd.DataFrame(
        {id_col: sample_submission[id_col].to_numpy(), target: by_id.loc[sample_submission[id_col]].to_numpy()}
    )


def validation_errors(sub: pd.DataFrame, sample_submission: pd.DataFrame, task: str = "binary") -> list[str]:
    """Every problem found (empty list = valid)."""
    errors = []
    if list(sub.columns) != list(sample_submission.columns):
        errors.append(f"columns {list(sub.columns)} != expected {list(sample_submission.columns)}")
        return errors
    id_col, target = sample_submission.columns
    if len(sub) != len(sample_submission):
        errors.append(f"{len(sub):,} rows != expected {len(sample_submission):,}")
    elif not (sub[id_col].to_numpy() == sample_submission[id_col].to_numpy()).all():
        errors.append("ids differ from the sample submission (set or order)")
    values = pd.to_numeric(sub[target], errors="coerce")
    if values.isna().any():
        errors.append(f"{int(values.isna().sum())} missing or non-numeric predictions")
    if np.isinf(values).any():
        errors.append("infinite predictions")
    if task == "binary" and ((values < 0) | (values > 1)).any():
        errors.append("probabilities outside [0, 1]")
    return errors


def validate_submission(sub: pd.DataFrame, sample_submission: pd.DataFrame, task: str = "binary") -> None:
    """Raise ValueError listing every problem."""
    errors = validation_errors(sub, sample_submission, task)
    if errors:
        raise ValueError("Invalid submission: " + "; ".join(errors))


def write_submission(sub: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


app = typer.Typer(help=__doc__)


@app.callback()
def _main() -> None:
    """Keep subcommand style even with a single command."""


@app.command()
def validate(path: Path = typer.Argument(..., help="Submission CSV to check.")) -> None:
    """Check a submission CSV against the competition's sample submission."""
    from src.datahelpers.dataloader import read_table
    from src.utils.config import data_dir, load_config
    from src.utils.paths import project_root

    cfg = load_config(project_root() / "configs" / "competition.yaml")
    sample = read_table(data_dir(cfg) / cfg["data"]["files"]["sample_submission"])
    sub = read_table(path)
    errors = validation_errors(sub, sample, cfg.get("task", "binary"))
    for e in errors:
        typer.echo(f"✗ {e}")
    if errors:
        raise typer.Exit(1)
    target = sample.columns[1]
    typer.echo(f"✓ valid: {len(sub):,} rows · mean {sub[target].mean():.4f} · min {sub[target].min():.4f} · max {sub[target].max():.4f}")


if __name__ == "__main__":
    app()
