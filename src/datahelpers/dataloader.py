#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loader module for handling dataset loading and preprocessing.

    pixi run python -m src.datahelpers.dataloader profile configs/competition.yaml
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import typer

from src.datahelpers.datawrangle import normalize_columns
from src.utils.config import data_dir, load_config


@dataclass
class CompetitionData:
    """Raw competition files, columns already renamed."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/parquet file with DuckDB into pandas (the I/O boundary)."""
    return duckdb.sql(f"SELECT * FROM '{path}'").df()


def load_competition_data(cfg: dict) -> CompetitionData:
    """Load train/test/sample_submission, rename columns, cast a binary target to int."""
    folder, files = data_dir(cfg), cfg["data"]["files"]
    frames = {
        name: normalize_columns(read_table(folder / files[name]), **cfg["columns"])
        for name in ("train", "test", "sample_submission")
    }
    if cfg.get("task") == "binary":
        frames["train"][cfg["target"]] = frames["train"][cfg["target"]].astype("int8")
    return CompetitionData(**frames)


def profile(data: CompetitionData, cfg: dict) -> pd.DataFrame:
    """Rows, columns, nulls and target rate per file."""
    rows = []
    for name in ("train", "test", "sample_submission"):
        df = getattr(data, name)
        nulls = df.isna().sum()
        rows.append(
            {
                "file": name,
                "rows": len(df),
                "columns": df.shape[1],
                "null_cells": int(nulls.sum()),
                "null_columns": ", ".join(f"{c} ({n})" for c, n in nulls[nulls > 0].items()),
                "target_rate": df[cfg["target"]].mean() if name == "train" else None,
            }
        )
    return pd.DataFrame(rows)


app = typer.Typer(help=__doc__)


@app.callback()
def _main() -> None:
    """Keep subcommand style even with a single command."""


@app.command("profile")
def profile_cmd(config: Path = typer.Argument(..., help="Competition or experiment YAML.")) -> None:
    """Print rows, nulls and target rate for each competition file."""
    cfg = load_config(config)
    typer.echo(profile(load_competition_data(cfg), cfg).to_string(index=False))


if __name__ == "__main__":
    app()
