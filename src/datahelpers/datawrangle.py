#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data wrangling helpers: column-name normalization and related transformations.

Importable from notebooks/scripts, and runnable from the command line:

    pixi run python -m src.datahelpers.datawrangle columns data/raw/<comp>/train.csv
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import typer


def _simple(name: str) -> str:
    """Lowercase and replace spaces with underscores (original behavior)."""
    return name.lower().replace(" ", "_")


def _snake_case(name: str) -> str:
    """Lowercase, collapse any run of non-alphanumerics into one underscore."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


# Add new naming styles here; they become available everywhere via `style=`.
NAME_STYLES: dict[str, Callable[[str], str]] = {
    "simple": _simple,
    "snake_case": _snake_case,
}


def normalize_name(name: str, style: str = "simple") -> str:
    """
    Normalize a single column name.

    Args:
        name: Raw column name.
        style: Key in NAME_STYLES.
            - "simple": "Departure/Arrival time" -> "departure/arrival_time"
            - "snake_case": "Departure/Arrival time" -> "departure_arrival_time"

    Returns:
        The normalized name.
    """
    if style not in NAME_STYLES:
        raise ValueError(
            f"Unknown style {style!r}; choose from {sorted(NAME_STYLES)}"
        )
    return NAME_STYLES[style](name)


def column_mapping(
    columns: Iterable[str],
    style: str = "simple",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build an {old_name: new_name} mapping for a set of column names.

    Args:
        columns: Raw column names.
        style: Key in NAME_STYLES.
        overrides: Explicit {old_name: new_name} entries that win over `style`.

    Returns:
        Mapping from raw to normalized names.

    Raises:
        ValueError: If two columns would end up with the same name.
    """
    overrides = overrides or {}
    mapping = {
        col: overrides.get(col, normalize_name(col, style)) for col in columns
    }

    seen: dict[str, str] = {}
    for old, new in mapping.items():
        if new in seen:
            raise ValueError(
                f"Columns {seen[new]!r} and {old!r} both normalize to {new!r}"
            )
        seen[new] = old
    return mapping


def normalize_columns(
    df,
    style: str = "simple",
    overrides: dict[str, str] | None = None,
):
    """
    Normalize dataframe column names.

    The default style ("simple") keeps the original behavior: lowercase and
    spaces -> underscores. Use style="snake_case" to also replace characters
    such as "/" and "-".

    Args:
        df: Input dataframe (pandas or polars).
        style: Key in NAME_STYLES.
        overrides: Explicit {old_name: new_name} entries that win over `style`.

    Returns:
        Dataframe with normalized column names.
    """
    mapping = column_mapping(df.columns, style=style, overrides=overrides)
    if isinstance(df, pd.DataFrame):
        return df.rename(columns=mapping)
    return df.rename(mapping)  # polars


app = typer.Typer(help=__doc__)


@app.command()
def columns(
    path: Path = typer.Argument(..., help="CSV/parquet file to read the header from."),
    style: str = typer.Option("snake_case", help=f"One of {sorted(NAME_STYLES)}."),
) -> None:
    """Print how each column in PATH would be renamed."""
    import duckdb

    raw = duckdb.sql(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"]
    for old, new in column_mapping(raw, style=style).items():
        typer.echo(f"{old!r:40} -> {new}")


@app.command()
def styles() -> None:
    """List the available naming styles."""
    for name in NAME_STYLES:
        typer.echo(name)


if __name__ == "__main__":
    app()
