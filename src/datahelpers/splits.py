#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic, salted-hash data splits (holdout, dev subset, CV folds).

Each row's assignment depends only on hash(salt, key), so it's identical across runs
and independent of row order. Changing the salt reshuffles everything.

    pixi run python -m src.datahelpers.splits summary configs/competition.yaml
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import typer

_MAX_U64 = float(2**64)


def purpose_salt(salt: str, purpose: str) -> str:
    """Derive an independent salt per purpose, e.g. "s6e10-v1/holdout"."""
    if not salt:
        raise ValueError("salt must be a non-empty string")
    return f"{salt}/{purpose}"


def hash_uniform(keys: Iterable, salt: str) -> np.ndarray:
    """
    Map each key to a deterministic pseudo-random number in [0, 1).

    Uses BLAKE2b (64-bit digest) of "<salt>:<key>".
    """
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(f"{salt}:{k}".encode(), digest_size=8).digest(), "big"
            )
            / _MAX_U64
            for k in keys
        ),
        dtype=np.float64,
    )


def _rank_within(u: np.ndarray, strata: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Rank of each row by hash value within its stratum, and the stratum size."""
    groups = np.zeros(len(u), dtype=int) if strata is None else pd.factorize(strata)[0]
    frame = pd.DataFrame({"u": u, "g": groups})
    rank = frame.groupby("g")["u"].rank(method="first").to_numpy(dtype=int) - 1
    size = frame.groupby("g")["u"].transform("size").to_numpy()
    return rank, size


def hash_split(
    keys: Iterable,
    frac: float,
    salt: str,
    strata: Iterable | None = None,
) -> np.ndarray:
    """
    Boolean mask selecting `frac` of the rows.

    With `strata` (e.g. the target), rows are ranked by hash within each class and the
    lowest round(frac * class size) are selected, giving an exact class balance.
    Without `strata`, a row is selected when its hash value is below `frac`.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    u = hash_uniform(keys, salt)
    if strata is None:
        return u < frac
    rank, size = _rank_within(u, np.asarray(list(strata)))
    return rank < np.round(frac * size)


def hash_folds(
    keys: Iterable,
    n_splits: int,
    salt: str,
    strata: Iterable | None = None,
) -> np.ndarray:
    """
    Fold index (0 .. n_splits-1) per row.

    With `strata`, folds are dealt round-robin by hash rank within each class, so every
    fold has the same class balance (sizes differ by at most one per class).
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    u = hash_uniform(keys, salt)
    if strata is None:
        return np.minimum((u * n_splits).astype(int), n_splits - 1)
    rank, _ = _rank_within(u, np.asarray(list(strata)))
    return rank % n_splits


def fold_indices(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert fold labels into sklearn-style (train_idx, valid_idx) pairs."""
    return [
        (np.flatnonzero(folds != k), np.flatnonzero(folds == k))
        for k in np.unique(folds)
    ]


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add `is_holdout`, `is_dev`, `fold` and `fold_full` columns to the labelled training frame.

    `cfg` is the competition config (randomization, splits, cv, target). Holdout rows
    get is_dev=False and fold=fold_full=-1. `fold` is assigned within the dev subset
    (Phases 2-3); `fold_full` over every non-holdout row (Phase 4), with its own salt.
    """
    rnd, splits = cfg["randomization"], cfg["splits"]
    key, salt = rnd["key"], rnd["salt"]
    strata = df[cfg["target"]] if rnd.get("stratify", True) else None

    out = df.copy()
    out["is_holdout"] = hash_split(
        out[key], splits["holdout_frac"], purpose_salt(salt, "holdout"), strata
    )

    rest = out.loc[~out["is_holdout"]]
    rest_strata = None if strata is None else rest[cfg["target"]]
    out["is_dev"] = False
    out.loc[rest.index, "is_dev"] = hash_split(
        rest[key], splits["dev_frac"], purpose_salt(salt, "dev"), rest_strata
    )

    dev = out.loc[out["is_dev"]]
    dev_strata = None if strata is None else dev[cfg["target"]]
    out["fold"] = -1
    out.loc[dev.index, "fold"] = hash_folds(
        dev[key], cfg["cv"]["n_splits"], purpose_salt(salt, "cv"), dev_strata
    )

    out["fold_full"] = -1
    out.loc[rest.index, "fold_full"] = hash_folds(
        rest[key], cfg["cv"]["n_splits"], purpose_salt(salt, "cv-full"), rest_strata
    )
    return out


def split_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows and target rate per split and fold (expects assign_splits output)."""
    part = np.where(df["is_holdout"], "holdout", np.where(df["is_dev"], "dev", "rest"))
    label = pd.Series(part, index=df.index).where(
        df["fold"] < 0, "dev · fold " + df["fold"].astype(str)
    )
    return (
        df.assign(split=label)
        .groupby("split")[target]
        .agg(rows="size", target_rate="mean")
        .reset_index()
    )


app = typer.Typer(help=__doc__)


@app.callback()
def _main() -> None:
    """Keep subcommand style (`splits summary ...`) even with a single command."""


@app.command()
def summary(
    config: Path = typer.Argument(..., help="Path to configs/competition.yaml."),
    salt: str | None = typer.Option(None, help="Override randomization.salt."),
) -> None:
    """Assign splits for the training data and print rows / target rate per split."""
    import duckdb
    import yaml

    from src.datahelpers.datawrangle import normalize_columns

    cfg = yaml.safe_load(config.read_text())
    if salt:
        cfg["randomization"]["salt"] = salt
    train_path = Path(cfg["data"]["dir"]) / cfg["data"]["files"]["train"]
    df = duckdb.sql(f"SELECT * FROM '{train_path}'").df()
    df = normalize_columns(df, **cfg["columns"])
    df[cfg["target"]] = df[cfg["target"]].astype(int)

    result = split_summary(assign_splits(df, cfg), cfg["target"])
    typer.echo(f"salt = {cfg['randomization']['salt']!r}")
    typer.echo(result.to_string(index=False))


if __name__ == "__main__":
    app()
