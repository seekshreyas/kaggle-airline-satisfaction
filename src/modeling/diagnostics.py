#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure functions returning DataFrames for tables and Plotly charts.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def coefficients(fitted_pipe) -> pd.DataFrame:
    """Feature, coefficient and |coefficient| for a fitted linear pipeline (sorted)."""
    names = fitted_pipe[:-1].get_feature_names_out()
    coef = fitted_pipe[-1].coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coef": coef, "abs_coef": np.abs(coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def leaderboard(experiments_dir: Path) -> pd.DataFrame:
    """All runs from leaderboard.csv, best CV first (empty frame if none yet)."""
    path = Path(experiments_dir) / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    return duckdb.sql(f"SELECT * FROM read_csv_auto('{path}') ORDER BY cv_mean DESC").df()


def fold_scores(experiments_dir: Path, metric: str) -> pd.DataFrame:
    """Per-fold scores of every saved run: run_id, fold, <metric>."""
    pattern = Path(experiments_dir) / "*" / "cv_scores.csv"
    if not list(Path(experiments_dir).glob("*/cv_scores.csv")):
        return pd.DataFrame(columns=["run_id", "fold", metric])
    return duckdb.sql(
        f"""
        SELECT regexp_extract(filename, '([^/]+)/cv_scores\\.csv$', 1) AS run_id, fold, {metric}
        FROM read_csv_auto('{pattern}', filename = true)
        """
    ).df()


def latest_runs(board: pd.DataFrame) -> pd.DataFrame:
    """Most recent run per experiment name."""
    return board.sort_values("run_id").groupby("name", as_index=False).last()


def paired_deltas(board: pd.DataFrame, folds: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-fold metric difference of each experiment's latest run vs its parent's latest run."""
    latest = latest_runs(board)
    run_of = latest.set_index("name")["run_id"]
    wide = folds.pivot(index="fold", columns="run_id", values=metric)
    rows = []
    for _, r in latest.dropna(subset=["parent"]).iterrows():
        child, parent = r["run_id"], run_of.get(r["parent"])
        if parent in wide and child in wide:
            for fold, d in (wide[child] - wide[parent]).items():
                rows.append({"name": r["name"], "parent": r["parent"], "fold": fold, "delta": d})
    return pd.DataFrame(rows, columns=["name", "parent", "fold", "delta"])


def ladder(board: pd.DataFrame, keep_rule: dict, folds: pd.DataFrame | None = None, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Latest run per experiment vs its parent, with the keep/revert verdict.

    keep if (cv_mean - parent cv_mean) >= min_gain and the gain exceeds the noise level
    chosen by keep_rule["noise"]: "paired_std" (std of per-fold deltas vs the parent; needs
    `folds`), "fold_std" (the experiment's fold std) or "none".
    """
    if board.empty:
        return board
    if "data_subset" in board:  # only dev-subset runs share folds; full-data runs aren't comparable
        board = board[board["data_subset"] == "dev"]
    latest = latest_runs(board)
    by_name = latest.set_index("name")["cv_mean"]
    out = latest[["name", "parent", "n_features", "cv_mean", "cv_std", "fit_seconds"]].copy()
    out["parent_cv"] = out["parent"].map(by_name)
    out["delta"] = out["cv_mean"] - out["parent_cv"]
    if folds is not None and not folds.empty:
        paired = paired_deltas(board, folds, metric).groupby("name")["delta"]
        out["paired_std"] = out["name"].map(paired.std(ddof=0))
        out["folds_improved"] = out["name"].map(paired.apply(lambda d: f"{(d > 0).sum()}/{len(d)}"))

    def verdict(r) -> str:
        if pd.isna(r["parent_cv"]):
            return "reference"
        noise = {"paired_std": r.get("paired_std"), "fold_std": r["cv_std"], "none": 0.0}[keep_rule.get("noise", "paired_std")]
        if pd.isna(noise):
            raise ValueError("keep_rule.noise=paired_std needs per-fold scores (`folds`)")
        return "keep" if r["delta"] >= keep_rule["min_gain"] and r["delta"] > noise else "revert"

    out["verdict"] = out.apply(verdict, axis=1)
    return out.sort_values("name").reset_index(drop=True)


def load_coefficients(experiments_dir: Path, run_id: str) -> pd.DataFrame:
    """coefficients.csv of one saved run."""
    return pd.read_csv(Path(experiments_dir) / run_id / "coefficients.csv")


# ---------------------------------------------------------------------------
# Per-run diagnostics (all computed from the saved OOF predictions)
# ---------------------------------------------------------------------------


@dataclass
class RunDiagnostics:
    """Everything needed to plot one run. Built by `diagnose_run`."""

    name: str
    run_id: str
    scores: dict  # headline metrics on the OOF predictions
    roc: pd.DataFrame  # fpr, tpr
    calibration: pd.DataFrame  # bin, mean_pred, frac_pos, n
    histogram: pd.DataFrame  # bin_center, label, share
    confusion: pd.DataFrame  # actual, predicted, n, share
    segments: pd.DataFrame  # segment_col, segment, n, pos_rate, auc
    folds: pd.DataFrame  # fold, <metric>
    coefs: pd.DataFrame  # feature, coef, abs_coef


def load_oof(experiments_dir: Path, run_id: str) -> pd.DataFrame:
    """oof.parquet of one saved run: id, y, oof."""
    path = Path(experiments_dir) / run_id / "oof.parquet"
    return duckdb.sql(f"SELECT * FROM '{path}'").df()


def roc_points(y, p, n_points: int = 201) -> pd.DataFrame:
    """ROC curve resampled on an even FPR grid (keeps plots light for large data)."""
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0, 1, n_points)
    # np.interp takes the top of a vertical jump at a repeated FPR, so prepend the true
    # origin (0, 0) to keep the initial vertical segment.
    return pd.DataFrame({"fpr": np.r_[0.0, grid], "tpr": np.r_[0.0, np.interp(grid, fpr, tpr)]})


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed positive rate per equal-width probability bin."""
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    return (
        df.groupby("bin")
        .agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def score_histogram(y, p, labels: tuple[str, str], n_bins: int = 40) -> pd.DataFrame:
    """Share of each class per predicted-probability bin (pre-binned for plotting)."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for cls, label in enumerate(labels):
        counts, _ = np.histogram(p[y == cls], bins=edges)
        rows.append(pd.DataFrame({"bin_center": centers, "label": label, "share": counts / max(counts.sum(), 1)}))
    return pd.concat(rows, ignore_index=True)


def confusion_table(y, p, labels: tuple[str, str], threshold: float = 0.5) -> pd.DataFrame:
    """Counts and row shares (share of each actual class) at `threshold`."""
    cm = confusion_matrix(y, np.asarray(p) >= threshold, labels=[0, 1])
    rows = [
        {"actual": labels[a], "predicted": labels[b], "n": int(cm[a, b]), "share": cm[a, b] / cm[a].sum()}
        for a in (0, 1)
        for b in (0, 1)
    ]
    return pd.DataFrame(rows)


def score_by_segment(y, p, segments: pd.DataFrame) -> pd.DataFrame:
    """AUC, size and positive rate per value of each segment column."""
    df = segments.reset_index(drop=True).assign(_y=np.asarray(y), _p=np.asarray(p))
    rows = []
    for col in segments.columns:
        for value, g in df.groupby(col, observed=True):
            auc = roc_auc_score(g["_y"], g["_p"]) if g["_y"].nunique() == 2 else np.nan
            rows.append({"segment_col": col, "segment": str(value), "n": len(g), "pos_rate": g["_y"].mean(), "auc": auc})
    return pd.DataFrame(rows)


def diagnose_run(
    experiments_dir: Path,
    run_id: str,
    segment_frame: pd.DataFrame,
    id_col: str,
    metric: str = "roc_auc",
    labels: tuple[str, str] = ("not satisfied", "satisfied"),
    threshold: float = 0.5,
) -> RunDiagnostics:
    """
    Build all diagnostics for a saved run from its OOF predictions.

    `segment_frame` has `id_col` plus the columns to slice AUC by (e.g. class, type_of_travel).
    """
    oof = load_oof(experiments_dir, run_id)
    y, p = oof["y"].to_numpy(), oof["oof"].to_numpy()
    seg = segment_frame.set_index(id_col).loc[oof["id"]]
    folds = fold_scores(experiments_dir, metric)
    meta = json.loads((Path(experiments_dir) / run_id / "meta.json").read_text())
    scores = {
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "accuracy": accuracy_score(y, p >= threshold),
        "brier": brier_score_loss(y, p),
        "n_rows": len(y),
        "n_features": meta["n_features"],
    }
    return RunDiagnostics(
        name=meta["name"],
        run_id=run_id,
        scores=scores,
        roc=roc_points(y, p),
        calibration=calibration_table(y, p),
        histogram=score_histogram(y, p, labels),
        confusion=confusion_table(y, p, labels, threshold),
        segments=score_by_segment(y, p, seg),
        folds=folds[folds["run_id"] == run_id][["fold", metric]].reset_index(drop=True),
        coefs=load_coefficients(experiments_dir, run_id),
    )


def grid_table(board: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """
    Latest run of each grid point of `experiment` as columns: one per grid parameter, plus
    `penalty` (L2 / L1 / elastic net from l1_ratio), cv_mean, cv_std, fit_seconds.

    Grid runs are named "<experiment>[k=v,k=v,...]" by modeling.experiment.expand_grid.
    """
    runs = latest_runs(board[board["name"].str.startswith(f"{experiment}[")])
    params = runs["name"].str.extract(r"\[(.*)\]$")[0].str.split(",")
    rows = []
    for (_, r), kvs in zip(runs.iterrows(), params):
        p = dict(kv.split("=", 1) for kv in kvs)
        p = {k: (float(v) if v.replace(".", "", 1).isdigit() else v) for k, v in p.items()}
        ratio = p.get("l1_ratio", 0.0)
        penalty = "L2" if ratio == 0 else "L1" if ratio == 1 else f"elastic net ({ratio})"
        rows.append({**p, "penalty": penalty, "cv_mean": r["cv_mean"], "cv_std": r["cv_std"], "fit_seconds": r["fit_seconds"]})
    return pd.DataFrame(rows)
