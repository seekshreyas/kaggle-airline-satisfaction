#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one experiment config end to end: rows -> pipeline -> k-fold CV -> artifacts + leaderboard.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import copy
import itertools
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import sklearn
import yaml
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import src.features  # noqa: F401  (registers feature blocks)
from src.features.build import build_preprocessor
from src.modeling.diagnostics import coefficients
from src.modeling.metrics import get_metrics
from src.modeling.models import build_model
from src.modeling.validate import CVResult, cross_validate_oof
from src.submission import make_submission, validate_submission, write_submission
from src.utils.config import config_hash
from src.utils.paths import project_root


@dataclass
class RunResult:
    run_id: str
    cfg: dict
    cv: CVResult
    coefs: pd.DataFrame
    row: dict = field(default_factory=dict)  # leaderboard row
    test_pred: pd.DataFrame | None = None  # id, pred (when predict_test)
    submission: pd.DataFrame | None = None  # validated, sample-submission order


def feature_columns(cfg: dict) -> list[str]:
    """All raw feature columns declared in the schema."""
    return [c for cols in cfg["schema"].values() for c in cols]


def select_rows(train: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rows and fold labels for the configured subset.

    dev  -> the dev subset with `fold` (Phases 2-3)
    full -> every non-holdout row with `fold_full` (Phase 4); the holdout is never included
    """
    subset = cfg.get("data", {}).get("subset", "dev")
    if subset == "dev":
        rows = train[train["is_dev"]]
        return rows, rows["fold"]
    if subset == "full":
        rows = train[~train["is_holdout"]]
        return rows, rows["fold_full"]
    raise ValueError(f"Unknown data.subset {subset!r} (use dev or full)")


def holdout_already_scored(cfg: dict) -> pd.DataFrame:
    """Leaderboard rows that already scored the holdout (it should be scored once)."""
    board = project_root() / cfg["paths"]["experiments_dir"] / "leaderboard.csv"
    if not board.exists():
        return pd.DataFrame()
    lb = pd.read_csv(board)
    return lb[lb["holdout_auc"].notna()] if "holdout_auc" in lb else pd.DataFrame()


def build_pipeline(cfg: dict) -> Pipeline:
    pre = build_preprocessor(cfg)
    return Pipeline([*pre.steps, ("model", build_model(cfg["model"], cfg["seed"]))])


def expand_grid(cfg: dict) -> list[dict]:
    """
    One config per model.grid combination (name gets a [k=v,...] suffix).

    `grid` is a dict (full product of its lists) or a list of such dicts (union of their
    products), e.g. to pair each penalty with a solver that supports it.
    """
    grid = (cfg.get("model") or {}).get("grid")
    if not grid:
        return [cfg]
    out = []
    for part in grid if isinstance(grid, list) else [grid]:
        keys = list(part)
        for values in itertools.product(*(part[k] for k in keys)):
            c = copy.deepcopy(cfg)
            c["model"]["params"] = {**(c["model"].get("params") or {}), **dict(zip(keys, values))}
            c["model"]["grid"] = None
            c["name"] = f"{cfg['name']}[{','.join(f'{k}={v}' for k, v in zip(keys, values))}]"
            out.append(c)
    return out


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_experiment(
    cfg: dict,
    train: pd.DataFrame,
    save: bool = True,
    test: pd.DataFrame | None = None,
    sample_submission: pd.DataFrame | None = None,
    allow_holdout_rescore: bool = False,
) -> RunResult:
    """
    CV the configured pipeline on the configured subset and (optionally) save the run.

    `train` is the labelled frame with split columns (output of assign_splits).
    Phase 4 flags in the config:
      score_holdout: true -> score the model fit on the subset on the holdout (once!)
      predict_test:  true -> refit on ALL labelled rows, predict `test`, build a
                             validated submission (needs `test` and `sample_submission`)
    """
    if cfg.get("score_holdout") and not allow_holdout_rescore:
        previous = holdout_already_scored(cfg)
        if not previous.empty:
            raise RuntimeError(
                f"Holdout already scored by {previous['run_id'].tolist()}; scoring it again turns it "
                "into a tuning set. Pass allow_holdout_rescore=True only if you mean it."
            )
    rows, folds = select_rows(train, cfg)
    X, y = rows[feature_columns(cfg)], rows[cfg["target"]]
    metric_names = [cfg["metric"], *cfg.get("extra_metrics", [])]
    pipe = build_pipeline(cfg)

    cv = cross_validate_oof(pipe, X, y, folds.to_numpy(), get_metrics(metric_names))
    fitted = clone(pipe).fit(X, y)  # on the whole subset, for coefficients
    coefs = coefficients(fitted)

    holdout_auc = None
    if cfg.get("score_holdout"):
        hold = train[train["is_holdout"]]
        holdout_auc = roc_auc_score(hold[cfg["target"]], fitted.predict_proba(hold[X.columns])[:, 1])

    test_pred = submission = None
    if cfg.get("predict_test"):
        if test is None or sample_submission is None:
            raise ValueError("predict_test needs `test` and `sample_submission`")
        final = clone(pipe).fit(train[X.columns], train[cfg["target"]])  # all labelled rows
        preds = final.predict_proba(test[X.columns])[:, 1]
        test_pred = pd.DataFrame({cfg["id_col"]: test[cfg["id_col"]].to_numpy(), "pred": preds})
        submission = make_submission(test[cfg["id_col"]], preds, sample_submission, cfg["id_col"], cfg["target"])
        validate_submission(submission, sample_submission, cfg.get("task", "binary"))

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}"
    oof_scores = {f"oof_{m}": f(y, cv.oof) for m, f in get_metrics(metric_names).items()}
    lineage = cfg.get("lineage", [])
    row = {
        "run_id": run_id,
        "name": cfg["name"],
        "parent": lineage[-2] if len(lineage) > 2 else None,
        "description": cfg.get("description", ""),
        "data_subset": cfg.get("data", {}).get("subset", "dev"),
        "n_rows": len(X),
        "n_features": len(coefs),
        "n_splits": cfg["cv"]["n_splits"],
        **cv.summary(cfg["metric"]),
        **oof_scores,
        "holdout_auc": holdout_auc,
        "submission_path": None,
        "salt": cfg["randomization"]["salt"],
        "config_hash": config_hash(cfg),
        "git_commit": _git_commit(),
        "public_lb": None,
    }
    result = RunResult(run_id, cfg, cv, coefs, row, test_pred, submission)
    if save:
        if submission is not None:
            path = project_root() / cfg["paths"]["submissions_dir"] / f"{run_id}.csv"
            row["submission_path"] = str(write_submission(submission, path).relative_to(project_root()))
        save_run(result, rows[cfg["id_col"]], y)
    return result


def save_run(result: RunResult, ids: pd.Series, y: pd.Series) -> Path:
    """Write the run folder and append to leaderboard.csv."""
    exp_dir = project_root() / result.cfg["paths"]["experiments_dir"]
    run_dir = exp_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(result.cfg, sort_keys=False))
    meta = {**result.row, "sklearn": sklearn.__version__, "saved_at": time.time()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    result.cv.fold_scores.to_csv(run_dir / "cv_scores.csv", index=False)
    result.coefs.to_csv(run_dir / "coefficients.csv", index=False)
    oof = pd.DataFrame({"id": ids.to_numpy(), "y": y.to_numpy(), "oof": result.cv.oof})  # noqa: F841
    duckdb.sql(f"COPY oof TO '{run_dir / 'oof.parquet'}' (FORMAT parquet)")
    if result.test_pred is not None:
        test_pred = result.test_pred  # noqa: F841
        duckdb.sql(f"COPY test_pred TO '{run_dir / 'test_pred.parquet'}' (FORMAT parquet)")

    append_leaderboard(exp_dir / "leaderboard.csv", result.row)
    return run_dir


def append_leaderboard(board: Path, row: dict) -> None:
    """Add a row, aligning columns with the existing file (new columns are added, never shifted)."""
    new = pd.DataFrame([row])
    if board.exists():
        new = pd.concat([pd.read_csv(board), new], ignore_index=True)
    new.to_csv(board, index=False)
