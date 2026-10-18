#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-fold cross-validation with out-of-fold (OOF) predictions.
Folds come from the salted-hash `fold` column, so every experiment uses identical folds.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.datahelpers.splits import fold_indices


@dataclass
class CVResult:
    fold_scores: pd.DataFrame  # one row per fold, one column per metric
    oof: np.ndarray  # OOF probability for every row
    fit_seconds: float

    def summary(self, primary: str) -> dict:
        return {
            "cv_mean": self.fold_scores[primary].mean(),
            "cv_std": self.fold_scores[primary].std(ddof=0),
            "fit_seconds": round(self.fit_seconds, 1),
        }


def cross_validate_oof(
    pipe, X: pd.DataFrame, y: pd.Series, folds: np.ndarray, metrics: dict[str, Callable]
) -> CVResult:
    """Fit a fresh clone of `pipe` per fold; score the held-out fold; collect OOF predictions."""
    oof = np.full(len(y), np.nan)
    rows, start = [], time.perf_counter()
    for k, (tr, va) in enumerate(fold_indices(np.asarray(folds))):
        model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        rows.append({"fold": k, "n_valid": len(va), **{m: f(y.iloc[va], oof[va]) for m, f in metrics.items()}})
    return CVResult(pd.DataFrame(rows), oof, time.perf_counter() - start)
