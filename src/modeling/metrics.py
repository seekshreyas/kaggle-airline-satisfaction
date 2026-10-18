#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metric registry: name -> f(y_true, p_pred) on predicted probabilities.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
    "accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
}


def get_metrics(names: list[str]) -> dict[str, Callable]:
    unknown = set(names) - set(METRICS)
    if unknown:
        raise KeyError(f"Unknown metrics {sorted(unknown)}; known: {sorted(METRICS)}")
    return {n: METRICS[n] for n in names}
