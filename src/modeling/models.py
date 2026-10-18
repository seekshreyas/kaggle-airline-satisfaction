#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model registry: config `model.name` -> estimator. Add a model with one decorator.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from collections.abc import Callable

from sklearn.linear_model import LogisticRegression

MODELS: dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    def wrap(fn: Callable) -> Callable:
        MODELS[name] = fn
        return fn

    return wrap


@register_model("logreg")
def logreg(seed: int, **params) -> LogisticRegression:
    """Logistic regression; regularization via C and l1_ratio (penalty is deprecated)."""
    return LogisticRegression(random_state=seed, **params)


def build_model(model_cfg: dict, seed: int):
    name = model_cfg["name"]
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name](seed=seed, **(model_cfg.get("params") or {}))
