#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn an experiment config into an sklearn preprocessing pipeline:

    derive (stateless blocks) -> encode (ColumnTransformer, one group per config key)
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from src.features.registry import DERIVED_GROUPS, get_derive, get_encode
from src.utils.config import expand_ref


class DeriveTransformer(TransformerMixin, BaseEstimator):
    """Apply derive blocks in order; each adds new columns. Learns nothing from data."""

    def __init__(self, blocks: tuple = ()):
        self.blocks = blocks

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(self._derive(X.head(5)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._derive(X)

    def _derive(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X
        for name, params in self.blocks:
            out = pd.concat([out, get_derive(name)(out, **params)], axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def resolve_derive(cfg: dict) -> tuple:
    """((block_name, params_with_refs_expanded), ...) in config order."""
    return tuple(
        (name, {k: expand_ref(v, cfg) for k, v in (params or {}).items()})
        for name, params in (cfg.get("derive") or {}).items()
    )


def _group_columns(ref, cfg: dict):
    """Schema refs -> column list; @derived_* -> selector evaluated at fit time."""
    if isinstance(ref, str) and ref.startswith("@") and ref[1:] in DERIVED_GROUPS:
        return make_column_selector(pattern=DERIVED_GROUPS[ref[1:]])
    return list(expand_ref(ref, cfg))


def _group_pipeline(steps: list, cfg: dict):
    if steps == ["passthrough"]:
        return "passthrough"
    parts = []
    for i, step in enumerate(steps):
        (name, params), = step.items()
        params = {k: expand_ref(v, cfg) for k, v in (params or {}).items()}
        parts.append((f"{i}_{name}", get_encode(name)(**params)))
    return Pipeline(parts)


def build_preprocessor(cfg: dict) -> Pipeline:
    """derive -> ColumnTransformer(encode groups), with pandas output for readable names."""
    encode = ColumnTransformer(
        [
            (group, _group_pipeline(spec["steps"], cfg), _group_columns(spec["cols"], cfg))
            for group, spec in cfg["encode"].items()
        ],
        remainder=cfg.get("remainder", "drop"),
        verbose_feature_names_out=False,
    )
    pre = Pipeline([("derive", DeriveTransformer(resolve_derive(cfg))), ("encode", encode)])
    return pre.set_output(transform="pandas")
