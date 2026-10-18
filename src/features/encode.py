#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stateful encode steps. Each returns an sklearn transformer that is fit inside the CV fold.
Steps with `cols` only touch those columns and pass the rest of the group through.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)

from src.features.registry import register_encode


def _on_cols(transformer, cols: list[str] | None):
    """Apply `transformer` to `cols` only; other columns pass through unchanged."""
    if not cols:
        return transformer
    return ColumnTransformer(
        [("sub", transformer, list(cols))], remainder="passthrough", verbose_feature_names_out=False
    )


@register_encode("impute")
def impute(strategy: str = "median", cols: list[str] | None = None):
    """Fill missing values (median, mean, most_frequent)."""
    return _on_cols(SimpleImputer(strategy=strategy), cols)


@register_encode("log1p")
def log1p(cols: list[str] | None = None):
    """log(1 + x) for skewed, non-negative columns."""
    return _on_cols(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), cols)


@register_encode("spline")
def spline(cols: list[str] | None = None, n_knots: int = 5, degree: int = 3):
    """B-spline basis (knots at quantiles) for non-linear numeric effects."""
    return _on_cols(SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile"), cols)


@register_encode("scale")
def scale(method: str = "standard", cols: list[str] | None = None):
    """Scale numeric columns (standard, robust, minmax)."""
    scalers = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    return _on_cols(scalers[method](), cols)


@register_encode("onehot")
def onehot(
    drop: str | None = None,
    handle_unknown: str = "ignore",
    min_frequency: int | float | None = None,
    cols: list[str] | None = None,
):
    """One-hot encode categories (unseen categories in test are ignored)."""
    enc = OneHotEncoder(
        drop=drop, handle_unknown=handle_unknown, min_frequency=min_frequency, sparse_output=False
    )
    return _on_cols(enc, cols)
