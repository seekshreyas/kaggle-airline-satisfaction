#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Competition-specific derive blocks (airline satisfaction). Replace for a new competition.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import numpy as np
import pandas as pd

from src.features.registry import FLAG, NUM, register_derive

DEP, ARR = "departure_delay_in_minutes", "arrival_delay_in_minutes"


@register_derive("delay_features")
def delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Delay difference, log total delay and an any-delay flag (missing arrival = departure)."""
    arr = df[ARR].fillna(df[DEP])
    return pd.DataFrame(
        {
            f"{NUM}delay_diff": arr - df[DEP],
            f"{NUM}total_delay_log1p": np.log1p(df[DEP] + arr),
            f"{FLAG}is_delayed": ((df[DEP] > 0) | (arr > 0)).astype("int8"),
        },
        index=df.index,
    )


@register_derive("rating_aggregates")
def rating_aggregates(
    df: pd.DataFrame,
    cols: list[str],
    groups: dict[str, list[str]] | None = None,
    exclude_zero: bool = True,
    stats: list[str] = ("mean", "min", "max", "std"),
    counts: list[str] = ("n_zero", "n_ge4"),
) -> pd.DataFrame:
    """Rating stats over all ratings and per group (zeros = N/A excluded): num__rating_<stat>."""
    ratings = df[cols].astype("float64")
    valid = ratings.where(ratings > 0) if exclude_zero else ratings
    out = {f"{NUM}rating_{s}": valid.agg(s, axis=1) for s in stats}
    for name, members in (groups or {}).items():
        out[f"{NUM}grp_{name}_mean"] = valid[members].mean(axis=1)
    count_fns = {"n_zero": lambda r: (r == 0).sum(axis=1), "n_ge4": lambda r: (r >= 4).sum(axis=1)}
    for c in counts:
        out[f"{NUM}rating_{c}"] = count_fns[c](ratings)
    return pd.DataFrame(out, index=df.index)
