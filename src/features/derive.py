#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic, stateless derive blocks (reusable across competitions).
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import pandas as pd

from src.features.registry import CAT, FLAG, register_derive


def _cols(cols: list[str], exclude: list[str] | None) -> list[str]:
    return [c for c in cols if c not in set(exclude or [])]


@register_derive("missing_flag")
def missing_flag(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """1 where the value is missing: flag__<col>__isna."""
    return pd.DataFrame({f"{FLAG}{c}__isna": df[c].isna().astype("int8") for c in cols}, index=df.index)


@register_derive("zero_flag")
def zero_flag(df: pd.DataFrame, cols: list[str], exclude: list[str] | None = None) -> pd.DataFrame:
    """1 where the value is 0 (e.g. rating 'not applicable'): flag__<col>__is0."""
    return pd.DataFrame(
        {f"{FLAG}{c}__is0": (df[c] == 0).astype("int8") for c in _cols(cols, exclude)}, index=df.index
    )


@register_derive("ge_flag")
def ge_flag(
    df: pd.DataFrame, cols: list[str], threshold: float, exclude: list[str] | None = None
) -> pd.DataFrame:
    """1 where value >= threshold: flag__<col>__ge<threshold>."""
    return pd.DataFrame(
        {f"{FLAG}{c}__ge{threshold}": (df[c] >= threshold).astype("int8") for c in _cols(cols, exclude)},
        index=df.index,
    )


@register_derive("cross")
def cross(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    """Combine two or more columns into one categorical: cat__<a>__x__<b>."""
    return pd.DataFrame(
        {
            f"{CAT}{'__x__'.join(p)}": df[p[0]]
            .astype(str)
            .str.cat([df[c].astype(str) for c in p[1:]], sep="|")
            for p in pairs
        },
        index=df.index,
    )
