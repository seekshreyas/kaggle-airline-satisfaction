#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Registry of feature blocks, so YAML configs can refer to them by name.

- derive blocks: stateless, row-wise. f(df, **params) -> DataFrame of NEW columns.
  New column names start with a kind prefix (flag__, num__, cat__), which is how
  "@derived_flags" / "@derived_numeric" / "@derived_categorical" find them.
- encode steps: stateful. f(**params) -> sklearn transformer, fit inside each CV fold.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from collections.abc import Callable

import pandas as pd

FLAG, NUM, CAT = "flag__", "num__", "cat__"

# "@derived_<kind>" -> regex on derived column names
DERIVED_GROUPS = {
    "derived_flags": f"^{FLAG}",
    "derived_numeric": f"^{NUM}",
    "derived_categorical": f"^{CAT}",
}

DERIVE: dict[str, Callable[..., pd.DataFrame]] = {}
ENCODE: dict[str, Callable[..., object]] = {}


def register_derive(name: str) -> Callable:
    """Register a derive block under `name`."""

    def wrap(fn: Callable) -> Callable:
        DERIVE[name] = fn
        return fn

    return wrap


def register_encode(name: str) -> Callable:
    """Register an encode step under `name`."""

    def wrap(fn: Callable) -> Callable:
        ENCODE[name] = fn
        return fn

    return wrap


def _get(registry: dict, kind: str, name: str) -> Callable:
    if name not in registry:
        raise KeyError(f"Unknown {kind} block {name!r}; known: {sorted(registry)}")
    return registry[name]


def get_derive(name: str) -> Callable[..., pd.DataFrame]:
    return _get(DERIVE, "derive", name)


def get_encode(name: str) -> Callable[..., object]:
    return _get(ENCODE, "encode", name)


def list_blocks() -> pd.DataFrame:
    """Catalogue of registered blocks (kind, name, first docstring line)."""
    rows = [
        {"kind": kind, "name": name, "description": (fn.__doc__ or "").strip().split("\n")[0]}
        for kind, registry in (("derive", DERIVE), ("encode", ENCODE))
        for name, fn in registry.items()
    ]
    return pd.DataFrame(rows)
