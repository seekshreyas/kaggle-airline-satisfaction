#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn model feature names into plain language for executive charts.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from src.features.registry import CAT, FLAG, NUM


def pretty_feature(name: str, cfg: dict) -> str:
    """
    "inflight_wifi_service_0" -> "Wi-Fi rating: not rated"
    "type_of_travel_Personal Travel" -> "Trip type: Personal Travel"
    "num__rating_mean" -> "Average rating"
    Uses cfg["display_names"]; falls back to a cleaned-up column name.
    """
    names = cfg.get("display_names", {})
    ordinal = set(cfg["schema"]["ordinal"])
    for prefix in (NUM, FLAG, CAT):
        name = name.removeprefix(prefix)
    if name in names:
        return names[name]
    # one-hot level: longest known column that prefixes the name
    for col in sorted(names, key=len, reverse=True):
        if name.startswith(col + "_"):
            level = name[len(col) + 1 :]
            if col in ordinal:
                level = "not rated" if level == "0" else f"rated {level}"
            return f"{names[col]}: {level}"
    return name.replace("__", " ").replace("_", " ").strip().capitalize()
