#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project path helpers.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """`$DIR_PROJECT` if set, otherwise the nearest parent containing pixi.toml."""
    env = os.getenv("DIR_PROJECT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pixi.toml").exists():
            return parent
    raise FileNotFoundError("Could not find the project root (no pixi.toml above).")
