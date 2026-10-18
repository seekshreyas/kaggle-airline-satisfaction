#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load experiment configs: resolve `extends`, merge, expand "@group" references.

    pixi run python -m src.utils.config show configs/experiments/lr_04_splines.yaml
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml

from src.utils.paths import project_root


def deep_merge(base: dict, child: dict) -> dict:
    """Merge `child` into `base`: dicts merge recursively, everything else (lists too) replaces."""
    out = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path | str) -> dict:
    """
    Load a YAML config and everything it `extends`, merged parent-first.

    Adds `lineage` (list of config names, root first) and `config_path`.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        merged, lineage = raw, []
    else:
        parent = load_config(path.parent / parent_ref)
        lineage = parent.pop("lineage")
        parent.pop("config_path")
        merged = deep_merge(parent, raw)
    merged["lineage"] = [*lineage, path.stem]
    merged["config_path"] = str(path)
    return merged


def expand_ref(value: Any, cfg: dict, derived: dict[str, list[str]] | None = None) -> Any:
    """
    Expand "@name" into a column list (or mapping).

    Looks in `derived` first (e.g. "derived_flags"), then `cfg["schema"]`, then top-level
    cfg keys (e.g. "rating_groups"). Non-reference values are returned unchanged.
    """
    if not (isinstance(value, str) and value.startswith("@")):
        return value
    name = value[1:]
    for source in (derived or {}, cfg.get("schema", {}), cfg):
        if name in source:
            return copy.deepcopy(source[name])
    raise KeyError(f"Unknown reference {value!r}")


def data_dir(cfg: dict) -> Path:
    """Raw data directory: $DIR_KAGGLE_REL if set, else cfg.data.dir (relative to root)."""
    rel = os.getenv("DIR_KAGGLE_REL") or cfg["data"]["dir"]
    return project_root() / rel.strip("/\\")


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved config (ignores lineage, path, name, description)."""
    ignore = {"lineage", "config_path", "name", "description"}
    payload = {k: v for k, v in cfg.items() if k not in ignore}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


app = typer.Typer(help=__doc__)


@app.callback()
def _main() -> None:
    """Keep subcommand style even with a single command."""


@app.command()
def show(path: Path = typer.Argument(..., help="Config YAML to resolve.")) -> None:
    """Print the fully merged config."""
    cfg = load_config(path)
    typer.echo(f"# hash: {config_hash(cfg)}")
    typer.echo(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    app()
