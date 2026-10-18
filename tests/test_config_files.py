"""Structural checks on configs/competition.yaml and configs/experiments/*.yaml."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
COMPETITION = ROOT / "configs" / "competition.yaml"
EXPERIMENTS = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
DERIVED_GROUPS = {"derived_flags", "derived_numeric", "derived_categorical"}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _refs(node) -> set[str]:
    """All "@name" strings anywhere in a config."""
    if isinstance(node, str):
        return {node[1:]} if node.startswith("@") else set()
    if isinstance(node, dict):
        return set().union(*map(_refs, node.values()), set())
    if isinstance(node, list):
        return set().union(*map(_refs, node), set())
    return set()


def test_competition_cv_defaults_to_4_folds():
    cv = _load(COMPETITION)["cv"]
    assert cv["n_splits"] == 4


def test_randomization_uses_a_configured_salt():
    rnd = _load(COMPETITION)["randomization"]
    assert isinstance(rnd["salt"], str) and rnd["salt"].strip()
    assert rnd["key"] == _load(COMPETITION)["id_col"]


def test_schema_matches_real_headers():
    """Schema columns = the snake_case headers declared in tests/cases, minus id/target."""
    comp = _load(COMPETITION)
    cases = _load(ROOT / "tests" / "cases" / "datawrangle.yaml")["normalize_columns"]
    headers = next(c for c in cases if c["id"] == "airline_competition_headers")
    features = set(headers["expected"].values()) - {comp["id_col"], comp["target"]}

    schema_cols = [c for cols in comp["schema"].values() for c in cols]
    assert len(schema_cols) == len(set(schema_cols)), "a column is in two schema groups"
    assert set(schema_cols) == features


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_structure(path):
    cfg = _load(path)
    assert cfg["name"] == path.stem
    assert (path.parent / cfg["extends"]).resolve().exists()

    known = set(_load(COMPETITION)["schema"]) | {"rating_groups"} | DERIVED_GROUPS
    assert _refs(cfg) <= known, f"unknown @refs: {_refs(cfg) - known}"
