"""Runs the cases declared in tests/cases/config.yaml."""

import pytest
import yaml

from src.utils import config as uc
from src.utils.paths import project_root
from tests.case_loader import expectation, load_cases

CASES = "config.yaml"
COMPETITION = yaml.safe_load((project_root() / "configs" / "competition.yaml").read_text())


def _get(cfg: dict, dotted: str):
    for part in dotted.split("."):
        cfg = cfg[part]
    return cfg


@pytest.mark.parametrize("case", load_cases(CASES, "deep_merge"))
def test_deep_merge(case):
    assert uc.deep_merge(case["base"], case["child"]) == case["expected"]


@pytest.mark.parametrize("case", load_cases(CASES, "expand_ref"))
def test_expand_ref(case):
    with expectation(case):
        result = uc.expand_ref(case["value"], COMPETITION, case.get("derived"))
        if "expected_keys" in case:
            assert list(result) == case["expected_keys"]
        else:
            assert result == case["expected"]


@pytest.mark.parametrize("case", load_cases(CASES, "load_config"))
def test_load_config(case):
    cfg = uc.load_config(project_root() / case["path"])
    for dotted, expected in case["expected"].items():
        assert _get(cfg, dotted) == expected, dotted
