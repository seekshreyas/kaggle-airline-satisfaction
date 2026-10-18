"""Runs the cases declared in tests/cases/datawrangle.yaml."""

import pandas as pd
import pytest

from src.datahelpers import datawrangle as dw
from tests.case_loader import expectation, load_cases

CASES = "datawrangle.yaml"


def _columns(case: dict) -> list[str]:
    """Input columns: explicit `columns`, else the keys of `expected`."""
    return case.get("columns") or list(case["expected"])


@pytest.mark.parametrize("case", load_cases(CASES, "normalize_name"))
def test_normalize_name(case):
    with expectation(case):
        result = dw.normalize_name(case["name"], style=case.get("style", "simple"))
        assert result == case["expected"]


@pytest.mark.parametrize("case", load_cases(CASES, "normalize_columns"))
def test_normalize_columns(case):
    df = pd.DataFrame(columns=_columns(case))
    with expectation(case):
        result = dw.normalize_columns(
            df, style=case.get("style", "simple"), overrides=case.get("overrides")
        )
        assert list(result.columns) == list(case["expected"].values())


@pytest.mark.parametrize("case", load_cases(CASES, "column_mapping"))
def test_column_mapping(case):
    with expectation(case):
        result = dw.column_mapping(
            _columns(case), style=case.get("style", "simple"), overrides=case.get("overrides")
        )
        assert result == case["expected"]
