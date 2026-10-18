"""Runs the cases declared in tests/cases/submission.yaml."""

import pandas as pd
import pytest
import yaml

from src.submission import make_submission, validation_errors
from tests.case_loader import CASES_DIR, expectation, load_cases

CASES = "submission.yaml"
SAMPLE = pd.DataFrame(yaml.safe_load((CASES_DIR / CASES).read_text())["sample_submission"])


@pytest.mark.parametrize("case", load_cases(CASES, "make_submission"))
def test_make_submission(case):
    with expectation(case):
        sub = make_submission(pd.Series(case["ids"]), case["preds"], SAMPLE, "id", "satisfaction")
        assert sub["id"].tolist() == SAMPLE["id"].tolist()
        assert sub["satisfaction"].tolist() == case["expected"]
        assert validation_errors(sub, SAMPLE) == []


@pytest.mark.parametrize("case", load_cases(CASES, "validate"))
def test_validation_errors(case):
    errors = validation_errors(pd.DataFrame(case["submission"]), SAMPLE)
    assert len(errors) == len(case["errors"])
    for found, fragment in zip(errors, case["errors"]):
        assert fragment in found
