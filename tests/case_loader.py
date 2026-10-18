"""Load declarative test cases from tests/cases/*.yaml as pytest params."""

import builtins
import re
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest
import yaml

CASES_DIR = Path(__file__).parent / "cases"


def load_cases(file: str, section: str) -> list[Any]:
    """Return the cases under `section` in tests/cases/<file> as pytest params."""
    cases = yaml.safe_load((CASES_DIR / file).read_text())[section]
    return [pytest.param(case, id=case["id"]) for case in cases]


def expectation(case: dict) -> AbstractContextManager:
    """`pytest.raises(...)` if the case declares `raises`, otherwise a no-op."""
    if "raises" not in case:
        return nullcontext()
    spec = case["raises"]
    return pytest.raises(getattr(builtins, spec["error"]), match=re.escape(spec["match"]))
