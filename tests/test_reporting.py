"""Runs the cases declared in tests/cases/reporting.yaml."""

import pytest
import yaml

from src.reporting.deckset import Deck, image, notes
from src.reporting.feature_report import _sentence_case
from src.reporting.labels import pretty_feature
from src.utils.config import load_config
from src.utils.paths import project_root
from tests.case_loader import CASES_DIR, load_cases

CASES = "reporting.yaml"
CFG = load_config(project_root() / "configs" / "competition.yaml")


def _params(section):
    return [pytest.param(c, id=str(next(iter(c.values())))) for c in yaml.safe_load((CASES_DIR / CASES).read_text())[section]]


@pytest.mark.parametrize("case", _params("pretty_feature"))
def test_pretty_feature(case):
    assert pretty_feature(case["feature"], CFG) == case["expected"]


@pytest.mark.parametrize("case", _params("sentence_case"))
def test_sentence_case(case):
    assert _sentence_case(case["label"]) == case["expected"]


def test_deck_markdown():
    case = yaml.safe_load((CASES_DIR / CASES).read_text())["deck"]
    deck = Deck(theme=case["theme"], footer=case["footer"])
    for s in case["slides"]:
        blocks = list(s["blocks"])
        if "image" in s:
            blocks.append(image(s["image"]["path"], s["image"]["alt"]))
        deck.slide(*blocks, notes(*s.get("notes", [])), commands=tuple(s.get("commands", [])))
    md = deck.render()
    assert md.startswith(case["expected_header"] + "\n\n"), "global commands first, no blank lines between"
    for snippet in case["expected_contains"]:
        assert snippet in md
    assert md.count("\n---\n") == len(case["slides"]) - 1
