"""Runs the cases declared in tests/cases/splits.yaml."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.datahelpers import splits as sp
from tests.case_loader import CASES_DIR, expectation, load_cases

CASES = "splits.yaml"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    spec = yaml.safe_load((CASES_DIR / CASES).read_text())["synthetic"]
    ids = np.arange(spec["n_rows"])
    target = ((ids * 7) % 1000 < spec["positive_rate"] * 1000).astype(int)
    return pd.DataFrame({"id": ids, "target": target})


def _strata(case, df):
    return df["target"] if case["stratify"] else None


@pytest.mark.parametrize("case", load_cases(CASES, "hash_uniform"))
def test_hash_uniform(case):
    with expectation(case):
        salt = case["salt"]
        if "via_purpose" in case:
            salt = sp.purpose_salt(salt, case["via_purpose"])
        result = sp.hash_uniform(case["keys"], salt)
        assert np.round(result, case["decimals"]).tolist() == case["expected"]


@pytest.mark.parametrize("case", load_cases(CASES, "hash_split"))
def test_hash_split(case, data):
    with expectation(case):
        mask = sp.hash_split(data["id"], case["frac"], case["salt"], _strata(case, data))
        exp = case["expect"]
        if "frac_per_class" in exp:
            per_class = pd.Series(mask).groupby(data["target"]).mean()
            assert np.allclose(per_class, exp["frac_per_class"], atol=exp["tolerance"])
        else:
            assert abs(mask.mean() - exp["frac"]) <= exp["tolerance"]


@pytest.mark.parametrize("case", load_cases(CASES, "hash_folds"))
def test_hash_folds(case, data):
    folds = sp.hash_folds(data["id"], case["n_splits"], case["salt"], _strata(case, data))
    stats = data.groupby(folds)["target"].agg(["size", "mean"])
    exp = case["expect"]
    assert sorted(stats.index) == list(range(case["n_splits"]))
    assert stats["size"].max() - stats["size"].min() <= exp["max_size_diff"]
    assert stats["mean"].max() - stats["mean"].min() <= exp["max_rate_diff"]

    pairs = sp.fold_indices(folds)
    valid = np.sort(np.concatenate([v for _, v in pairs]))
    assert (valid == np.arange(len(data))).all(), "every row is validated exactly once"


@pytest.mark.parametrize("case", load_cases(CASES, "salt_properties"))
def test_salt_properties(case, data):
    folds_a = pd.Series(sp.hash_folds(data["id"], 4, case["salt_a"], data["target"]), data.index)
    df_b = data.sample(frac=1.0, random_state=0) if case.get("shuffle_rows") else data
    folds_b = pd.Series(sp.hash_folds(df_b["id"], 4, case["salt_b"], df_b["target"]), df_b.index)

    changed = (folds_a != folds_b.reindex(folds_a.index)).mean()
    exp = case["expect"]
    assert changed <= exp.get("changed_share_max", 1.0)
    assert changed >= exp.get("changed_share_min", 0.0)


@pytest.mark.parametrize("case", load_cases(CASES, "assign_splits"))
def test_assign_splits(case, data):
    out = sp.assign_splits(data, case["config"])
    exp = case["expect"]
    assert abs(out["is_holdout"].mean() - exp["holdout_share"]) <= exp["tolerance"]
    assert abs(out["is_dev"].mean() - exp["dev_share"]) <= exp["tolerance"]
    assert not (out["is_holdout"] & out["is_dev"]).any(), "holdout and dev overlap"
    assert ((out["fold"] >= 0) == out["is_dev"]).all(), "folds only on the dev subset"
    assert ((out["fold_full"] >= 0) == ~out["is_holdout"]).all(), "full folds on every non-holdout row"
    sizes = out.loc[~out["is_holdout"]].groupby("fold_full").size()
    assert sizes.max() - sizes.min() <= 2, "full folds balanced"
