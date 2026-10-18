#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Executive deck for the feature-engineering phase: PNG charts + Deckset markdown.

Every number in the narrative is computed from the saved experiment runs, so the deck can
be regenerated after new runs:

    pixi run report configs/reports/phase2_feature_engineering.yaml
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from src.modeling.diagnostics import (
    RunDiagnostics,
    diagnose_run,
    fold_scores,
    ladder,
    latest_runs,
    leaderboard,
)
from src.reporting.deckset import Deck, bullets, columns, image, notes
from src.reporting.labels import pretty_feature
from src.utils.config import load_config
from src.utils.paths import project_root
from src.viz import (
    calibration_figure,
    confusion_figure,
    drivers_figure,
    gains_figure,
    roc_figure,
    save_png,
    score_distribution_figure,
    segment_auc_figure,
)


@dataclass
class ReportData:
    """Everything the deck needs, computed once."""

    cfg: dict  # competition config
    report: dict  # report config
    table: pd.DataFrame  # ladder table
    labels: dict[str, str]  # experiment name -> plain-language label
    adopted: set[str]  # experiments on the final model's path
    winner: RunDiagnostics
    baseline: RunDiagnostics
    n_train: int
    n_dev: int


def experiment_labels(experiments_dir: Path) -> dict[str, str]:
    """name -> `label` from each experiment YAML."""
    out = {}
    for path in experiments_dir.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        out[raw["name"]] = raw.get("label", raw["name"])
    return out


def collect(report_cfg: dict, train: pd.DataFrame) -> ReportData:
    """Load runs and diagnose the baseline and the winner. `train` has the split columns."""
    root = project_root()
    cfg = load_config(root / "configs" / "competition.yaml")
    exp_dir = root / cfg["paths"]["experiments_dir"]
    board = leaderboard(exp_dir)
    table = ladder(board, cfg["keep_rule"], fold_scores(exp_dir, cfg["metric"]), cfg["metric"])
    latest = latest_runs(board).set_index("name")
    segments = train[[cfg["id_col"], "class", "type_of_travel", "customer_type"]]

    def _diag(name: str) -> RunDiagnostics:
        return diagnose_run(exp_dir, latest.loc[name, "run_id"], segments, cfg["id_col"], cfg["metric"])

    labels = experiment_labels(root / "configs" / "experiments")
    winner, baseline = _diag(report_cfg["winner"]), _diag(report_cfg["baseline"])
    winner.name = "Final model"
    baseline.name = "Baseline"
    lineage = load_config(root / "configs" / "experiments" / f"{report_cfg['winner']}.yaml")["lineage"]
    return ReportData(
        cfg=cfg,
        report=report_cfg,
        table=table,
        labels=labels,
        adopted=set(lineage),
        winner=winner,
        baseline=baseline,
        n_train=len(train),
        n_dev=int(train["is_dev"].sum()),
    )


def save_charts(d: ReportData) -> dict[str, Path]:
    """Write every chart used in the deck as PNG; returns name -> path."""
    root, r = project_root(), d.report
    out_dir = root / r["plots_dir"]
    size = r["image"]
    figs = {
        "gains": gains_figure(d.table, d.labels, r["baseline"], d.adopted),
        "scores": score_distribution_figure(d.winner),
        "confusion": confusion_figure(d.winner),
        "drivers": drivers_figure(d.winner, label=lambda f: pretty_feature(f, d.cfg), top=8),
        "segments": segment_auc_figure(d.winner, d.cfg.get("display_names")),
        "roc": roc_figure(d.winner, d.baseline),
        "calibration": calibration_figure(d.winner),
    }
    return {
        name: save_png(fig, out_dir / f"{r['plot_prefix']}_{i:02d}_{name}.png", **size)
        for i, (name, fig) in enumerate(figs.items(), start=1)
    }


def _source_label(feature: str, cfg: dict) -> str:
    """Display name of the column a feature came from (one-hot level stripped)."""
    return pretty_feature(feature, cfg).split(":")[0]


def _sentence_case(label: str) -> str:
    """Lowercase the first letter unless the first word is a proper term (e.g. "Wi-Fi")."""
    first = label.split()[0]
    return label if any(ch.isupper() for ch in first[1:]) else label[0].lower() + label[1:]


def build_deck(d: ReportData, charts: dict[str, Path]) -> str:
    """Compose the Deckset markdown. Image paths are relative to the deck file."""
    r, cfg, w, b = d.report, d.cfg, d.winner, d.baseline
    deck_dir = (project_root() / r["output"]).parent
    img = {k: os.path.relpath(v, deck_dir) for k, v in charts.items()}

    # --- numbers used in the narrative -------------------------------------------------
    auc_w, auc_b = w.scores["roc_auc"], b.scores["roc_auc"]
    tested = d.table[d.table["name"] != r["baseline"]]
    adopted = tested[tested["name"].isin(d.adopted)]
    best_step = tested.loc[tested["delta"].idxmax()]
    conf = w.confusion.set_index(["actual", "predicted"])["share"]
    missed_sat = conf[("satisfied", "not satisfied")]
    false_sat = conf[("not satisfied", "satisfied")]
    seg = w.segments.dropna(subset=["auc"])
    weak, strong = seg.loc[seg["auc"].idxmin()], seg.loc[seg["auc"].idxmax()]
    names = cfg.get("display_names", {})
    weak_label = f"{weak['segment']} ({names.get(weak['segment_col'], weak['segment_col']).lower()})"
    drivers = list(dict.fromkeys(_source_label(f, cfg) for f in w.coefs["feature"].head(8)))[:3]
    drivers = [drivers[0]] + [_sentence_case(x) for x in drivers[1:]]
    best_label = d.labels.get(best_step["name"], best_step["name"])

    def decision(name: str, verdict: str) -> str:
        if name == r["baseline"]:
            return "starting point"
        if name in d.adopted:
            return "adopted"
        return "alternative, not used" if verdict == "keep" else "dropped"
    calib = w.calibration[w.calibration["n"] >= 1000]
    calib_gap = (calib["mean_pred"] - calib["frac_pos"]).abs().max()
    k = cfg["cv"]["n_splits"]
    holdout_pct = int(cfg["splits"]["holdout_frac"] * 100)

    deck = Deck(**{k_: v for k_, v in r["deckset"].items() if k_ in ("theme", "footer", "slidenumbers", "autoscale")})

    # 1 · Title
    deck.slide(
        f"# {r['title']}",
        f"### {r['subtitle']}",
        f"{date.today():%B %Y}",
        notes(
            "This deck summarizes the feature-engineering phase: which ideas improved the "
            "prediction, how good the model is today, and where it still struggles."
        ),
        commands=("[.hide-footer]", "[.slidenumbers: false]"),
    )

    # 2 · Bottom line
    deck.slide(
        "## The bottom line",
        bullets(
            f"The model ranks a satisfied passenger above an unsatisfied one **{auc_w * 100:.0f} times in 100** "
            f"(up from {auc_b * 100:.0f} with a basic setup)",
            f"It labels **{w.scores['accuracy']:.0%}** of passengers correctly",
            f"One idea did most of the work: **{best_label.lower()}**",
            f"Weak spot: **{weak_label}** passengers (ranking score {weak['auc']:.2f}), the next focus",
        ),
        notes(
            f"Score: ROC AUC {auc_w:.4f} vs {auc_b:.4f} for the baseline, measured on passengers "
            "the model did not see during training.",
            f"{len(tested)} feature ideas were tested; {len(adopted)} are in the final model.",
        ),
    )

    # 3 · What we predict and how we test
    deck.slide(
        "## What we predict, and how we keep score",
        columns(
            "**The question**\n\n"
            f"Will a passenger report being *satisfied*?\n\n"
            f"{d.n_train:,} past passengers · 21 inputs: 13 service ratings (0-5), "
            "trip details, delays and passenger profile",
            "**The score: ROC AUC**\n\n"
            "Pick one satisfied and one unsatisfied passenger at random. AUC is how often the "
            "model ranks the satisfied one higher.\n\n0.5 = coin flip · 1.0 = perfect",
            "**Fair testing**\n\n"
            f"{holdout_pct}% of passengers are locked away for a final check.\n\n"
            f"Ideas are compared on a {d.n_dev / 1000:.0f}k sample, scored {k} times on "
            "passengers each model never saw.",
        ),
        notes(
            "All comparisons use the same passengers and the same test splits, so differences "
            "between ideas are real, not luck of the draw.",
            "The model is deliberately simple (logistic regression) in this phase, so every "
            "gain comes from how the data is presented to it.",
        ),
    )

    # 4 · What worked
    deck.slide(
        "## One idea drove almost all of the gain",
        f"**{best_label}**",
        image(img["gains"], "Bar chart of AUC improvement over baseline for each idea tested"),
        notes(
            f"Biggest gain: +{best_step['delta'] * 100:.2f} AUC points. Ratings behave in "
            "steps, not straight lines: a Wi-Fi rating of 0 means 'not rated', and those "
            "passengers are mostly satisfied. A rating of 3 is much closer to 1 than to 5.",
            "In the final model: " + ", ".join(d.labels.get(n, n) for n in adopted["name"]) + ".",
            "An idea is adopted only if it adds at least 0.05 AUC points, and more than its "
            "normal variation across the test splits. Everything else is dropped to keep the model simple.",
        ),
    )

    # 5 · Confidence
    deck.slide(
        "## Most passengers get a confident, correct score",
        image(img["scores"], "Histogram of predicted satisfaction probability for satisfied and unsatisfied passengers"),
        notes(
            "Each curve shows where passengers' predicted probability of satisfaction falls. "
            "Satisfied passengers (blue) cluster near 1, unsatisfied (orange) near 0.",
            "The overlap in the middle is where the model is unsure, and where its mistakes are.",
        ),
    )

    # 6 · Accuracy
    deck.slide(
        f"## {w.scores['accuracy']:.0%} of passengers are labeled correctly",
        image(img["confusion"], "Confusion matrix of actual versus predicted satisfaction"),
        notes(
            f"{missed_sat:.0%} of satisfied passengers are labeled unsatisfied, and {false_sat:.0%} "
            "of unsatisfied passengers are labeled satisfied (at a 50% cut-off).",
            "The cut-off can be moved to favor catching unhappy passengers, at the cost of more false alarms.",
        ),
    )

    # 7 · Drivers
    deck.slide(
        f"## {', '.join(drivers[:-1])} and {drivers[-1]} carry the strongest signals",
        image(img["drivers"], "Bar chart of the strongest positive and negative signals in the model"),
        notes(
            "Blue pushes towards 'satisfied', red away from it. Personal (leisure) trips and "
            "disloyal customers lean unsatisfied; a top Wi-Fi rating leans satisfied.",
            "These are associations in the data, not causes, and each one is measured with all the "
            "others held fixed.",
        ),
    )

    # 8 · Where it struggles
    deck.slide(
        f"## Weakest on {weak_label} passengers",
        image(img["segments"], "Bar chart of ranking quality within each passenger segment"),
        notes(
            f"Ranking quality for {weak_label} passengers is {weak['auc']:.2f}, against "
            f"{strong['auc']:.2f} for the strongest segment. Only {weak['pos_rate']:.0%} of them are "
            "satisfied, so the few happy ones are hard to pick out.",
            "The next idea tested targets this group directly: letting ratings matter differently by trip type.",
        ),
    )

    # 9 · Next steps
    deck.slide(
        "## Next steps",
        bullets(
            f"Target the weak spot: let rating effects differ for {weak['segment'].lower()} passengers",
            "Tune how strongly the model is constrained",
            f"Confirm on all data, then once on the locked-away {holdout_pct}%",
            "Submit the first competition entry",
        ),
        notes("Each step changes one thing at a time, so we always know what helped."),
    )

    # Appendix
    deck.slide("# Appendix", commands=("[.hide-footer]",))
    deck.slide(
        "## Model vs baseline: ranking quality",
        image(img["roc"], "ROC curves of the final model and the baseline"),
        notes("A curve closer to the top-left corner ranks better. The area under it is the AUC."),
    )
    deck.slide(
        "## Predicted probabilities can be taken at face value"
        if calib_gap < 0.03
        else f"## Predicted probabilities are close to reality (within {calib_gap * 100:.0f} points)",
        image(img["calibration"], "Calibration curve of predicted versus observed satisfaction rate"),
        notes(
            f"Points on the dotted diagonal would be perfect. The largest gap across well-populated "
            f"bins is {calib_gap * 100:.1f} percentage points: mid-range scores are slightly too "
            "cautious, which matters little for ranking but could be corrected if probabilities are used directly."
        ),
    )
    rows = "\n".join(
        f"| {d.labels.get(t['name'], t['name'])} | {t['cv_mean']:.4f} | "
        f"{'' if pd.isna(t['delta']) else f'+{max(t["delta"], 0) * 100:.2f}'} | {decision(t['name'], t['verdict'])} |"
        for _, t in d.table.iterrows()
    )
    deck.slide(
        "## All ideas tested",
        "| Idea | AUC | Change (pts) | Decision |\n|:---|---:|---:|:---|\n" + rows,
        notes(
            f"AUC is the average over {k} test splits on the {d.n_dev / 1000:.0f}k sample. Change is "
            "measured against the step each idea built on. 'Alternative, not used' beat the baseline "
            "but lost to a better idea for the same job."
        ),
    )
    return deck.render()


def generate(report_path: Path, train: pd.DataFrame) -> tuple[Path, dict[str, Path]]:
    """Write the PNGs and the markdown deck; returns (deck path, chart paths)."""
    report_cfg = yaml.safe_load(Path(report_path).read_text())
    data = collect(report_cfg, train)
    charts = save_charts(data)
    out = project_root() / report_cfg["output"]
    out.write_text(build_deck(data, charts))
    return out, charts
