#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Across runs: the feature ladder, paired per-fold gains, per-fold scores, gain per idea.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.viz.theme import BLUE, GRAY, VERDICT_STYLE, run_styles, style

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics


def ladder_figure(ladder_table: pd.DataFrame) -> go.Figure:
    """CV AUC per rung as points with fold-std error bars (a dot plot: zoomed axis is honest)."""
    fig = go.Figure()
    for verdict, (color, symbol) in VERDICT_STYLE.items():
        t = ladder_table[ladder_table["verdict"] == verdict]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["name"], y=t["cv_mean"], mode="markers+text", name=verdict,
            error_y={"type": "data", "array": t["cv_std"], "color": color, "thickness": 1.5, "width": 6},
            marker={"color": color, "size": 12, "symbol": symbol, "line": {"width": 2, "color": color}},
            text=t["cv_mean"].map("{:.4f}".format), textposition="middle right",
            customdata=t[["delta", "n_features"]].values,
            hovertemplate="%{x}<br>AUC %{y:.5f}<br>Δ vs parent %{customdata[0]:+.5f}"
                          "<br>%{customdata[1]} features<extra>%{fullData.name}</extra>",
        ))
    fig.update_xaxes(categoryorder="array", categoryarray=sorted(ladder_table["name"]))
    fig.update_yaxes(title="4-fold CV ROC AUC", tickformat=".3f")
    return style(fig, "CV ROC AUC per rung (error bar = fold std)", height=420)


def paired_deltas_figure(deltas: pd.DataFrame, min_gain: float) -> go.Figure:
    """Per-fold gain vs parent; points above the dashed line beat min_gain on that fold."""
    fig = go.Figure(go.Scatter(
        x=deltas["name"], y=deltas["delta"], mode="markers", name="fold",
        marker={"color": BLUE, "size": 11, "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
        customdata=deltas[["fold", "parent"]].values,
        hovertemplate="%{x}<br>fold %{customdata[0]}: Δ %{y:+.5f} vs %{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=min_gain, line_dash="dash", line_color=GRAY, annotation_text=f"min_gain {min_gain}")
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.6)", line_width=1)
    fig.update_yaxes(title="Δ AUC per fold vs parent", tickformat="+.4f")
    fig.update_layout(showlegend=False)
    return style(fig, "Per-fold AUC gain vs parent (same folds ⇒ paired)", height=380)


def folds_figure(run: "RunDiagnostics", other: "RunDiagnostics | None" = None, metric: str = "roc_auc") -> go.Figure:
    """Per-fold score; with a comparison run, paired lines show if the gain is consistent."""
    fig = go.Figure()
    for r, line in run_styles(run, other):
        fig.add_trace(go.Scatter(
            x=r.folds["fold"], y=r.folds[metric], mode="lines+markers", name=r.name,
            line=line, marker={"size": 10, "color": line["color"]},
            hovertemplate="fold %{x}<br>AUC %{y:.5f}<extra>" + r.name + "</extra>",
        ))
    fig.update_xaxes(title="fold", dtick=1)
    fig.update_yaxes(title="ROC AUC", tickformat=".4f")
    return style(fig, "Per-fold ROC AUC")


def gains_figure(ladder_table: pd.DataFrame, labels: dict[str, str], baseline: str, adopted: set[str]) -> go.Figure:
    """Each idea's own gain over the step it built on (bars start at zero); adopted path in blue."""
    t = ladder_table[ladder_table["name"] != baseline].copy()
    t["gain"] = (t["delta"] * 100).clip(lower=0)  # AUC points; tiny negatives shown as 0
    t["label"] = t["name"].map(labels).fillna(t["name"])
    t = t.sort_values("gain")
    is_adopted = t["name"].isin(adopted)

    def tag(row, used: bool) -> str:
        if used:
            return "  ✓ adopted"
        return "  alternative, not used" if row["verdict"] == "keep" else "  dropped"

    fig = go.Figure(go.Bar(
        x=t["gain"], y=t["label"], orientation="h",
        marker={"color": [BLUE if a else GRAY for a in is_adopted],
                "pattern": {"shape": ["" if a else "/" for a in is_adopted]}},
        text=[f"+{g:.2f}{tag(r, a)}" for (_, r), g, a in zip(t.iterrows(), t["gain"], is_adopted)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.3f} AUC points vs the step it built on<extra></extra>",
    ))
    fig.update_xaxes(title="Gain over the step it built on (AUC points)", range=[0, t["gain"].max() * 1.6])
    fig.update_layout(showlegend=False)
    return style(fig, "What each idea added", height=420)


def grid_figure(grid_table: pd.DataFrame, param: str = "C", series: str = "penalty") -> go.Figure:
    """Hyperparameter grid: CV AUC vs `param` (log scale), one line per `series`, fold-std bars."""
    fig = go.Figure()
    styles = [(BLUE, "solid", "circle"), (GRAY, "dash", "square")]
    for (name, t), (color, dash, symbol) in zip(grid_table.groupby(series, sort=True), styles):
        t = t.sort_values(param)
        fig.add_trace(go.Scatter(
            x=t[param], y=t["cv_mean"], mode="lines+markers", name=str(name),
            line={"color": color, "dash": dash, "width": 2.5},
            marker={"color": color, "size": 10, "symbol": symbol},
            error_y={"type": "data", "array": t["cv_std"], "color": color, "thickness": 1.2, "width": 5},
            hovertemplate=f"{param} = %{{x}}<br>AUC %{{y:.5f}}<extra>{name}</extra>",
        ))
    best = grid_table.loc[grid_table["cv_mean"].idxmax()]
    fig.add_annotation(x=np.log10(best[param]), y=best["cv_mean"], text=f"best {best['cv_mean']:.5f}",
                       showarrow=True, arrowhead=0, ay=-35, font={"size": 12})
    fig.update_xaxes(title=f"{param} (log scale; larger = weaker regularization)", type="log")
    fig.update_yaxes(title="CV ROC AUC", tickformat=".4f")
    return style(fig, f"Regularization grid: CV AUC by {param}", height=400)
