#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model quality within segments (e.g. cabin class, trip type, customer type).
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import BLUE, ORANGE, run_styles, style

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics


def segments_figure(run: "RunDiagnostics", other: "RunDiagnostics | None" = None) -> go.Figure:
    """AUC within each segment value, one panel per segment column (notebook view)."""
    cols = list(dict.fromkeys(run.segments["segment_col"]))
    fig = make_subplots(rows=1, cols=len(cols), shared_yaxes=True, subplot_titles=cols, horizontal_spacing=0.04)
    for i, col in enumerate(cols, start=1):
        for r, line in run_styles(run, other):
            s = r.segments[r.segments["segment_col"] == col]
            fig.add_trace(go.Bar(
                x=s["segment"], y=s["auc"], name=r.name, legendgroup=r.name, showlegend=i == 1,
                marker={"color": line["color"], "pattern": {"shape": "/" if line["dash"] == "dash" else ""}},
                customdata=s[["n", "pos_rate"]].values,
                hovertemplate="%{x}<br>AUC %{y:.4f}<br>%{customdata[0]:,} rows · %{customdata[1]:.1%} satisfied<extra>" + r.name + "</extra>",
            ))
    lo = min(s["auc"].min() for s in [run.segments] + ([other.segments] if other else []))
    fig.update_yaxes(range=[max(0.5, lo - 0.02), 1.0], tickformat=".2f")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return style(fig, "ROC AUC within segments", height=400)


def segment_auc_figure(run: "RunDiagnostics", display_names: dict[str, str] | None = None) -> go.Figure:
    """AUC per segment value, ranked, weakest highlighted, size in the label (deck view)."""
    names = display_names or {}
    s = run.segments.copy()
    s["label"] = [f"{names.get(c, c)}: {v}" for c, v in zip(s["segment_col"], s["segment"])]
    s = s.sort_values("auc")
    weakest = s["auc"].idxmin()
    fig = go.Figure(go.Bar(
        x=s["auc"], y=s["label"], orientation="h",
        marker={"color": [ORANGE if i == weakest else BLUE for i in s.index]},
        text=[f"{a:.2f}  ·  {n / 1000:.0f}k passengers" for a, n in zip(s["auc"], s["n"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="ROC AUC within segment (1.0 = perfect ranking)", range=[0.5, 1.12], tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.update_layout(showlegend=False)
    return style(fig, "Ranking quality by segment", height=420)
