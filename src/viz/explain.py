#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
What the model learned: coefficients (technical) and drivers (plain language).
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from collections.abc import Callable
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from src.viz.theme import BLUE, RED, style

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics


def coefficients_figure(run: "RunDiagnostics", top: int = 20) -> go.Figure:
    """Largest coefficients (log-odds per scaled unit / per one-hot level)."""
    c = run.coefs.head(top).sort_values("coef")
    fig = go.Figure(go.Bar(
        x=c["coef"], y=c["feature"], orientation="h",
        marker={"color": [BLUE if v > 0 else RED for v in c["coef"]]},
        hovertemplate="%{y}<br>coef %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(128,128,128,0.6)", line_width=1)
    fig.update_xaxes(title="← less likely satisfied · coefficient · more likely satisfied →")
    return style(fig, f"Top {top} coefficients · {run.name}", height=max(380, 22 * top + 80))


def drivers_figure(run: "RunDiagnostics", label: Callable[[str], str] | None = None, top: int = 12) -> go.Figure:
    """Strongest signals, colored by direction; `label` turns feature names into plain language."""
    c = run.coefs.head(top).copy()
    c["label"] = [label(f) if label else f for f in c["feature"]]
    c = c.sort_values("coef")
    fig = go.Figure(go.Bar(
        x=c["coef"], y=c["label"], orientation="h",
        marker={"color": [BLUE if v > 0 else RED for v in c["coef"]]},
        hovertemplate="%{y}<br>%{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(128,128,128,0.6)", line_width=1)
    fig.update_xaxes(title="← less likely satisfied        more likely satisfied →", showticklabels=False)
    return style(fig, "Strongest signals", height=420)
