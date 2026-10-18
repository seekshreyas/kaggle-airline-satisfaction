#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared look for every plot: the validated palette, base styling and comparison styles.

Colors (validated with the dataviz palette checker, light mode):
- selected run: blue; comparison run: gray dashed (identity never relies on color alone)
- classes: blue = satisfied, orange = not satisfied
- sign (diverging): blue = towards satisfied, red = away
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from collections.abc import Iterator
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics

BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#8a8984"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CLASS_COLORS = {"satisfied": BLUE, "not satisfied": ORANGE}
REFERENCE_LINE = {"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dot"}

VERDICT_STYLE = {  # verdict -> (color, marker symbol); symbol is the secondary encoding
    "reference": (GRAY, "diamond"),
    "keep": (BLUE, "circle"),
    "revert": (ORANGE, "x"),
}


def style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Shared look: transparent surface (works in light and dark), recessive grid."""
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        height=height,
        margin={"l": 50, "r": 20, "t": 60, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    """'#2a78d6', 0.2 -> 'rgba(42,120,214,0.2)'."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def run_styles(
    run: "RunDiagnostics", other: "RunDiagnostics | None"
) -> Iterator[tuple["RunDiagnostics", dict]]:
    """(diagnostics, line style) for the selected run and the optional comparison."""
    yield run, {"color": BLUE, "width": 2.5, "dash": "solid"}
    if other is not None:
        yield other, {"color": GRAY, "width": 2, "dash": "dash"}
