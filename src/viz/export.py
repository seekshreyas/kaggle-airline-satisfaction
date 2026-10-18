#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export figures for slides and documents.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from pathlib import Path

import plotly.graph_objects as go


def for_slides(fig: go.Figure, font_size: int = 22) -> go.Figure:
    """Restyle a figure for a slide: white surface, larger type, no title (the slide has one)."""
    fig.update_layout(
        title=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": font_size, "color": "#0b0b0b"},
        margin={"l": 80, "r": 40, "t": 30, "b": 80},
        legend={"font": {"size": font_size - 2}},
    )
    fig.update_annotations(font_size=font_size)
    return fig


def save_png(fig: go.Figure, path: Path, width: int = 1600, height: int = 800, scale: int = 2) -> Path:
    """Write a slide-styled PNG (the input figure is left unchanged); returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for_slides(go.Figure(fig)).write_image(str(path), width=width, height=height, scale=scale)
    return path
