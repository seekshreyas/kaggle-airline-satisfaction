#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-run model quality for a binary classifier: ranking, calibration, scores, errors.
Each function takes RunDiagnostics (plus an optional comparison run) and returns a figure.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from src.viz.theme import BLUE_RAMP, CLASS_COLORS, REFERENCE_LINE, rgba, run_styles, style

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics


def roc_figure(run: "RunDiagnostics", other: "RunDiagnostics | None" = None) -> go.Figure:
    """ROC curve: how well scores rank satisfied above unsatisfied passengers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random (AUC 0.5)",
                             line=REFERENCE_LINE, hoverinfo="skip"))
    for r, line in run_styles(run, other):
        fig.add_trace(go.Scatter(
            x=r.roc["fpr"], y=r.roc["tpr"], mode="lines", line=line,
            name=f"{r.name} · AUC {r.scores['roc_auc']:.4f}",
            hovertemplate="FPR %{x:.2f}<br>TPR %{y:.3f}<extra>" + r.name + "</extra>",
        ))
    fig.update_xaxes(title="false positive rate", range=[0, 1])
    fig.update_yaxes(title="true positive rate", range=[0, 1.01])
    return style(fig, "ROC curve (OOF)")


def calibration_figure(run: "RunDiagnostics", other: "RunDiagnostics | None" = None) -> go.Figure:
    """Reliability: predicted probability vs observed satisfaction rate per bin."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated",
                             line=REFERENCE_LINE, hoverinfo="skip"))
    for r, line in run_styles(run, other):
        c = r.calibration
        fig.add_trace(go.Scatter(
            x=c["mean_pred"], y=c["frac_pos"], mode="lines+markers", line=line,
            marker={"size": 8, "color": line["color"]}, name=r.name, customdata=c["n"],
            hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>%{customdata:,} rows<extra>" + r.name + "</extra>",
        ))
    fig.update_xaxes(title="mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="observed satisfied rate", range=[0, 1.02])
    return style(fig, f"Calibration (Brier {run.scores['brier']:.4f})")


def score_distribution_figure(run: "RunDiagnostics") -> go.Figure:
    """Where each class's predicted probabilities fall; overlap = hard cases."""
    fig = go.Figure()
    for label, color in CLASS_COLORS.items():
        h = run.histogram[run.histogram["label"] == label]
        fig.add_trace(go.Scatter(
            x=h["bin_center"], y=h["share"], name=label, mode="lines", line_shape="hvh",
            line={"color": color, "width": 2.5}, fill="tozeroy", fillcolor=rgba(color, 0.18),
            hovertemplate="p ≈ %{x:.3f}<br>%{y:.1%} of class<extra>" + label + "</extra>",
        ))
    fig.update_xaxes(title="predicted probability of satisfied", range=[0, 1])
    fig.update_yaxes(title="share of class", tickformat=".0%")
    return style(fig, f"Score distribution by actual class · {run.name}")


def confusion_figure(run: "RunDiagnostics", threshold: float = 0.5) -> go.Figure:
    """Confusion matrix at the threshold; color = share of the actual class (row)."""
    c = run.confusion
    z = c.pivot(index="actual", columns="predicted", values="share")
    n = c.pivot(index="actual", columns="predicted", values="n").loc[z.index, z.columns]
    text = [[f"{n.iloc[i, j]:,}<br>{z.iloc[i, j]:.1%}" for j in range(z.shape[1])] for i in range(z.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f"Predicted {x}" for x in z.columns], y=[f"Actually {y}" for y in z.index],
        text=text, texttemplate="%{text}", colorscale=BLUE_RAMP, zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, f"Confusion at p ≥ {threshold} · accuracy {run.scores['accuracy']:.2%}", height=340)
