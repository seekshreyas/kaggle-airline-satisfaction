# Visualization conventions

## Plotting library (`src/viz/`)
- Plots live in `src/viz/`, split by job: `theme`, `export`, `classification`, `experiments`, `explain`, `segments`, `dashboards`. I want to grow this library.
- Every plot function takes data in (a tidy DataFrame or `RunDiagnostics`) and returns a `plotly.graph_objects.Figure`. No file I/O, and no dependency on `src.reporting` (pass label functions in).
- Finish every figure with `theme.style(fig, title)`, and use colors from `theme` only.
- **Add every new `*_figure` to `CATALOG` in `src/viz/__init__.py`.** The smoke test in `tests/test_viz.py` builds every catalogued plot and fails on any plot that isn't catalogued.

## Chart design
- Validate any new categorical palette with the dataviz palette checker; never pick colors by eye.
- Never rely on color alone: comparison runs are dashed gray (hatched in bar charts), verdicts use marker shapes, and there's always a legend for two or more series.
- Bars start at zero. If differences are small, use a dot plot with error bars rather than a truncated bar axis.
- Show what each idea added (its step gain), not cumulative gains that make dropped ideas look good.
- Use plain-language labels in anything for an audience (display names from `competition.yaml`, experiment `label`s), not internal run names.
- **Render and look at every chart** before shipping, checking for overlaps, clipped or missing labels, muddy overlapping colors, and misleading framing.

Reports and slides that use these charts follow `.agents/instructions/reporting.md`.
