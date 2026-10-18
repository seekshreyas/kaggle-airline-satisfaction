# Report and slide conventions

I present results to an executive audience as slides rendered with [Deckset](https://www.deckset.com/)
from markdown. A report is **PNG charts in `reports/plots/`** plus a **Deckset markdown deck in `reports/`**.

## Workflow
1. **Configure:** one YAML per report in `configs/reports/<name>.yaml` (title, subtitle, runs to compare, output paths, plot prefix, Deckset commands, image size).
2. **Generate:** `pixi run report configs/reports/<name>.yaml`. Builders live in `src/reporting/` (`deckset.py` for markdown, `labels.py` for plain language, one `*_report.py` per report type). Charts come from `src/viz/` and are exported with `viz.save_png`.
3. **Review:** open every PNG and read the whole deck before handing it over. Fix overlaps, clipped labels, internal names and overstated titles, then regenerate.
4. Regenerate after new runs rather than editing the markdown by hand: **every number in the narrative is computed from saved results**, never typed in.

## Deck structure
| # | Slide | Content |
|---|---|---|
| 1 | Title | Title, subtitle (phase · theme), month and year; footer and slide number hidden |
| 2 | **The bottom line** | 3–4 bullets: the headline result in plain terms, the key lever, the weak spot, what's next |
| 3 | Context | The question, the data, how the score works in plain words, how testing stays fair (columns work well here) |
| 4–n | One finding per slide | Action title + one chart |
| n+1 | Next steps | 3–4 concrete bullets |
| — | Appendix | Technical charts (ROC, calibration), method details, a table of everything tried |

## Writing for executives
- **Action titles:** each slide title states the takeaway ("One idea drove almost all of the gain"), not the topic ("Feature results").
- **Concise slide, detail in notes:** at most ~4 bullets or one chart on the slide; the explanation, caveats and exact numbers go in presenter notes (`^`).
- **Plain language:** explain metrics in everyday terms ("ranks a satisfied passenger above an unsatisfied one 95 times in 100"). Use experiment `label`s and feature `display_names`, never internal names (`lr_06_…`, `inflight_wifi_service_0`).
- **Sentence case** inside titles, but keep proper terms intact ("Wi-Fi", not "wi-Fi").
- **Honest framing:**
  - associations, not causes
  - say what was dropped and why
  - derive qualitative titles from the data (e.g. "within 7 points"), don't overstate ("can be taken at face value")
  - when two numbers are compared, say against what
- **Consistent vocabulary** across charts, tables and notes (e.g. adopted / alternative, not used / dropped everywhere).
- Round for the audience (95 in 100, 91%), with exact values in the notes or appendix.

## Charts for slides
- Export at 16:9-ish (default 1600×800, scale 2) with `for_slides` styling: white background, large type, no chart title (the slide title carries it).
- One message per chart. Choose the chart that answers the slide's claim: per-idea step gains, not cumulative ones; ranked segments with the weakest highlighted; plain-language driver labels.
- Name PNGs `<prefix>_<NN>_<chart>.png` so they sort in deck order; link them with paths relative to the deck file.
- Follow the chart-design rules in `.agents/instructions/visualization.md`.

## Deckset syntax (verified against docs.deckset.com)
- **Global commands** go on the first lines with **no blank lines between them**: `theme: <Name>, <swatch>` (optional; may be left to choose in the app), `footer: …`, `slidenumbers: true`, `autoscale: true`.
- **Slides** are separated by `---` on its own line, with blank lines around it.
- **Per-slide commands** go at the top of a slide: `[.hide-footer]`, `[.slidenumbers: false]`, `[.autoscale: true]`, `[.build-lists: true]`.
- **Headings:** `#` large, `##` regular, `###` small. `# [fit] Text` makes a heading fill the slide.
- **Images:** `![inline fill alt(Description)](plots/x.png)` for a chart under a title; `![right fit alt(…)](…)` for a split slide; `![fit decorative](…)` for a background. Always give informative images an `alt(...)` description.
- **Columns:** start each with `[.column]`; `[.column: width(3)]` for uneven widths.
- **Presenter notes:** paragraphs starting with `^`, one `^` per paragraph.
- **Tables:** standard markdown tables, with alignment via `:---` / `---:`.
- **Hide a slide** without deleting it by wrapping it in `<!-- … -->`.
- When unsure of syntax, check the current docs rather than guessing.
