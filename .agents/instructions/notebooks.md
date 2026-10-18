# Notebook conventions

- Emphasis on readability and explanability of the notebook
- I prefer using marimo notebooks and prefer to present my output as slide or app mode. Hence when collaborating on notebook emphasize storytelling of the output.
- Before creating, editing, or debugging a marimo notebook, consult `references/marimo-llms.txt` for current documentation relevant to the task; do not rely only on pretrained knowledge for marimo APIs, CLI options, configuration, or supported workflows.
- `references/marimo-llms.txt` is a documentation index, not the full reference. Follow its relevant official documentation links when details are missing, and check behavior against the marimo version available in the project's Pixi environment. Do not assume undocumented APIs or change dependencies just to match a documentation example.
- Keep reusable computation in src/ rather than duplicating notebook code. I like keeping my notebook clean with very short, succinct code. Bigger code chunks should be in src/ folder for reusability and refactoring.
- Separate data loading, transformations, and presentation.
- Do not embed secrets or private data in notebook source.
- Explain metric definitions, units, and analytical assumptions.
- Preserve the project's existing Marimo structure and conventions.

## Storytelling structure
- **Notebook first, then promote.** Prototype a step in a cell, check it on real data, move the code to `src/` with a test, and replace the cell with a 1–3 line call.
- Organize by phase: each section opens with a markdown cell (code hidden) that says *what* is being done and *why*, with the setup (data, model, metric) and the decision rule.
- Follow each chart with a one- or two-sentence takeaway. Use `mo.stat` for headline numbers, `mo.callout` for decisions or caveats, and `mo.ui.table` for detail.
- Charts come from `src/viz/` (no inline styling); see `.agents/instructions/visualization.md`.
- Put expensive steps (CV runs, full fits) behind `mo.ui.run_button` so they don't re-run on every change.
- Interactive pickers (e.g. run vs comparison run) beat many static copies of the same chart.

## Driving a live marimo session (agents)
- I run `marimo edit --watch` with the notebook open in a browser. **Editing an existing cell's code can be silently reverted** by the browser's copy. To change a cell, create the replacement next to it and delete the old one, then verify the change reached the file on disk.
- Autoreload is off. After changing `src/`, reload the modules deepest-first (submodules before packages) before re-running cells.
- Scratchpad variables are a snapshot taken before cells run in that call; read updated values in a new call.
- Send a short `mo.status.toast` when a milestone is ready for me to look at.
