# Config conventions (YAML)

- `configs/competition.yaml` holds everything competition-specific: data paths, id/target, metric, column naming, column schema (roles after renaming), display names, randomization salt, splits, CV folds and the keep rule. It's the only file to rewrite for a new competition.
- One YAML per experiment in `configs/experiments/`, which `extends` its parent and states **only what changed**. Merging: dicts merge, lists replace, the child wins.
- Key repeatable sections **by name** (`derive` blocks, `encode` groups) so a child can override one entry without restating the rest.
- Refer to column groups with `"@group"` references to the schema instead of repeating column lists.
- Every experiment has `name` (= file stem), a technical `description`, and a plain-language `label` used in executive reports.
- Comment the *why* inline (e.g. why a rung extends an earlier winner, why an option was dropped). Use `NOTE:` for placeholders that must be updated after a decision.
- Report settings live in `configs/reports/*.yaml` (title, runs compared, Deckset commands, image size).
- I may reformat these files (4-space indents, reordered keys): re-read before editing and keep my formatting.
- Keep `tests/test_config_files.py` passing: every file parses, every `extends` target exists, the schema matches the real headers, and every `@ref` resolves.
