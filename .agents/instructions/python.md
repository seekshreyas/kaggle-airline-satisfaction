# Python conventions

- Use type annotations for public functions.
- Separate computation from filesystem, network, and database access. Pure functions return DataFrames or figures; I/O sits at the edges (loaders, CLI commands, `save_*` functions).
- Prefer explicit inputs over hidden global state. Pass configs in as dicts; don't read YAML inside computation.
- Use pathlib for filesystem paths. Resolve paths against the project root (`src.utils.paths.project_root()`), not the working directory.
- Preserve reproducibility. Expose seeds where relevant, but use the configurable salted hash (`src.datahelpers.splits`) for data splits and folds.
- Follow the project's configured formatter, linter, and test commands.
- I prefer my scripts to have dual purpose - I could pull them in notebooks or other scripts. But also I want to be able to invoke them on command line with arguments for short succinct tasks.

## Module conventions
- Start every module with the shared header: shebang, `# -*- coding: utf-8 -*-`, a docstring (what it does plus a CLI usage example), `__author__` and `__email__`.
- CLIs use `typer` with subcommands (`python -m src.<pkg>.<module> <command> ...`). Add an `@app.callback()` so a module with a single command still uses subcommand style.
- Things that should be selectable from YAML go in a **registry** with a decorator (e.g. `@register_derive("zero_flag")`, `@register_model("logreg")`), so a new option is one function plus one config line.
- Name things for readability: derived feature columns carry a kind prefix (`flag__`, `num__`, `cat__`); model output feature names stay human-readable end to end.
- Keep layers separate: `modeling/diagnostics.py` **computes** tables, `src/viz/` **draws** them, notebooks and `src/reporting/` **compose** them.

## Library preferences
I prefer
- `Duckdb` > `Polars` > `Pandas` for dataframe operations. DuckDB is also the parquet reader/writer here (pandas' pyarrow is unreliable in this environment).
- `Plotly` > `yellowbrick` > `Seaborn` > `Matplotlib` for data visualization and plots
- `marimo` > `jupyter` for notebook
- `Scikit-Learn` as preferred machine learning library. Fit anything that learns from data inside the CV fold (pipelines), and check the installed version's API: e.g. in sklearn ≥ 1.8 `LogisticRegression(penalty=...)` is deprecated; use `l1_ratio` and `C`.
- `typer` for command line interface in scripts
- `pyyaml` for configs
