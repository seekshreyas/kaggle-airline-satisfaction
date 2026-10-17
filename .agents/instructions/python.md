# Python conventions

- Use type annotations for public functions.
- Separate computation from filesystem, network, and database access.
- Prefer explicit inputs over hidden global state.
- Use pathlib for filesystem paths.
- Preserve reproducibility; expose random seeds where relevant.
- Follow the project's configured formatter, linter, and test commands.
- I prefer my scripts to have dual purpose - I could pull them in notebooks or other scripts. But also I want to be able to invoke them on command line with arguments for short succinct tasks. 


## Library preferences
I prefer
- `Duckdb` > `Polars` > `Pandas` for dataframe operations.
- `Plotly` > `yellowbrick` > `Seaborn` > `Matplotlib` for data visualization and plots
- `marimo` > `jupyter` for notebook 
- `Scikit-Learn` as preferred machine learning library. 
- `typer` for command line interface in scripts
