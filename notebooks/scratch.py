import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    import marimo as mo
    import duckdb as ddb
    from pathlib import Path
    import plotly.express as px
    import plotly.io as pio

    con = ddb.connect()

    # project defaults
    KAGGLEDATAFOLDER = "./data/raw/playground-series-s6e10/"

    # preferences
    pd.options.plotting.backend = "plotly"


@app.cell
def _():
    # load data
    df_train = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{KAGGLEDATAFOLDER}train.csv')
    """).fetchdf()

    df_test = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{KAGGLEDATAFOLDER}test.csv')
    """).fetchdf()

    df_submission = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{KAGGLEDATAFOLDER}sample_submission.csv')
    """).fetchdf()

    con.register("df_train", df_train)
    con.register("df_test", df_test)
    con.register("df_submission", df_submission)
    return (df_train,)


@app.cell
def _():
    # con.execute("""
    #     SELECT *
    #     FROM df_train
    #     LIMIT 10
    # """).fetchdf()
    return


@app.function
def normalize_columns(df):
    """
    Normalize dataframe column names by:
    - Converting to lowercase
    - Replacing whitespace and special characters with underscores

    Args:
        df: Input dataframe (pandas or polars)

    Returns:
        Dataframe with normalized column names
    """
    # Create a mapping of old column names to normalized names
    normalized_names = {}
    for col in df.columns:
        # Convert to lowercase and replace whitespace with underscore
        normalized = col.lower().replace(" ", "_")
        normalized_names[col] = normalized

    # Rename columns based on the mapping
    return df.rename(columns=normalized_names)


@app.cell
def _(df_train):
    df_in = normalize_columns(df_train)

    print(list(df_in.columns))
    df_in
    return (df_in,)


@app.cell
def _(df_in):
    # inspecting target columns
    px.histogram(df_in, x="satisfaction").show()
    return


@app.cell
def _(df_in):
    # inspecting numerical columns
    for numcol in df_in.loc[:, "age":"arrival_delay_in_minutes"].columns:
        px.histogram(
            df_in,
            x=numcol,
            color="satisfaction",
            # marginal="box",
            title=f"{numcol} distribution over satisfaction",
        ).show()
        # break
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
