import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import marimo as mo
    import duckdb as ddb
    from pathlib import Path

    con = ddb.connect()

    KAGGLEDATAFOLDER = "./data/raw/playground-series-s6e10/"


@app.cell
def _():
    # load data
    df_train = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{KAGGLEDATAFOLDER}train.csv')
    """).fetchdf()

    con.register("train", df_train)
    return


@app.cell
def _():
    con.execute("""
        SELECT *
        FROM train
        LIMIT 10
    """).fetchdf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
