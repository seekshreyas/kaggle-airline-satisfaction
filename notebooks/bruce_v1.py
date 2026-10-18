import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    import pandas as pd
    import numpy as np
    import logging
    import marimo as mo
    import duckdb as ddb
    import icecream as ic

    from pathlib import Path

    # plotting
    import plotly.io as pio
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # eda
    import sweetviz as sv
    from skrub import TableReport

    from dotenv import load_dotenv

    load_dotenv()
    con = ddb.connect()

    def _get_required_env(var_name: str) -> str:
        value = os.getenv(var_name)
        if value is None or value == "":
            raise ValueError(
                f"Missing required environment variable: {var_name}"
            )
        return value

    # preferences
    pio.get_chrome()
    pd.options.plotting.backend = "plotly"

    # project defaults
    DIR_PROJECT = _get_required_env("DIR_PROJECT").rstrip("/\\")
    DIR_KAGGLE_REL = _get_required_env("DIR_KAGGLE_REL").lstrip("/\\")
    DIR_KAGGLE = os.path.join(DIR_PROJECT, DIR_KAGGLE_REL)

    print(f"Working directory: {DIR_PROJECT}")
    print(f"Kaggle directory: {DIR_KAGGLE}")

    sys.path.insert(1, DIR_PROJECT)
    sv.config_parser.read(DIR_PROJECT + "/configs/sweetviz_config.ini")

    # Notebook status
    NOTEBOOKSTATUS = "dev"


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Project Details
    ===============

    This is my notebook for the [kaggle competition]


    [kaggle competition]: https://www.kaggle.com/competitions/playground-series-s6e10/overview
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
