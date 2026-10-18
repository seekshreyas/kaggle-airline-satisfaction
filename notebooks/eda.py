# /// script
# dependencies = [
#     "duckdb==1.5.6",
#     "icecream==2.2.0",
#     "marimo>=0.25.1",
#     "numpy==2.5.4",
#     "pandas==3.0.6",
#     "plotly==7.1.0",
#     "python-dotenv==1.2.4",
#     "skrub==0.11.0",
# ]
# ///

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
            raise ValueError(f"Missing required environment variable: {var_name}")
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
    sv.config_parser.read(DIR_PROJECT + '/configs/sweetviz_config.ini')

    # Notebook status
    NOTEBOOKSTATUS = "dev"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Project Details
    ===============

    This is my notebook for the [kaggle competition]


    [kaggle competition]: https://www.kaggle.com/competitions/playground-series-s6e10/overview
    """)
    return


@app.cell
def _():
    ## 1. Load Kaggle Dataset
    return


@app.cell
def _():
    from src.datahelpers import datawrangle

    return (datawrangle,)


@app.cell
def _(datawrangle):
    # load data
    df_raw_train = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{DIR_KAGGLE}train.csv')
    """).fetchdf()

    df_raw_test = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{DIR_KAGGLE}test.csv')
    """).fetchdf()

    df_raw_submission = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{DIR_KAGGLE}sample_submission.csv')
    """).fetchdf()

    df_train = datawrangle.normalize_columns(df_raw_train)
    df_test = datawrangle.normalize_columns(df_raw_test)
    df_sub = datawrangle.normalize_columns(df_raw_submission)

    con.register("df_train", df_train)
    con.register("df_test", df_test)
    con.register("df_submission", df_sub)

    TARGET = "satisfaction"
    ID_COL = "id"
    RANDOM_STATE = 42
    return ID_COL, TARGET, df_train


@app.cell(hide_code=True)
def _(df_train):
    df_train
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Inspect Dataset

    ### a. Automatic EDA
    """)
    return


@app.cell
def _(ID_COL, TARGET, df_train):
    sv_feature_config = sv.FeatureConfig(skip=ID_COL)

    # analyze: training columns vs TARGET
    # sv.analyze([df_train,'Train'],
    #            target_feat=TARGET,
    #            feat_cfg=sv_feature_config)\
    #     .show_html(DIR_PROJECT + "/reports/eda/report_train_vs_target.html")

    # # compare: train vs test
    # sv.compare([df_train, "Train"],
    #            [df_test, "Test"],
    #            target_feat=TARGET,
    #            feat_cfg=sv_feature_config)\
    #     .show_html(DIR_PROJECT + "/reports/eda/comparison_report_train_vs_test.html")

    # compare intra
    sv.compare_intra(df_train, 
                     df_train["class"] == 'Eco', ["Economy", "Non-Economy"], 
                    target_feat=TARGET,feat_cfg=sv_feature_config)\
        .show_html(DIR_PROJECT + "/reports/eda/intro_report_train_with_ecoclass.html")
    return


@app.cell
def _(df_train):
    TableReport(df_train)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
