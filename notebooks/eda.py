import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    import marimo as mo
    import duckdb as ddb

    from pathlib import Path
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.io as pio

    import xgboost as xgb

    from scipy.stats import ks_2samp
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    from src.helpers import datawrangle

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
    return df_test, df_train


@app.cell
def _():
    TARGET = "satisfaction"
    ID_COL = "id"
    RANDOM_STATE = 42
    return ID_COL, RANDOM_STATE, TARGET


@app.cell
def _(df_train):
    df_in = datawrangle.normalize_columns(df_train)
    df_in
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Basic EDA - Summary Statistics
    """)
    return


@app.cell
def _(ID_COL, TARGET, df_test, df_train):
    overview = pd.DataFrame(
        {
            "Metric": [
                "Train shape",
                "Test shape",
                "Number of features",
                "Missing values",
                "Train duplicates",
                "Test duplicates",
                "Train/Test columns match",
                "Train/Test dtypes match",
            ],
            "Value": [
                df_train.shape,
                df_test.shape,
                df_train.drop(columns=[TARGET, ID_COL]).shape[1],
                df_train.isna().sum().sum(),
                df_train.duplicated().sum(),
                df_test.duplicated().sum(),
                set(df_train.columns) - {TARGET} == set(df_test.columns),
                all(
                    df_train[c].dtype == df_test[c].dtype
                    for c in df_test.columns
                ),
            ],
        }
    )
    overview
    return


@app.cell
def _(TARGET, df_train):
    # Target distribution
    target_distribution = (
        df_train[TARGET]
        .value_counts()
        .rename_axis(TARGET)
        .reset_index(name="count")
    )

    target_distribution["percentage"] = (
        target_distribution["count"] / len(df_train) * 100
    )

    target_distribution
    return


@app.cell
def _(df_train):
    # Continuous feature distributions
    continuous_features = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]

    ordinal_features = [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
    ]

    categorical_features = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col in zip(axes.flat, continuous_features):
        ax.hist(
            df_train[col].dropna(),
            bins=40,
            edgecolor="white",
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    return categorical_features, ordinal_features


@app.cell
def _(df_train, ordinal_features):
    def _():
        # Ordinal feature distributions
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        axes = axes.flat

        for ax, col in zip(axes, ordinal_features):
            counts = df_train[col].value_counts().sort_index()
            ax.bar(
                counts.index.astype(str),
                counts.values,
                color="#4C78A8",
                edgecolor="white",
            )
            ax.set_title(col, color="#333333", fontsize=10)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)

        for ax in axes[len(ordinal_features) :]:
            ax.remove()

        fig.suptitle(
            "Ordinal Feature Distributions", fontsize=14, color="#333333"
        )
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(categorical_features, df_train):
    def _():
        # Categorical Feature Distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for ax, col in zip(axes.flat, categorical_features):
            counts = df_train[col].value_counts()
            ax.bar(
                counts.index.astype(str),
                counts.values,
                color="#4C78A8",
                edgecolor="white",
            )
            ax.set_title(col, color="#333333")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)

        fig.suptitle(
            "Categorical Feature Distributions", fontsize=14, color="#333333"
        )
        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Continuous features vs target
        continuous_binned = ["Age", "Flight Distance"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, col in zip(axes, continuous_binned):
            temp = df_train[[col, TARGET]].dropna().copy()
            temp["bin"] = pd.qcut(temp[col], q=10, duplicates="drop")

            target_rate = temp.groupby("bin", observed=True)[TARGET].mean()

            ax.plot(
                range(len(target_rate)),
                target_rate.values,
                marker="o",
                color="#4C78A8",
            )
            ax.set_title(col, color="#333333")
            ax.set_xlabel("Quantile bin")
            ax.set_ylabel("Satisfaction rate")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(target_rate)))
            ax.set_xticklabels(range(1, len(target_rate) + 1))

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Delay features vs target
        delay_bins = [-1, 0, 5, 15, 30, 60, 120, np.inf]
        delay_labels = ["0", "1–5", "6–15", "16–30", "31–60", "61–120", ">120"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, col in zip(
            axes, ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
        ):
            temp = df_train[[col, TARGET]].dropna().copy()
            temp["bin"] = pd.cut(
                temp[col], bins=delay_bins, labels=delay_labels
            )

            target_rate = temp.groupby("bin", observed=True)[TARGET].mean()

            ax.plot(
                range(len(target_rate)),
                target_rate.values,
                marker="o",
                color="#4C78A8",
            )

            ax.set_title(col, color="#333333")
            ax.set_xlabel("Delay range (minutes)")
            ax.set_ylabel("Satisfaction rate")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(target_rate)))
            ax.set_xticklabels(delay_labels)

        plt.tight_layout()
        return plt.gca()

    _()
    return


@app.cell
def _(TARGET, df_train, ordinal_features):
    def _():
        # Ordinal features vs target
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        axes = axes.flat

        for ax, col in zip(axes, ordinal_features):
            target_rate = df_train.groupby(col, observed=True)[TARGET].mean()

            ax.plot(
                target_rate.index,
                target_rate.values,
                marker="o",
                # color=PLOT_COLOR
            )
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Satisfaction rate")
            ax.set_ylim(0, 1)

        for ax in axes[len(ordinal_features) :]:
            ax.remove()

        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(TARGET, categorical_features, df_train):
    def _():
        # Categorical features vs target
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for ax, col in zip(axes.flat, categorical_features):
            target_rate = df_train.groupby(col, observed=True)[TARGET].mean()

            ax.bar(
                target_rate.index.astype(str),
                target_rate.values,
                color="#4C78A8",
                edgecolor="white",
            )

            ax.set_title(col, color="#333333")
            ax.set_xlabel("")
            ax.set_ylabel("Satisfaction rate")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=0)

        fig.suptitle(
            "Categorical Features vs Satisfaction", fontsize=14, color="#333333"
        )
        plt.tight_layout()
        return plt.gca()


    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        #  Nonlinearities: continuous features
        nonlinear_features = ["Age", "Flight Distance"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, col in zip(axes, nonlinear_features):
            temp = df_train[[col, TARGET]].dropna().copy()
            temp["bin"] = pd.qcut(temp[col], q=20, duplicates="drop")

            grouped = temp.groupby("bin", observed=True)[TARGET].agg(
                satisfaction="mean",
                count="size"
            )

            ax.plot(
                range(len(grouped)),
                grouped["satisfaction"],
                marker="o",
                # color=PLOT_COLOR
            )

            ax.set_title(col)
            ax.set_xlabel("Quantile bin")
            ax.set_ylabel("Satisfaction rate")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(range(1, len(grouped) + 1))

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Identify nonlinear thresholds
        for col in ["Age", "Flight Distance"]:
            temp = df_train[[col, TARGET]].dropna().copy()
            temp["bin"] = pd.qcut(temp[col], q=20, duplicates="drop")

            grouped = (
                temp.groupby("bin", observed=True)[TARGET]
                .agg(
                    count="size",
                    satisfaction="mean"
                )
                .reset_index()
            )
        return grouped


    _()
    return


@app.cell
def _(df_train):
    # Flight Distance ↔ Class / Type of Travel
    distance_bins = [0, 500, 1000, 1200, 1500, 1800, 2500, 3500, np.inf]
    distance_labels = [
        "<500",
        "500–1000",
        "1000–1200",
        "1200–1500",
        "1500–1800",
        "1800–2500",
        "2500–3500",
        ">3500"
    ]

    temp = df_train[["Flight Distance", "Class", "Type of Travel"]].copy()
    temp["Distance range"] = pd.cut(
        temp["Flight Distance"],
        bins=distance_bins,
        labels=distance_labels
    )

    class_share = (
        pd.crosstab(
            temp["Distance range"],
            temp["Class"],
            normalize="index"
        ) * 100
    ).round(1)

    travel_share = (
        pd.crosstab(
            temp["Distance range"],
            temp["Type of Travel"],
            normalize="index"
        ) * 100
    ).round(1)

    class_share.reset_index()
    # travel_share.reset_index()
    return


@app.cell
def _(df_train):
    # Class vs Type of Travel
    class_travel = (
        pd.crosstab(
            df_train["Class"],
            df_train["Type of Travel"],
            normalize="index"
        ) * 100
    ).round(1)


    class_travel.reset_index()
    return


@app.cell
def _(df_train):
    def _():
        # Age vs categorical features
        age_bins = [0, 17, 22, 30, 38, 40, 50, 60, np.inf]
        age_labels = [
            "<18",
            "18–22",
            "23–30",
            "31–38",
            "39–40",
            "41–50",
            "51–60",
            "60+"
        ]

        temp = df_train[["Age", "Class", "Type of Travel"]].copy()
        temp["Age range"] = pd.cut(
            temp["Age"],
            bins=age_bins,
            labels=age_labels
        )

        class_share = (
            pd.crosstab(
                temp["Age range"],
                temp["Class"],
                normalize="index"
            ) * 100
        ).round(1)

        travel_share = (
            pd.crosstab(
                temp["Age range"],
                temp["Type of Travel"],
                normalize="index"
            ) * 100
        ).round(1)


        class_share.reset_index()
        return travel_share.reset_index()


    _()
    return


@app.cell
def _(df_train):
    # Departure Delay vs Arrival Delay
    delay_corr = df_train[
        ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
    ].corr().iloc[0, 1]

    delay_comparison = pd.DataFrame({
        "Metric": [
            "Pearson correlation",
            "Same delay value (%)",
            "Arrival delay > departure delay (%)",
            "Arrival delay < departure delay (%)"
        ],
        "Value": [
            delay_corr,
            (
                df_train["Departure Delay in Minutes"]
                == df_train["Arrival Delay in Minutes"]
            ).mean() * 100,
            (
                df_train["Arrival Delay in Minutes"]
                > df_train["Departure Delay in Minutes"]
            ).mean() * 100,
            (
                df_train["Arrival Delay in Minutes"]
                < df_train["Departure Delay in Minutes"]
            ).mean() * 100
        ]
    })

    delay_comparison["Value"] = delay_comparison["Value"].round(2)


    delay_comparison
    return


@app.cell
def _(TARGET, df_train):
    # Interaction: Class × Type of Travel
    interaction = (
        df_train.groupby(["Class", "Type of Travel"], observed=True)[TARGET]
        .agg(
            count="size",
            satisfaction="mean"
        )
        .reset_index()
    )

    interaction
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Interaction: Age × Class
        age_bins = [0, 17, 22, 30, 38, 40, 50, 60, np.inf]
        age_labels = ["<18", "18–22", "23–30", "31–38", "39–40", "41–50", "51–60", "60+"]

        temp = df_train[["Age", "Class", TARGET]].copy()
        temp["Age range"] = pd.cut(
            temp["Age"],
            bins=age_bins,
            labels=age_labels
        )

        interaction = (
            temp.groupby(["Age range", "Class"], observed=True)[TARGET]
            .agg(
                count="size",
                satisfaction="mean"
            )
            .reset_index()
        )
        return interaction


    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Interaction: Flight Distance × Class
        distance_bins = [0, 500, 1000, 1200, 1500, 1800, 2500, 3500, np.inf]
        distance_labels = [
            "<500", "500–1000", "1000–1200", "1200–1500",
            "1500–1800", "1800–2500", "2500–3500", ">3500"
        ]

        temp = df_train[["Flight Distance", "Class", TARGET]].copy()
        temp["Distance range"] = pd.cut(
            temp["Flight Distance"],
            bins=distance_bins,
            labels=distance_labels
        )

        interaction = (
            temp.groupby(["Distance range", "Class"], observed=True)[TARGET]
            .agg(
                count="size",
                satisfaction="mean"
            )
            .reset_index()
        )
        return interaction


    _()
    return


@app.cell
def _(TARGET, df_train):
    def _():
        # Interaction: Arrival Delay × Class
        delay_bins = [-np.inf, 0, 5, 15, 30, 60, 120, np.inf]
        delay_labels = [
            "0", "1–5", "6–15", "16–30",
            "31–60", "61–120", ">120"
        ]

        temp = df_train[["Arrival Delay in Minutes", "Class", TARGET]].dropna().copy()
        temp["Delay range"] = pd.cut(
            temp["Arrival Delay in Minutes"],
            bins=delay_bins,
            labels=delay_labels
        )

        interaction = (
            temp.groupby(["Delay range", "Class"], observed=True)[TARGET]
            .agg(
                count="size",
                satisfaction="mean"
            )
            .reset_index()
        )
        return interaction


    _()
    return


@app.cell
def _(df_test, df_train):
    def add_features(train, test):
        train = train.copy()
        test = test.copy()

        # Missing arrival delay
        train["Arrival_Delay_Missing"] = train["Arrival Delay in Minutes"].isna().astype("int8")
        test["Arrival_Delay_Missing"] = test["Arrival Delay in Minutes"].isna().astype("int8")

        # Delay difference
        train["Delay_Difference"] = train["Arrival Delay in Minutes"] - train["Departure Delay in Minutes"]
        test["Delay_Difference"] = test["Arrival Delay in Minutes"] - test["Departure Delay in Minutes"]

        # Flight distance × class
        class_map = {"Eco": 0, "Eco Plus": 1, "Business": 2}

        train["Distance_Class"] = train["Flight Distance"] * train["Class"].map(class_map)
        test["Distance_Class"] = test["Flight Distance"] * test["Class"].map(class_map)

        # Digital decomposition of flight distance
        train["Distance_Hundreds"] = (train["Flight Distance"] % 1000) // 100
        train["Distance_Last2"] = train["Flight Distance"] % 100

        test["Distance_Hundreds"] = (test["Flight Distance"] % 1000) // 100
        test["Distance_Last2"] = test["Flight Distance"] % 100

        # Class × Type of Travel
        train["Class_TypeTravel"] = train["Class"].astype(str) + "_" + train["Type of Travel"].astype(str)
        test["Class_TypeTravel"] = test["Class"].astype(str) + "_" + test["Type of Travel"].astype(str)

        # Class × Customer Type
        train["Class_CustomerType"] = train["Class"].astype(str) + "_" + train["Customer Type"].astype(str)
        test["Class_CustomerType"] = test["Class"].astype(str) + "_" + test["Customer Type"].astype(str)

        # Class × Type of Travel × Customer Type
        train["Class_TypeTravel_CustomerType"] = train["Class"].astype(str) + "_" + train["Type of Travel"].astype(str) + "_" + train["Customer Type"].astype(str)
        test["Class_TypeTravel_CustomerType"] = test["Class"].astype(str) + "_" + test["Type of Travel"].astype(str) + "_" + test["Customer Type"].astype(str)

        # Class × Wi-Fi
        train["Class_Wifi"] = train["Class"].astype(str) + "_" + train["Inflight wifi service"].astype(str)
        test["Class_Wifi"] = test["Class"].astype(str) + "_" + test["Inflight wifi service"].astype(str)

        # Distance transformations
        train["Distance_Log"] = np.log1p(train["Flight Distance"])
        test["Distance_Log"] = np.log1p(test["Flight Distance"])

        train["Distance_Sqrt"] = np.sqrt(train["Flight Distance"])
        test["Distance_Sqrt"] = np.sqrt(test["Flight Distance"])

    
        # Distance bins
        distance_bins = [-np.inf, 500, 1000, 1500, 2000, 3000, 5000, np.inf]
        distance_labels = [
            "0_500",
            "501_1000",
            "1001_1500",
            "1501_2000",
            "2001_3000",
            "3001_5000",
            "5001_plus"
        ]

        train["Distance_Bin"] = pd.cut(
            train["Flight Distance"], bins=distance_bins, labels=distance_labels
        ).astype(str)

        test["Distance_Bin"] = pd.cut(
            test["Flight Distance"], bins=distance_bins, labels=distance_labels
        ).astype(str)

        # Structural decomposition by 1000 km
        train["Distance_1000s"] = train["Flight Distance"] // 1000
        test["Distance_1000s"] = test["Flight Distance"] // 1000

        train["Distance_Mod_1000"] = train["Flight Distance"] % 1000
        test["Distance_Mod_1000"] = test["Flight Distance"] % 1000

        # Distance modulo 250
        train["Distance_Mod_250"] = train["Flight Distance"] % 250
        test["Distance_Mod_250"] = test["Flight Distance"] % 250

        # Binary flags + interactions
        ordinal_features = [
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Cleanliness"
        ]
    
        for col in ordinal_features:
            # 1. Create the flag
            flag_name = col.replace(" ", "_").replace("/", "_") + "_is_zero"
            train[flag_name] = (train[col] == 0).astype("int8")
            test[flag_name]  = (test[col] == 0).astype("int8")
    
            # 2. Interaction with Class
            train[f"{flag_name}_x_Class"] = train[flag_name].astype(str) + "_" + train["Class"].astype(str)
            test[f"{flag_name}_x_Class"]  = test[flag_name].astype(str) + "_" + test["Class"].astype(str)
    
            # 3. Interaction with Type of Travel
            train[f"{flag_name}_x_Travel"] = train[flag_name].astype(str) + "_" + train["Type of Travel"].astype(str)
            test[f"{flag_name}_x_Travel"]  = test[flag_name].astype(str) + "_" + test["Type of Travel"].astype(str)

        return train, test


    train, test = add_features(df_train, df_test)
    return test, train


@app.cell
def _(ID_COL, TARGET, test, train):
    y = train[TARGET].astype(int)

    X = train.drop(columns=[TARGET, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    print(f"Features:    {X.shape[1]}")
    print(f"Categorical: {len(categorical_cols)}")
    print(f"Numerical:   {len(numeric_cols)}")
    return X, X_test, categorical_cols, numeric_cols, y


@app.cell
def _(RANDOM_STATE, X, categorical_cols, numeric_cols, y):
    N_SPLITS = 5
    N_ESTIMATORS_CV = 3000
    LEARNING_RATE = 0.01783780580019371
    EARLY_STOPPING_ROUNDS = 180

    MAX_DEPTH = 9
    MIN_CHILD_WEIGHT = 7
    SUBSAMPLE = 0.8593270986225269
    COLSAMPLE_BYTREE = 0.6616735383572131
    GAMMA = 0.6136611655414813
    REG_ALPHA = 0.009239335165807
    REG_LAMBDA = 3.6009252105250233

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof = np.zeros(len(X))

    fold_scores = []
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y), 1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True
                    ),
                    categorical_cols
                ),
                (
                    "num",
                    "passthrough",
                    numeric_cols
                )
            ]
        )

        X_train_enc = preprocessor.fit_transform(X_train)
        X_valid_enc = preprocessor.transform(X_valid)

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS_CV,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_CHILD_WEIGHT,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            gamma=GAMMA,
            reg_alpha=REG_ALPHA,
            reg_lambda=REG_LAMBDA,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_valid_enc, y_valid)],
            verbose=False
        )

        valid_pred = model.predict_proba(X_valid_enc)[:, 1]

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        oof[valid_idx] = valid_pred
        fold_scores.append(fold_auc)
        best_iterations.append(model.best_iteration)

        print(
            f"Fold {fold}: "
            f"AUC={fold_auc:.6f}, "
            f"best_iter={model.best_iteration}"
        )

    return (
        COLSAMPLE_BYTREE,
        GAMMA,
        LEARNING_RATE,
        MAX_DEPTH,
        MIN_CHILD_WEIGHT,
        REG_ALPHA,
        REG_LAMBDA,
        SUBSAMPLE,
        best_iterations,
        fold_scores,
        oof,
    )


@app.cell
def _(best_iterations, fold_scores, oof, y):
    # Cross-validation results

    oof_auc = roc_auc_score(y, oof)
    mean_fold_auc = np.mean(fold_scores)
    std_fold_auc = np.std(fold_scores)
    mean_best_iteration = np.mean(best_iterations)

    print("Cross-validation results") 
    print(f"OOF ROC-AUC:{oof_auc:.6f}")
    return


@app.cell
def _(
    COLSAMPLE_BYTREE,
    GAMMA,
    LEARNING_RATE,
    MAX_DEPTH,
    MIN_CHILD_WEIGHT,
    RANDOM_STATE,
    REG_ALPHA,
    REG_LAMBDA,
    SUBSAMPLE,
    X,
    X_test,
    categorical_cols,
    numeric_cols,
    y,
):
    FINAL_N_ESTIMATORS = 1456

    final_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_cols
            ),
            (
                "num",
                "passthrough",
                numeric_cols
            )
        ]
    )

    X_enc = final_preprocessor.fit_transform(X)
    X_test_enc = final_preprocessor.transform(X_test)

    final_model = xgb.XGBClassifier(
        n_estimators=FINAL_N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        gamma=GAMMA,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    final_model.fit(
        X_enc,
        y,
        verbose=False
    )
    return X_test_enc, final_model


@app.cell
def _(ID_COL, TARGET, X_test_enc, final_model, test):
    # Generate submission
    test_pred = final_model.predict_proba(X_test_enc)[:, 1]

    submission = pd.DataFrame({
        "id": test[ID_COL],
        TARGET: test_pred
    })
    SUBMISSION_PATH = Path("data/interim/submission.csv")
    SUBMISSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    submission.to_csv(
        SUBMISSION_PATH,
        index=False
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
