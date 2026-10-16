"""5115056029 HW2 - Multiple Linear Regression.

本主程式依照 CRISP-DM 六階段完成學生生產力分數回歸分析：

1. Business Understanding：定義預測目標與評估指標。
2. Data Understanding：檢查資料結構、缺失值、描述統計與相關性。
3. Data Preparation：排除識別欄、補值、標準化與 One-Hot Encoding。
4. Modeling：比較 Dummy、Linear Regression、Ridge、Lasso，並執行
   SelectKBest Feature Selection。
5. Evaluation：使用 MAE、RMSE、R²、5-fold CV、殘差與 95% 預測區間。
6. Deployment：輸出完整 sklearn Pipeline、Schema 與分析摘要。

預設資料檔為 StudentsPerformance_train.csv，目標欄位為
``productivity_score``。``id`` 與 ``student_id`` 只用於識別資料列，
不會進入模型。

執行方式：
    python 5115056029_hw2.py
    python 5115056029_hw2.py --data-file StudentsPerformance_train.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "productivity_score"
DATA_SOURCE_URL = (
    "https://www.kaggle.com/competitions/"
    "student-performance-prediction-machine-learning-challenge/"
    "data?select=train.csv"
)
ID_COLUMNS = ["id", "student_id"]
RAW_FEATURES = [
    "age",
    "gender",
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
    "breaks_per_day",
    "coffee_intake_mg",
    "exercise_minutes",
    "assignments_completed",
    "attendance_percentage",
    "stress_level",
    "focus_score",
    "final_grade",
]
N_SELECTED_FEATURES = 12
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.25
RANDOM_STATE = 42

# Shared visual language for the analysis figures and the web dashboard.
PLOT_BG = "#071824"
PLOT_PANEL = "#0B2230"
PLOT_TEXT = "#EAF7F7"
PLOT_MUTED = "#91AAB5"
PLOT_GRID = "#294755"
PLOT_CYAN = "#54E7D4"
PLOT_BLUE = "#57A8FF"
PLOT_LIME = "#C4F36B"
PLOT_RED = "#FF7184"
PLOT_ORANGE = "#FFAE69"


def configure_plot_theme() -> None:
    """Keep exported research figures visually consistent with the website."""
    plt.rcParams.update(
        {
            "figure.facecolor": PLOT_BG,
            "savefig.facecolor": PLOT_BG,
            "axes.facecolor": PLOT_BG,
            "axes.edgecolor": PLOT_GRID,
            "axes.labelcolor": PLOT_TEXT,
            "axes.titlecolor": PLOT_TEXT,
            "axes.titleweight": "bold",
            "text.color": PLOT_TEXT,
            "xtick.color": PLOT_MUTED,
            "ytick.color": PLOT_MUTED,
            "grid.color": PLOT_GRID,
            "grid.alpha": 0.34,
            "legend.facecolor": PLOT_PANEL,
            "legend.edgecolor": PLOT_GRID,
            "legend.labelcolor": PLOT_TEXT,
            "font.size": 11,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW2 multiple linear regression for student productivity scores"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("StudentsPerformance_train.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots in addition to saving PNG files.",
    )
    return parser.parse_args()


def print_title(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Model": model_name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    print_title(f"Model evaluation: {model_name}")
    print(
        f"MAE: {result['MAE']:.4f}  "
        f"RMSE: {result['RMSE']:.4f}  "
        f"R2: {result['R2']:.4f}"
    )
    return result


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def make_model(
    numeric_features: list[str],
    categorical_features: list[str],
    estimator: Any,
    select_k: int | None = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("preprocessor", make_preprocessor(numeric_features, categorical_features))
    ]
    if select_k is not None:
        steps.append(("feature_selection", SelectKBest(f_regression, k=select_k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def save_figure(output_dir: Path, filename: str, show_plots: bool) -> None:
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved figure: {path}")


def conformal_quantile(errors: np.ndarray, alpha: float = 0.05) -> float:
    """Return the finite-sample split-conformal error quantile."""
    n = len(errors)
    probability = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(errors, probability, method="higher"))


def cross_validation_summary(
    models: list[tuple[str, Pipeline]], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    rows: list[dict[str, Any]] = []
    for name, model in models:
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
        rows.append(
            {
                "Model": name,
                "CV_MAE_Mean": -scores["test_MAE"].mean(),
                "CV_MAE_SD": scores["test_MAE"].std(ddof=1),
                "CV_RMSE_Mean": -scores["test_RMSE"].mean(),
                "CV_RMSE_SD": scores["test_RMSE"].std(ddof=1),
                "CV_R2_Mean": scores["test_R2"].mean(),
                "CV_R2_SD": scores["test_R2"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows).sort_values("CV_RMSE_Mean").reset_index(drop=True)


def run() -> None:
    """依序執行 CRISP-DM 六階段，並將所有成果寫入 outputs。"""
    args = parse_args()
    configure_plot_theme()
    if not args.data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_file.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # CRISP-DM 1 - Business Understanding（商業／問題理解）
    # 定義任務：以學生行為、生活與學業特徵預測 productivity_score。
    # 主要評估指標使用 RMSE，並以 DummyRegressor 作為平均值基準。
    # ======================================================================
    print_title("CRISP-DM 1: Business Understanding")
    print(
        "Goal: predict the official productivity_score target using behavioral, "
        "lifestyle, attendance, and academic features. RMSE is the primary "
        "comparison metric.\nThis analysis uses the supplied competition train CSV.\n"
        f"Data source: {DATA_SOURCE_URL}"
    )

    # ======================================================================
    # CRISP-DM 2 - Data Understanding（資料理解）
    # 讀取 CSV、確認必要欄位、資料筆數、型別、缺失與重複情況；
    # 再輸出描述統計、Target 分布、Pearson correlation 與完整相關矩陣。
    # ======================================================================
    print_title("CRISP-DM 2: Data Understanding")
    source_df = pd.read_csv(args.data_file)
    required_columns = ID_COLUMNS + RAW_FEATURES + [TARGET]
    missing_columns = [c for c in required_columns if c not in source_df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    df = source_df[required_columns].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError(f"Target contains {df[TARGET].isna().sum()} invalid values")
    if df.empty:
        raise ValueError("CSV contains no records")
    if (df["student_id"] != df["id"] + 1).any():
        print("Warning: student_id is not always id + 1; both IDs remain excluded.")

    X = df[RAW_FEATURES].copy()
    y = df[TARGET].copy()
    print(f"Dataset shape: {df.shape}")
    print(f"Raw predictor count: {len(RAW_FEATURES)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Excluded identifiers: {ID_COLUMNS}")
    print(df.head())

    missing_table = pd.DataFrame(
        {
            "Missing_Count": df.isna().sum(),
            "Missing_Rate_Percent": [
                df[column].isna().sum() / len(df) * 100 for column in df.columns
            ],
        },
        index=df.columns,
    ).sort_values("Missing_Rate_Percent", ascending=False)
    missing_table.to_csv(
        args.output_dir / "missing_values.csv", encoding="utf-8-sig"
    )
    df.describe(include="all").T.to_csv(
        args.output_dir / "descriptive_statistics.csv", encoding="utf-8-sig"
    )

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=24, color=PLOT_CYAN, alpha=0.82, edgecolor=PLOT_BG)
    plt.axvline(
        y.mean(), color=PLOT_ORANGE, linestyle="--", linewidth=2,
        label=f"Mean = {y.mean():.2f}",
    )
    plt.axvline(
        y.median(), color=PLOT_BLUE, linestyle=":", linewidth=2.5,
        label=f"Median = {y.median():.2f}",
    )
    plt.title("Distribution of Productivity Scores")
    plt.xlabel("Productivity Score")
    plt.ylabel("Count")
    plt.legend()
    save_figure(
        args.output_dir, "01_productivity_score_distribution.png", args.show_plots
    )

    numeric_raw_features = X.select_dtypes(include=np.number).columns.tolist()
    correlations = (
        df[numeric_raw_features + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    correlations.rename("Correlation").to_csv(
        args.output_dir / "feature_correlations.csv", encoding="utf-8-sig"
    )
    correlation_plot = correlations.sort_values()
    plt.figure(figsize=(10, 7))
    colors = np.where(correlation_plot >= 0, PLOT_CYAN, PLOT_RED)
    plt.barh(correlation_plot.index, correlation_plot.values, color=colors)
    plt.axvline(0, color=PLOT_MUTED, linewidth=0.8)
    plt.title("Numeric Feature Correlations with Productivity Score")
    plt.xlabel("Pearson Correlation")
    save_figure(args.output_dir, "02_feature_correlations.png", args.show_plots)

    # A complete Pearson matrix complements the target-only ranking above. IDs are
    # intentionally excluded because their numeric appearance does not make them
    # meaningful explanatory variables.
    # Put the target first so its entire row/column can be read immediately, as in
    # the research reference heatmap.
    correlation_columns = [TARGET] + numeric_raw_features
    correlation_matrix = df[correlation_columns].corr(method="pearson")
    correlation_matrix.to_csv(
        args.output_dir / "pearson_correlation_matrix.csv", encoding="utf-8-sig"
    )
    dark_background = PLOT_BG
    with plt.rc_context(
        {
            "figure.facecolor": dark_background,
            "axes.facecolor": dark_background,
            "axes.edgecolor": "#6f8796",
            "axes.labelcolor": PLOT_TEXT,
            "xtick.color": PLOT_TEXT,
            "ytick.color": PLOT_TEXT,
            "text.color": PLOT_TEXT,
        }
    ):
        figure, axis = plt.subplots(figsize=(14, 11))
        tech_diverging = LinearSegmentedColormap.from_list(
            "tech_diverging",
            ["#2676B8", "#173E55", "#102733", "#D26B55", "#A90D3F"],
        )
        image = axis.imshow(
            correlation_matrix, cmap=tech_diverging, vmin=-1, vmax=1, aspect="equal"
        )
        axis.set_xticks(range(len(correlation_columns)))
        axis.set_yticks(range(len(correlation_columns)))
        axis.set_xticklabels(
            correlation_columns, rotation=48, ha="right", fontsize=9
        )
        axis.set_yticklabels(correlation_columns, fontsize=9)
        axis.set_title(
            "Pearson Correlation Matrix · Target: productivity_score",
            loc="left",
            pad=18,
            fontweight="bold",
        )
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_color("#334d5c")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Pearson r", rotation=270, labelpad=18)
        colorbar.ax.tick_params(colors=PLOT_TEXT)
        colorbar.outline.set_edgecolor("#6f8796")
    save_figure(
        args.output_dir, "02a_pearson_correlation_heatmap.png", args.show_plots
    )

    # ======================================================================
    # CRISP-DM 3 - Data Preparation（資料準備）
    # X 排除 id、student_id 與 Target；y 使用 productivity_score。
    # 數值欄：中位數補值 + StandardScaler。
    # 類別欄：眾數補值 + OneHotEncoder(drop="first")。
    # 先切 train/validation，再只用 training data fit，避免資料洩漏。
    # ======================================================================
    print_title("CRISP-DM 3: Data Preparation")
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Train shape: {X_train.shape}; validation shape: {X_valid.shape}")

    probe = make_preprocessor(numeric_features, categorical_features)
    transformed = probe.fit_transform(X_train)
    transformed_feature_count = transformed.shape[1]
    select_k = min(N_SELECTED_FEATURES, transformed_feature_count)
    print(f"One-hot transformed feature count: {transformed_feature_count}")
    print(f"SelectKBest k: {select_k}")

    # ======================================================================
    # CRISP-DM 4 - Modeling（建模與特徵選擇）
    # 使用 SelectKBest(f_regression, k=12) 執行 Feature Selection，並比較：
    # Dummy Baseline、完整多元線性回歸、Selected Linear、Ridge、Lasso。
    # 所有前處理、選特徵與模型都封裝在同一個 sklearn Pipeline。
    # ======================================================================
    print_title("CRISP-DM 4: Modeling + Feature Selection")
    models: list[tuple[str, Pipeline]] = [
        (
            "Dummy Baseline",
            make_model(
                numeric_features,
                categorical_features,
                DummyRegressor(strategy="mean"),
            ),
        ),
        (
            "Full Multiple Linear Regression",
            make_model(
                numeric_features, categorical_features, LinearRegression()
            ),
        ),
        (
            f"Selected Linear Regression ({select_k} Features)",
            make_model(
                numeric_features,
                categorical_features,
                LinearRegression(),
                select_k,
            ),
        ),
        (
            f"Selected Ridge ({select_k} Features)",
            make_model(
                numeric_features,
                categorical_features,
                Ridge(alpha=1.0),
                select_k,
            ),
        ),
        (
            f"Selected Lasso ({select_k} Features)",
            make_model(
                numeric_features,
                categorical_features,
                Lasso(alpha=0.05, max_iter=30000),
                select_k,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models:
        model.fit(X_train, y_train)
        fitted_models[name] = model
        predictions[name] = model.predict(X_valid)
        results.append(evaluate_regression(y_valid, predictions[name], name))

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(
        args.output_dir / "holdout_model_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_results = cross_validation_summary(models, X, y)
    cv_results.to_csv(
        args.output_dir / "cross_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print_title("5-fold Cross-validation")
    print(cv_results.to_string(index=False))

    selected_name = f"Selected Linear Regression ({select_k} Features)"
    selected_pipeline = fitted_models[selected_name]
    selector = selected_pipeline.named_steps["feature_selection"]
    feature_names = selected_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    selected_mask = selector.get_support()
    feature_scores = pd.DataFrame(
        {
            "Feature": feature_names,
            "F_Score": selector.scores_,
            "P_Value": selector.pvalues_,
            "Selected": selected_mask,
        }
    ).sort_values("F_Score", ascending=False, na_position="last")
    feature_scores.to_csv(
        args.output_dir / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficient_df = pd.DataFrame(
        {
            "Feature": feature_names[selected_mask],
            "Coefficient": selected_pipeline.named_steps["model"].coef_,
        }
    )
    coefficient_df["Absolute_Coefficient"] = coefficient_df["Coefficient"].abs()
    coefficient_df = coefficient_df.sort_values(
        "Absolute_Coefficient", ascending=False
    )
    coefficient_df.to_csv(
        args.output_dir / "regression_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    score_plot = feature_scores.sort_values("F_Score")
    plt.figure(figsize=(11, 8))
    score_colors = np.where(score_plot["Selected"], PLOT_CYAN, "#425C68")
    plt.barh(score_plot["Feature"], score_plot["F_Score"], color=score_colors)
    plt.title("Feature Selection: F-regression Scores")
    plt.xlabel("F Score")
    save_figure(args.output_dir, "03_feature_selection_scores.png", args.show_plots)

    coefficient_plot = coefficient_df.sort_values("Coefficient")
    coefficient_colors = np.where(
        coefficient_plot["Coefficient"] >= 0, PLOT_CYAN, PLOT_RED
    )
    plt.figure(figsize=(11, 8))
    plt.barh(
        coefficient_plot["Feature"],
        coefficient_plot["Coefficient"],
        color=coefficient_colors,
    )
    plt.axvline(0, color=PLOT_MUTED, linewidth=0.8)
    plt.title("Selected Linear Regression Coefficients")
    plt.xlabel("Coefficient")
    save_figure(args.output_dir, "04_regression_coefficients.png", args.show_plots)

    # ======================================================================
    # CRISP-DM 5 - Evaluation（模型評估）
    # 使用未參與訓練的 validation set 計算 MAE、RMSE、R²；
    # 搭配 5-fold Cross-Validation、Actual vs Predicted、Residual Plot，
    # 並以 split conformal calibration 建立 95% prediction interval。
    # ======================================================================
    print_title("CRISP-DM 5: Evaluation")
    selected_pred = predictions[selected_name]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_valid, selected_pred, alpha=0.48, color=PLOT_CYAN)
    low = min(float(y_valid.min()), float(selected_pred.min()))
    high = max(float(y_valid.max()), float(selected_pred.max()))
    plt.plot([low, high], [low, high], "--", color=PLOT_ORANGE, label="Ideal")
    plt.xlabel("Actual Productivity Score")
    plt.ylabel("Predicted Productivity Score")
    plt.title("Actual vs. Predicted Productivity Scores")
    plt.legend()
    save_figure(args.output_dir, "05_actual_vs_predicted.png", args.show_plots)

    residuals = y_valid.to_numpy() - selected_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(selected_pred, residuals, alpha=0.45, color=PLOT_CYAN)
    plt.axhline(0, linestyle="--", color=PLOT_ORANGE)
    plt.xlabel("Predicted Productivity Score")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    save_figure(args.output_dir, "06_residual_plot.png", args.show_plots)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
    )
    interval_model = make_model(
        numeric_features, categorical_features, LinearRegression(), select_k
    )
    interval_model.fit(X_fit, y_fit)
    calibration_errors = np.abs(y_cal.to_numpy() - interval_model.predict(X_cal))
    interval_half_width = conformal_quantile(calibration_errors)
    interval_pred = interval_model.predict(X_valid)
    lower = interval_pred - interval_half_width
    upper = interval_pred + interval_half_width
    coverage = float(
        np.mean((y_valid.to_numpy() >= lower) & (y_valid.to_numpy() <= upper))
    )
    prediction_df = pd.DataFrame(
        {
            "Actual": y_valid.to_numpy(),
            "Predicted": interval_pred,
            "Lower_95": lower,
            "Upper_95": upper,
        }
    ).sort_values("Predicted").reset_index(drop=True)
    prediction_df.to_csv(
        args.output_dir / "validation_predictions_with_interval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"95% prediction interval half-width: {interval_half_width:.4f}; "
        f"holdout coverage: {coverage:.2%}"
    )

    plot_df = prediction_df.head(min(120, len(prediction_df)))
    x_axis = np.arange(len(plot_df))
    plt.figure(figsize=(13, 6))
    plt.fill_between(
        x_axis,
        plot_df["Lower_95"].to_numpy(),
        plot_df["Upper_95"].to_numpy(),
        alpha=0.22,
        color=PLOT_BLUE,
        label="95% prediction interval",
    )
    plt.plot(
        x_axis,
        plot_df["Predicted"].to_numpy(),
        color=PLOT_CYAN,
        label="Predicted",
    )
    plt.scatter(
        x_axis,
        plot_df["Actual"].to_numpy(),
        s=17,
        color=PLOT_ORANGE,
        label="Actual",
    )
    plt.xlabel("Validation Samples (sorted by prediction)")
    plt.ylabel("Productivity Score")
    plt.title("Productivity Score Predictions with 95% Prediction Intervals")
    plt.legend()
    save_figure(args.output_dir, "07_prediction_intervals.png", args.show_plots)

    plt.figure(figsize=(11, 5))
    plt.bar(results_df["Model"], results_df["RMSE"], color=PLOT_CYAN)
    plt.ylabel("RMSE (lower is better)")
    plt.title("Holdout Model Comparison")
    plt.xticks(rotation=18, ha="right")
    save_figure(args.output_dir, "08_model_comparison.png", args.show_plots)

    # ======================================================================
    # CRISP-DM 6 - Deployment（部署與成果輸出）
    # 使用全部資料重新 fit 最終 Pipeline，輸出 joblib 模型、欄位 Schema、
    # 評估 CSV、研究圖表與文字摘要，供後續網站及報告重現使用。
    # ======================================================================
    print_title("CRISP-DM 6: Deployment")
    final_model = make_model(
        numeric_features, categorical_features, LinearRegression(), select_k
    )
    final_model.fit(X, y)
    model_path = args.output_dir / "student_productivity_model.joblib"
    joblib.dump(final_model, model_path)
    schema = {
        "target": TARGET,
        "raw_features": RAW_FEATURES,
        "excluded_identifier_columns": ID_COLUMNS,
        "raw_feature_count": len(RAW_FEATURES),
        "transformed_feature_count": transformed_feature_count,
        "selected_feature_count": select_k,
        "data_source": DATA_SOURCE_URL,
        "task_scope": (
            "HW2 regression task predicting the official productivity_score target "
            "from the supplied competition train CSV."
        ),
        "limitation": (
            "Results are predictive associations, not causal effects; competition "
            "test labels are unavailable for independent final evaluation."
        ),
    }
    (args.output_dir / "deployment_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_holdout = results_df.loc[
        results_df["Model"] == selected_name
    ].iloc[0]
    selected_cv = cv_results.loc[cv_results["Model"] == selected_name].iloc[0]
    best_holdout = results_df.iloc[0]
    summary = (
        "HW2 Multiple Linear Regression Summary\n"
        f"Data source: {DATA_SOURCE_URL}\n"
        f"Data file: {args.data_file.name}\n"
        f"Rows: {len(df)}\n"
        f"Target: {TARGET}\n"
        f"Excluded identifiers: {', '.join(ID_COLUMNS)}\n"
        f"Raw predictor count: {len(RAW_FEATURES)}\n"
        f"One-hot transformed feature count: {transformed_feature_count}\n"
        f"SelectKBest selected feature count: {select_k}\n"
        f"Best holdout model: {best_holdout['Model']}\n"
        f"Selected Linear MAE: {selected_holdout['MAE']:.4f}\n"
        f"Selected Linear RMSE: {selected_holdout['RMSE']:.4f}\n"
        f"Selected Linear R2: {selected_holdout['R2']:.4f}\n"
        f"Selected Linear 5-fold CV RMSE: {selected_cv['CV_RMSE_Mean']:.4f} "
        f"+/- {selected_cv['CV_RMSE_SD']:.4f}\n"
        f"95% prediction interval half-width: {interval_half_width:.4f}\n"
        f"95% prediction interval holdout coverage: {coverage:.2%}\n"
        "Important: the supplied file is the competition training export and contains "
        "the official productivity_score target.\n"
        "Interpretation: holdout and cross-validation results estimate predictive "
        "performance; they do not establish causal effects.\n"
    )
    (args.output_dir / "analysis_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"All outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    run()
