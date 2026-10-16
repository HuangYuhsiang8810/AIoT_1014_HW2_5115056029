"""Generate static model-lab data from StudentsPerformance_train.csv."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "StudentsPerformance_train.csv"
OUTPUT_FILE = ROOT / "site-demo" / "dist" / "assets" / "model_experiments.json"
TARGET = "productivity_score"
ID_COLUMNS = ["id", "student_id"]
FEATURES = [
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
NUMERIC = [feature for feature in FEATURES if feature != "gender"]
CATEGORICAL = ["gender"]
RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        sparse_threshold=0,
    )


def forward_sfs_ranking(X: np.ndarray, y: np.ndarray) -> list[int]:
    selected: list[int] = []
    remaining = list(range(X.shape[1]))
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    while remaining:
        candidates: list[tuple[float, int]] = []
        for candidate in remaining:
            columns = selected + [candidate]
            rmse = -cross_val_score(
                LinearRegression(),
                X[:, columns],
                y,
                scoring="neg_root_mean_squared_error",
                cv=cv,
            ).mean()
            candidates.append((float(rmse), candidate))
        _, winner = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(winner)
        remaining.remove(winner)
    return selected


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET].to_numpy(dtype=float)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor = make_preprocessor()
    train_matrix = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    valid_matrix = np.asarray(preprocessor.transform(X_valid), dtype=float)
    feature_names = preprocessor.get_feature_names_out().tolist()
    feature_count = len(feature_names)

    f_scores, _ = f_regression(train_matrix, y_train)
    kbest_ranking = np.argsort(-np.nan_to_num(f_scores, nan=-np.inf)).tolist()

    rfe = RFE(LinearRegression(), n_features_to_select=1, step=1)
    rfe.fit(train_matrix, y_train)
    rfe_ranking = np.argsort(rfe.ranking_).tolist()
    sfs_ranking = forward_sfs_ranking(train_matrix, y_train)

    lasso_selector = LassoCV(cv=5, max_iter=30000, random_state=RANDOM_STATE)
    lasso_selector.fit(train_matrix, y_train)
    lasso_ranking = np.argsort(-np.abs(lasso_selector.coef_)).tolist()

    permutation_model = Ridge(alpha=1.0).fit(train_matrix, y_train)
    permutation = permutation_importance(
        permutation_model,
        train_matrix,
        y_train,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    permutation_ranking = np.argsort(-permutation.importances_mean).tolist()

    rankings = {
        "kbest": kbest_ranking,
        "rfe": rfe_ranking,
        "sfs": sfs_ranking,
        "lasso": lasso_ranking,
        "permutation": permutation_ranking,
    }
    method_metadata = {
        "kbest": {
            "label": "SelectKBest",
            "detail": "F-regression 單變量排序。速度快，但不會同時評估特徵組合。",
        },
        "rfe": {
            "label": "RFE",
            "detail": "用線性回歸反覆移除係數較弱的特徵。",
        },
        "sfs": {
            "label": "Forward SFS",
            "detail": "使用訓練資料 3-fold CV，逐步加入能降低 RMSE 的特徵。",
        },
        "lasso": {
            "label": "Lasso ranking",
            "detail": "依 LassoCV 係數絕對值排序，偏好稀疏特徵集合。",
        },
        "permutation": {
            "label": "Permutation Importance",
            "detail": "以 Ridge 為基礎，依打亂欄位後的訓練 RMSE 劣化幅度排序。",
        },
    }
    model_metadata = {
        "linear": {
            "label": "LinearRegression",
            "detail": "一般最小平方法，沒有正則化。",
        },
        "ridge": {
            "label": "Ridge",
            "detail": "L2 正則化，alpha = 1.0。",
        },
        "lasso": {
            "label": "Lasso",
            "detail": "L1 正則化，alpha = 0.05。",
        },
    }

    baseline_prediction = np.full_like(y_valid, y_train.mean(), dtype=float)
    configs: dict[str, dict] = {}
    for method_key, ranking in rankings.items():
        for k in range(1, feature_count + 1):
            indices = ranking[:k]
            selected_names = [feature_names[index] for index in indices]
            for model_key in model_metadata:
                if model_key == "linear":
                    estimator = LinearRegression()
                elif model_key == "ridge":
                    estimator = Ridge(alpha=1.0)
                else:
                    estimator = Lasso(alpha=0.05, max_iter=30000)
                estimator.fit(train_matrix[:, indices], y_train)
                prediction = estimator.predict(valid_matrix[:, indices])
                configs[f"{method_key}|{k}|{model_key}"] = {
                    "method": method_key,
                    "k": k,
                    "model": model_key,
                    "selected": selected_names,
                    "metrics": regression_metrics(y_valid, prediction),
                    "intercept": float(estimator.intercept_),
                    "coefficients": {
                        name: float(value)
                        for name, value in zip(
                            selected_names, estimator.coef_, strict=True
                        )
                    },
                    "nonzero_count": int(
                        np.count_nonzero(np.abs(estimator.coef_) > 1e-10)
                    ),
                }

    numeric_pipeline = preprocessor.named_transformers_["num"]
    scaler = numeric_pipeline.named_steps["scaler"]
    input_schema: dict[str, dict] = {}
    for feature in NUMERIC:
        series = df[feature]
        is_integer = bool(pd.api.types.is_integer_dtype(series.dtype))
        input_schema[feature] = {
            "type": "number",
            "min": float(series.min()),
            "max": float(series.max()),
            "default": float(series.median()),
            "step": 1 if is_integer else 0.01,
        }
    for feature in CATEGORICAL:
        input_schema[feature] = {
            "type": "category",
            "values": sorted(df[feature].astype(str).unique().tolist()),
            "default": str(df[feature].mode().iloc[0]),
        }

    best_key, best_config = min(
        configs.items(), key=lambda item: item[1]["metrics"]["rmse"]
    )
    payload = {
        "meta": {
            "rows": len(df),
            "raw_feature_count": len(FEATURES),
            "candidate_count": feature_count,
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "random_state": RANDOM_STATE,
            "target": TARGET,
            "excluded_identifiers": ID_COLUMNS,
            "total_configurations": len(configs),
            "best_holdout_key": best_key,
            "best_holdout_rmse": best_config["metrics"]["rmse"],
        },
        "raw_features": FEATURES,
        "input_schema": input_schema,
        "feature_names": feature_names,
        "numeric_stats": {
            name: {"mean": float(mean), "scale": float(scale)}
            for name, mean, scale in zip(
                NUMERIC, scaler.mean_, scaler.scale_, strict=True
            )
        },
        "methods": method_metadata,
        "models": model_metadata,
        "baseline_metrics": regression_metrics(y_valid, baseline_prediction),
        "configs": configs,
    }
    OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Generated {len(configs)} configurations -> {OUTPUT_FILE}")
    print(f"Best holdout setting: {best_key}; RMSE={best_config['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
