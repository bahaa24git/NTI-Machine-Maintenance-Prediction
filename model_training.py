from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


TARGET = "Remaining_Useful_Life_days"
FAILURE_TARGET = "Failure_Within_7_Days"
ID_COLUMN = "Machine_ID"
MODEL_NAMES = ["Random Forest", "Decision Tree", "Neural Network", "Linear Regression"]
RANDOM_STATE = 42
TRAINING_SAMPLE_SIZE = 60_000
EVALUATION_SAMPLE_SIZE = 12_000


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET, FAILURE_TARGET, ID_COLUMN}
    return [column for column in df.columns if column not in excluded]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def make_regression_pipeline(name: str, features: pd.DataFrame) -> Pipeline:
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Neural Network": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        ),
    }
    return Pipeline([("preprocessor", build_preprocessor(features)), ("model", estimators[name])])


def make_failure_pipeline(features: pd.DataFrame) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", build_preprocessor(features)), ("model", classifier)])


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.maximum(np.asarray(predicted), 0)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R²": float(r2_score(actual, predicted)),
    }


def evaluate_robustness(
    template: Pipeline, data: pd.DataFrame, features: list[str]
) -> dict[str, dict[str, float | str]]:
    results: dict[str, dict[str, float | str]] = {}
    sorted_types = sorted(data["Machine_Type"].unique())
    held_out_types = sorted_types[-max(1, len(sorted_types) // 5) :]
    splits = {
        "Machine-type holdout": data["Machine_Type"].isin(held_out_types),
        "Installation-year holdout": data["Installation_Year"] >= 2035,
    }
    for split_name, test_mask in splits.items():
        train = data.loc[~test_mask]
        test = data.loc[test_mask]
        if train.empty or test.empty:
            results[split_name] = {"Status": "Unavailable"}
            continue
        candidate = clone(template)
        candidate.fit(train[features], train[TARGET])
        prediction = candidate.predict(test[features])
        results[split_name] = {
            **regression_metrics(test[TARGET], prediction),
            "Train rows": int(len(train)),
            "Test rows": int(len(test)),
            "Held out": ", ".join(held_out_types) if split_name == "Machine-type holdout" else "2035–2040",
        }
    return results


def build_reference_profile(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    return {
        "numeric": {
            column: {
                "median": float(df[column].median()) if df[column].notna().any() else 0.0,
                "p05": float(df[column].quantile(0.05)) if df[column].notna().any() else 0.0,
                "p95": float(df[column].quantile(0.95)) if df[column].notna().any() else 0.0,
                "mean": float(df[column].mean()) if df[column].notna().any() else 0.0,
                "std": float(df[column].std()) if df[column].notna().any() else 0.0,
            }
            for column in numeric
        },
        "categorical": {
            column: df[column].mode(dropna=True).iloc[0] if df[column].notna().any() else None
            for column in categorical
        },
    }


def train_artifact(df: pd.DataFrame) -> dict[str, Any]:
    modeling = df.dropna(subset=[TARGET, FAILURE_TARGET]).copy()
    if len(modeling) > TRAINING_SAMPLE_SIZE:
        modeling = modeling.sample(TRAINING_SAMPLE_SIZE, random_state=RANDOM_STATE)
    features = get_feature_columns(modeling)
    x = modeling[features]
    y = modeling[TARGET]
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        x, y, modeling.index, test_size=0.2, random_state=RANDOM_STATE
    )

    evaluation_data = modeling.sample(
        min(EVALUATION_SAMPLE_SIZE, len(modeling)), random_state=RANDOM_STATE
    )
    cv_data = evaluation_data.sample(min(8_000, len(evaluation_data)), random_state=7)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    regression: dict[str, Any] = {}

    for name in MODEL_NAMES:
        pipeline = make_regression_pipeline(name, x_train)
        pipeline.fit(x_train, y_train)
        train_prediction = np.maximum(pipeline.predict(x_train), 0)
        test_prediction = np.maximum(pipeline.predict(x_test), 0)
        train_metrics = regression_metrics(y_train, train_prediction)
        test_metrics = regression_metrics(y_test, test_prediction)
        critical_mask = y_test <= 30
        critical_mae = (
            float(mean_absolute_error(y_test[critical_mask], test_prediction[critical_mask]))
            if critical_mask.any()
            else float("nan")
        )
        predictions = pd.DataFrame(
            {
                "Actual": y_test.to_numpy(),
                "Predicted": test_prediction,
                "Residual": y_test.to_numpy() - test_prediction,
                "Machine_Type": modeling.loc[test_idx, "Machine_Type"].to_numpy(),
                "Machine_ID": modeling.loc[test_idx, ID_COLUMN].to_numpy(),
            }
        )
        by_type = (
            predictions.groupby("Machine_Type")
            .apply(
                lambda group: pd.Series(
                    {
                        "Machines": len(group),
                        "MAE": mean_absolute_error(group["Actual"], group["Predicted"]),
                        "RMSE": mean_squared_error(group["Actual"], group["Predicted"]) ** 0.5,
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        cv_mae = -cross_val_score(
            make_regression_pipeline(name, cv_data[features]),
            cv_data[features],
            cv_data[TARGET],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
        )
        overfit_gap = train_metrics["R²"] - test_metrics["R²"]
        model_step = pipeline.named_steps["model"]
        learning_history = None
        if name == "Neural Network":
            fitted_mlp = model_step.regressor_
            learning_history = {
                "Loss": [float(value) for value in fitted_mlp.loss_curve_],
                "Validation score": [float(value) for value in fitted_mlp.validation_scores_],
            }
        regression[name] = {
            "pipeline": pipeline,
            "metrics": test_metrics,
            "train_metrics": train_metrics,
            "critical_mae": critical_mae,
            "cv_mae_mean": float(cv_mae.mean()),
            "cv_mae_std": float(cv_mae.std()),
            "overfit_warning": bool(overfit_gap > 0.05),
            "predictions": predictions.sample(min(5_000, len(predictions)), random_state=RANDOM_STATE),
            "by_machine_type": by_type,
            "robustness": evaluate_robustness(
                make_regression_pipeline(name, evaluation_data[features]), evaluation_data, features
            ),
            "learning_history": learning_history,
        }

    failure_pipeline = make_failure_pipeline(x_train)
    failure_pipeline.fit(x_train, modeling.loc[train_idx, FAILURE_TARGET].astype(bool))
    failure_actual = modeling.loc[test_idx, FAILURE_TARGET].astype(bool)
    failure_probability = failure_pipeline.predict_proba(x_test)[:, 1]
    failure_prediction = failure_probability >= 0.5
    classifier_metrics = {
        "ROC AUC": float(roc_auc_score(failure_actual, failure_probability)),
        "Accuracy": float(accuracy_score(failure_actual, failure_prediction)),
        "Precision": float(precision_score(failure_actual, failure_prediction, zero_division=0)),
        "Recall": float(recall_score(failure_actual, failure_prediction, zero_division=0)),
        "F1": float(f1_score(failure_actual, failure_prediction, zero_division=0)),
        "Positive rate": float(failure_actual.mean()),
    }

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "version": f"rul-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        "trained_at": trained_at,
        "training_rows": int(len(modeling)),
        "features": features,
        "regression": regression,
        "failure_classifier": {"pipeline": failure_pipeline, "metrics": classifier_metrics},
        "reference": build_reference_profile(modeling, features),
        "notes": {
            "split": "Fixed random 80/20 split; robustness includes machine-type and installation-year holdouts.",
            "time_split": "Unavailable because the dataset contains no observation timestamp.",
        },
    }


def save_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
