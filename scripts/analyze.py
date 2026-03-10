"""Train and evaluate channel-preference models on cleaned data."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep runtime stable in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
INPUT_FILENAME = "shopping_behavior_clean.csv"
TARGET_COLUMN = "shopping_preference"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_ORDER = ["online", "store", "hybrid"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def model_registry() -> dict[str, object]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            multi_class="multinomial",
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    }


def export_confusion_matrix(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    matrix_df = pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER)

    table_path = TABLES_DIR / f"confusion_matrix_{model_name}.csv"
    matrix_df.to_csv(table_path, index=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()


def export_feature_importance(
    *,
    fitted_pipeline: Pipeline,
    model_name: str,
) -> None:
    model = fitted_pipeline.named_steps["model"]
    preprocess = fitted_pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        feature_importance = pd.DataFrame(coef).abs().mean(axis=0).to_numpy()

    if feature_importance is None:
        return

    importance_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance},
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(TABLES_DIR / f"feature_importance_{model_name}.csv", index=False)

    top = importance_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top 20 Features - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{model_name}.png", dpi=150)
    plt.close()


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    input_path = PROCESSED_DIR / INPUT_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean input dataset not found at {input_path}. Run prepare_data.py first."
        )

    df = pd.read_csv(input_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from {input_path}.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_metrics_rows: list[dict[str, object]] = []
    class_metrics_rows: list[dict[str, object]] = []
    model_outputs: dict[str, dict[str, object]] = {}

    for model_name, estimator in model_registry().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        model_metrics_rows.append(
            {
                "model": model_name,
                "accuracy": round(float(accuracy), 6),
                "macro_f1": round(float(macro_f1), 6),
                "weighted_f1": round(float(weighted_f1), 6),
            }
        )

        for class_label in CLASS_ORDER:
            if class_label not in report:
                continue
            class_metrics_rows.append(
                {
                    "model": model_name,
                    "class": class_label,
                    "precision": round(float(report[class_label]["precision"]), 6),
                    "recall": round(float(report[class_label]["recall"]), 6),
                    "f1_score": round(float(report[class_label]["f1-score"]), 6),
                    "support": int(report[class_label]["support"]),
                }
            )

        export_confusion_matrix(y_true=y_test, y_pred=y_pred, model_name=model_name)

        model_outputs[model_name] = {
            "pipeline": pipeline,
            "predictions": y_pred,
            "report": report,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    metrics_df = pd.DataFrame(model_metrics_rows).sort_values(
        "macro_f1", ascending=False
    )
    class_metrics_df = pd.DataFrame(class_metrics_rows)
    metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)
    class_metrics_df.to_csv(TABLES_DIR / "model_class_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_output = model_outputs[best_model_name]
    best_pipeline = best_output["pipeline"]
    y_pred_best = best_output["predictions"]
    export_feature_importance(fitted_pipeline=best_pipeline, model_name=best_model_name)

    predictions_df = X_test.copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted"] = y_pred_best
    predictions_df.to_csv(TABLES_DIR / "test_predictions_best_model.csv", index=False)

    if hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(X_test)
        class_labels = best_pipeline.named_steps["model"].classes_
        proba_df = pd.DataFrame(
            probabilities,
            columns=[f"proba_{label}" for label in class_labels],
            index=predictions_df.index,
        )
        proba_output = pd.concat(
            [
                predictions_df[["actual", "predicted"]],
                proba_df,
            ],
            axis=1,
        )
        proba_output.to_csv(
            TABLES_DIR / "test_predictions_with_probabilities.csv",
            index=False,
        )

    summary = {
        "best_model": best_model_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_distribution_train": y_train.value_counts().to_dict(),
        "target_distribution_test": y_test.value_counts().to_dict(),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    (TABLES_DIR / "modeling_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Model training and evaluation completed.")
    print(f"Input data: {input_path}")
    print(f"Best model by macro F1: {best_model_name}")
    print(f"Metrics table: {TABLES_DIR / 'model_metrics.csv'}")
    print(f"Summary: {TABLES_DIR / 'modeling_summary.json'}")


if __name__ == "__main__":
    main()
