"""Generate business-facing figures and tables from model outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
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


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
INPUT_CLEAN = PROCESSED_DIR / "shopping_behavior_clean.csv"
INPUT_MODEL_METRICS = TABLES_DIR / "model_metrics.csv"
INPUT_MODEL_CLASS_METRICS = TABLES_DIR / "model_class_metrics.csv"
INPUT_MODEL_SUMMARY = TABLES_DIR / "modeling_summary.json"
INPUT_TEST_PRED = TABLES_DIR / "test_predictions_best_model.csv"
INPUT_TEST_PROBA = TABLES_DIR / "test_predictions_with_probabilities.csv"
TARGET_ORDER = ["online", "store", "hybrid"]


def check_inputs() -> None:
    required = [
        INPUT_CLEAN,
        INPUT_MODEL_METRICS,
        INPUT_MODEL_CLASS_METRICS,
        INPUT_MODEL_SUMMARY,
        INPUT_TEST_PRED,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files. Run prepare_data.py and analyze.py first:\n"
            + "\n".join(missing)
        )


def plot_channel_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["shopping_preference"]
        .value_counts(normalize=False)
        .reindex(TARGET_ORDER)
        .fillna(0)
        .astype(int)
        .rename_axis("channel")
        .reset_index(name="customers")
    )
    counts["share"] = (counts["customers"] / counts["customers"].sum()).round(4)
    counts.to_csv(TABLES_DIR / "executive_channel_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts, x="channel", y="customers", palette="Blues_d")
    plt.title("Customer Channel Preference Distribution")
    plt.xlabel("Channel")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "executive_channel_distribution.png", dpi=150)
    plt.close()

    return counts


def plot_model_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = metrics_df.copy()
    metrics_df["model_display"] = metrics_df["model"].str.replace("_", " ").str.title()
    metrics_df = metrics_df.sort_values("macro_f1", ascending=False)

    metrics_df.to_csv(TABLES_DIR / "executive_model_comparison.csv", index=False)

    melted = metrics_df.melt(
        id_vars=["model", "model_display"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model_display", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "executive_model_comparison.png", dpi=150)
    plt.close()

    return metrics_df


def export_best_model_kpis(
    *,
    summary: dict[str, object],
    metrics_df: pd.DataFrame,
    channel_counts: pd.DataFrame,
) -> pd.DataFrame:
    best_model = str(summary["best_model"])
    row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
    kpis = pd.DataFrame(
        [
            {"kpi": "best_model", "value": best_model},
            {"kpi": "accuracy", "value": float(row["accuracy"])},
            {"kpi": "macro_f1", "value": float(row["macro_f1"])},
            {"kpi": "weighted_f1", "value": float(row["weighted_f1"])},
            {"kpi": "train_rows", "value": int(summary["train_rows"])},
            {"kpi": "test_rows", "value": int(summary["test_rows"])},
            {"kpi": "total_customers", "value": int(channel_counts["customers"].sum())},
        ]
    )
    kpis.to_csv(TABLES_DIR / "executive_kpi_summary.csv", index=False)
    return kpis


def load_prediction_frame() -> pd.DataFrame:
    pred_df = pd.read_csv(INPUT_TEST_PRED)
    if INPUT_TEST_PROBA.exists():
        proba_df = pd.read_csv(INPUT_TEST_PROBA)
        if len(proba_df) == len(pred_df):
            proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
            for col in proba_cols:
                pred_df[col] = proba_df[col].values
    return pred_df


def export_predicted_mix(pred_df: pd.DataFrame) -> pd.DataFrame:
    mix = (
        pred_df["predicted"]
        .value_counts()
        .reindex(TARGET_ORDER)
        .fillna(0)
        .astype(int)
        .rename_axis("predicted_channel")
        .reset_index(name="customers")
    )
    mix["share"] = (mix["customers"] / mix["customers"].sum()).round(4)
    mix.to_csv(TABLES_DIR / "executive_predicted_channel_mix.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=mix, x="predicted_channel", y="customers", palette="Greens_d")
    plt.title("Predicted Channel Mix (Test Customers)")
    plt.xlabel("Predicted Channel")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "executive_predicted_channel_mix.png", dpi=150)
    plt.close()

    return mix


def export_segment_recommendations(pred_df: pd.DataFrame) -> pd.DataFrame:
    required_segment_cols = {"gender", "city_tier", "predicted"}
    if not required_segment_cols.issubset(pred_df.columns):
        return pd.DataFrame()

    segment = (
        pred_df.groupby(["gender", "city_tier"], dropna=False)
        .agg(
            customers=("predicted", "size"),
            share_pred_online=("predicted", lambda x: (x == "online").mean()),
            share_pred_store=("predicted", lambda x: (x == "store").mean()),
            share_pred_hybrid=("predicted", lambda x: (x == "hybrid").mean()),
        )
        .reset_index()
    )

    if {"proba_online", "proba_store", "proba_hybrid"}.issubset(pred_df.columns):
        probs = (
            pred_df.groupby(["gender", "city_tier"], dropna=False)[
                ["proba_online", "proba_store", "proba_hybrid"]
            ]
            .mean()
            .reset_index()
        )
        segment = segment.merge(probs, on=["gender", "city_tier"], how="left")

    def recommend(row: pd.Series) -> str:
        scores = {
            "online": row.get("share_pred_online", 0.0),
            "store": row.get("share_pred_store", 0.0),
            "hybrid": row.get("share_pred_hybrid", 0.0),
        }
        top_channel = max(scores, key=scores.get)
        if top_channel == "online":
            return "Prioritize digital ads and online conversion funnels"
        if top_channel == "store":
            return "Prioritize in-store promotions and local retail experience"
        return "Prioritize omni-channel campaigns and cross-channel offers"

    segment["recommendation"] = segment.apply(recommend, axis=1)
    rounded_cols = [
        "share_pred_online",
        "share_pred_store",
        "share_pred_hybrid",
        "proba_online",
        "proba_store",
        "proba_hybrid",
    ]
    for col in rounded_cols:
        if col in segment.columns:
            segment[col] = segment[col].round(4)

    segment = segment.sort_values(["customers"], ascending=False).reset_index(drop=True)
    segment.to_csv(TABLES_DIR / "executive_segment_recommendations.csv", index=False)

    heatmap_df = segment.pivot(
        index="gender",
        columns="city_tier",
        values="share_pred_online",
    )
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Share Predicted Online by Segment")
    plt.xlabel("City Tier")
    plt.ylabel("Gender")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "executive_segment_online_heatmap.png", dpi=150)
    plt.close()

    return segment


def export_driver_table(summary: dict[str, object]) -> pd.DataFrame:
    best_model = str(summary["best_model"])
    path = TABLES_DIR / f"feature_importance_{best_model}.csv"
    if not path.exists():
        return pd.DataFrame()

    drivers = pd.read_csv(path).head(15).copy()
    drivers["rank"] = range(1, len(drivers) + 1)
    drivers = drivers[["rank", "feature", "importance"]]
    drivers["importance"] = drivers["importance"].round(6)
    drivers.to_csv(TABLES_DIR / "executive_top_drivers.csv", index=False)

    top = drivers.head(12).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"], color="#2a9d8f")
    plt.title("Top Drivers Behind Predictions")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "executive_top_drivers.png", dpi=150)
    plt.close()

    return drivers


def export_output_manifest(files: list[Path]) -> None:
    manifest = {
        "generated_files": [str(path) for path in files],
    }
    (TABLES_DIR / "executive_outputs_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    check_inputs()

    clean_df = pd.read_csv(INPUT_CLEAN)
    metrics_df = pd.read_csv(INPUT_MODEL_METRICS)
    summary = json.loads(INPUT_MODEL_SUMMARY.read_text(encoding="utf-8"))
    pred_df = load_prediction_frame()

    channel_counts = plot_channel_distribution(clean_df)
    model_comp = plot_model_comparison(metrics_df)
    kpi_df = export_best_model_kpis(
        summary=summary,
        metrics_df=model_comp,
        channel_counts=channel_counts,
    )
    predicted_mix = export_predicted_mix(pred_df)
    segment_recs = export_segment_recommendations(pred_df)
    drivers = export_driver_table(summary)

    generated_paths = [
        TABLES_DIR / "executive_channel_distribution.csv",
        TABLES_DIR / "executive_model_comparison.csv",
        TABLES_DIR / "executive_kpi_summary.csv",
        TABLES_DIR / "executive_predicted_channel_mix.csv",
        TABLES_DIR / "executive_segment_recommendations.csv",
        TABLES_DIR / "executive_top_drivers.csv",
        FIGURES_DIR / "executive_channel_distribution.png",
        FIGURES_DIR / "executive_model_comparison.png",
        FIGURES_DIR / "executive_predicted_channel_mix.png",
        FIGURES_DIR / "executive_segment_online_heatmap.png",
        FIGURES_DIR / "executive_top_drivers.png",
    ]
    existing = [p for p in generated_paths if p.exists()]
    export_output_manifest(existing)

    print("Business-ready outputs generated.")
    print(f"Best model: {summary['best_model']}")
    print(f"KPI rows exported: {len(kpi_df)}")
    print(f"Predicted mix rows exported: {len(predicted_mix)}")
    print(f"Segment recommendation rows exported: {len(segment_recs)}")
    print(f"Top driver rows exported: {len(drivers)}")
    print(f"Manifest: {TABLES_DIR / 'executive_outputs_manifest.json'}")


if __name__ == "__main__":
    main()
