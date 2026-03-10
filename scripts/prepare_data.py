"""Prepare raw data for downstream analysis and modeling."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT_FILENAME = "online vs store shopping dataset.csv"
OUTPUT_CLEAN_FILENAME = "shopping_behavior_clean.csv"
OUTPUT_REPORT_FILENAME = "prepare_data_report.json"

REQUIRED_COLUMNS = {
    "age",
    "monthly_income",
    "daily_internet_hours",
    "smartphone_usage_years",
    "social_media_hours",
    "online_payment_trust_score",
    "tech_savvy_score",
    "monthly_online_orders",
    "monthly_store_visits",
    "avg_online_spend",
    "avg_store_spend",
    "discount_sensitivity",
    "return_frequency",
    "avg_delivery_days",
    "delivery_fee_sensitivity",
    "free_return_importance",
    "product_availability_online",
    "impulse_buying_score",
    "need_touch_feel_score",
    "brand_loyalty_score",
    "environmental_awareness",
    "time_pressure_level",
    "gender",
    "city_tier",
    "shopping_preference",
}

NUMERIC_RANGE_RULES = {
    "age": (16, 100),
    "monthly_income": (0, 1_000_000),
    "daily_internet_hours": (0, 24),
    "smartphone_usage_years": (0, 80),
    "social_media_hours": (0, 24),
    "online_payment_trust_score": (1, 10),
    "tech_savvy_score": (1, 10),
    "monthly_online_orders": (0, 1_000),
    "monthly_store_visits": (0, 1_000),
    "avg_online_spend": (0, 1_000_000),
    "avg_store_spend": (0, 1_000_000),
    "discount_sensitivity": (1, 10),
    "return_frequency": (0, 365),
    "avg_delivery_days": (0, 365),
    "delivery_fee_sensitivity": (1, 10),
    "free_return_importance": (1, 10),
    "product_availability_online": (1, 10),
    "impulse_buying_score": (1, 10),
    "need_touch_feel_score": (1, 10),
    "brand_loyalty_score": (1, 10),
    "environmental_awareness": (1, 10),
    "time_pressure_level": (1, 10),
}


def to_snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def resolve_input_path(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return RAW_DIR / DEFAULT_INPUT_FILENAME


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ("gender", "city_tier", "shopping_preference"):
        out[col] = out[col].astype(str).str.strip()

    out["gender"] = out["gender"].str.lower()
    out["city_tier"] = (
        out["city_tier"]
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
    )
    out["shopping_preference"] = out["shopping_preference"].str.lower()

    shopping_map = {
        "online": "online",
        "store": "store",
        "in-store": "store",
        "instore": "store",
        "hybrid": "hybrid",
        "omnichannel": "hybrid",
        "omni-channel": "hybrid",
    }
    out["shopping_preference"] = out["shopping_preference"].replace(shopping_map)

    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_RANGE_RULES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], int]:
    invalid_mask = pd.Series(False, index=df.index)
    invalid_per_column: dict[str, int] = {}

    for col, (lower, upper) in NUMERIC_RANGE_RULES.items():
        col_invalid = df[col].isna() | (df[col] < lower) | (df[col] > upper)
        invalid_per_column[col] = int(col_invalid.sum())
        invalid_mask = invalid_mask | col_invalid

    unknown_target = ~df["shopping_preference"].isin({"online", "store", "hybrid"})
    invalid_per_column["shopping_preference"] = int(unknown_target.sum())
    invalid_mask = invalid_mask | unknown_target

    unknown_gender = ~df["gender"].isin({"male", "female", "other"})
    invalid_per_column["gender"] = int(unknown_gender.sum())
    invalid_mask = invalid_mask | unknown_gender

    unknown_city_tier = ~df["city_tier"].isin({"tier_1", "tier_2", "tier_3"})
    invalid_per_column["city_tier"] = int(unknown_city_tier.sum())
    invalid_mask = invalid_mask | unknown_city_tier

    rows_removed = int(invalid_mask.sum())
    return df.loc[~invalid_mask].copy(), invalid_per_column, rows_removed


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_monthly_orders"] = out["monthly_online_orders"] + out["monthly_store_visits"]
    out["online_order_share"] = np.where(
        out["total_monthly_orders"] > 0,
        out["monthly_online_orders"] / out["total_monthly_orders"],
        0.0,
    )

    out["total_avg_spend"] = out["avg_online_spend"] + out["avg_store_spend"]
    out["online_spend_share"] = np.where(
        out["total_avg_spend"] > 0,
        out["avg_online_spend"] / out["total_avg_spend"],
        0.0,
    )

    out["digital_engagement_score"] = out[
        [
            "daily_internet_hours",
            "social_media_hours",
            "tech_savvy_score",
            "online_payment_trust_score",
        ]
    ].mean(axis=1)

    out["shopping_preference"] = pd.Categorical(
        out["shopping_preference"], categories=["online", "store", "hybrid"]
    )
    out["gender"] = pd.Categorical(out["gender"], categories=["male", "female", "other"])
    out["city_tier"] = pd.Categorical(
        out["city_tier"], categories=["tier_1", "tier_2", "tier_3"]
    )

    return out


def build_report(
    *,
    input_path: Path,
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    duplicates_removed: int,
    rows_removed_invalid: int,
    invalid_per_column: dict[str, int],
) -> dict[str, object]:
    return {
        "input_file": str(input_path),
        "rows_raw": int(len(raw_df)),
        "columns_raw": int(raw_df.shape[1]),
        "rows_after_cleaning": int(len(cleaned_df)),
        "columns_after_cleaning": int(cleaned_df.shape[1]),
        "missing_values_raw_total": int(raw_df.isna().sum().sum()),
        "missing_values_cleaned_total": int(cleaned_df.isna().sum().sum()),
        "duplicates_removed": duplicates_removed,
        "rows_removed_invalid": rows_removed_invalid,
        "invalid_counts_by_column": invalid_per_column,
        "class_distribution": {
            key: int(value)
            for key, value in cleaned_df["shopping_preference"].value_counts().to_dict().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shopping preference dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to raw CSV (default: data/raw/online vs store shopping dataset.csv).",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    raw_df.columns = [to_snake_case(c) for c in raw_df.columns]

    missing_required = sorted(REQUIRED_COLUMNS - set(raw_df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = normalize_categories(raw_df)
    df = coerce_numeric_columns(df)

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df, invalid_per_column, rows_removed_invalid = drop_invalid_rows(df)
    df = add_features(df)

    output_clean_path = PROCESSED_DIR / OUTPUT_CLEAN_FILENAME
    output_report_path = PROCESSED_DIR / OUTPUT_REPORT_FILENAME

    df.to_csv(output_clean_path, index=False)

    report = build_report(
        input_path=input_path,
        raw_df=raw_df,
        cleaned_df=df,
        duplicates_removed=duplicates_removed,
        rows_removed_invalid=rows_removed_invalid,
        invalid_per_column=invalid_per_column,
    )
    output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Data preparation completed.")
    print(f"Input: {input_path}")
    print(f"Cleaned data written to: {output_clean_path}")
    print(f"Report written to: {output_report_path}")
    print(
        "Rows: "
        f"{report['rows_raw']} raw -> {report['rows_after_cleaning']} cleaned "
        f"(removed {report['rows_raw'] - report['rows_after_cleaning']})."
    )


if __name__ == "__main__":
    main()
