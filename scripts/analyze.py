"""Run core business analysis on processed data."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Analysis scaffold is ready.")
    print(f"Read processed data from: {PROCESSED_DIR}")
    print(f"Write result tables to: {TABLES_DIR}")
    print("Next: implement KPI calculations and model/statistical logic.")


if __name__ == "__main__":
    main()
