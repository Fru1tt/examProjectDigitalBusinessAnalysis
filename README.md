# EDI36001 Exam Project (Spring 2026)

Group project repository for Digital Business Analysis (DBA).

Formal assignment info:
https://portal.bi.no/en/exams/assignment-thesis/

## Purpose

Build and document an end-to-end, data-driven business analysis:
1. Define a business problem.
2. Acquire and prepare relevant data.
3. Perform analysis.
4. Present findings with effective visualizations.
5. Deliver a clear technical and business-oriented report.

## Project Structure

```text
examProject/
├── data/
│   ├── raw/            # Original source files (not versioned)
│   └── processed/      # Cleaned/analysis-ready data (not versioned)
├── notebooks/          # Exploratory work and visuals
├── scripts/            # Repeatable pipeline scripts
├── outputs/
│   ├── figures/        # Final plots for report/presentation
│   └── tables/         # Final tables (csv/xlsx/md)
├── docs/               # Problem framing, method, report outline
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Recommended Workflow

1. Start with [`docs/01_problem_definition.md`](docs/01_problem_definition.md).
2. Document data availability in [`docs/02_data_plan.md`](docs/02_data_plan.md).
3. Run pipeline scripts as they mature:
   - `python scripts/prepare_data.py`
   - `python scripts/analyze.py`
   - `python scripts/make_outputs.py`
4. Keep exploratory analysis in `notebooks/` and move finalized logic into `scripts/`.
5. Use [`docs/04_report_outline.md`](docs/04_report_outline.md) to structure final delivery.

## Team Conventions

- Use branch names like `feature/<short-topic>` or `analysis/<topic>`.
- Keep commits small and descriptive.
- Save final visuals/tables in `outputs/` with clear names.
- Never rewrite raw source data manually; clean in code.
