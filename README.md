# Open Data Network Analysis — Chicago Crime

This repo holds prompts, documentation, and workflow guidance for exploring the Chicago crime incidents CSV (`chicago_crime_odp_2022_202511.csv`) and for coordinating work with an LLM teammate. The heavy CSV is **not committed**; keep it in `data/` locally or alongside the repo.

## Quickstart (local)
- Place the CSV at `data/chicago_crime_odp_2022_202511.csv` (or symlink from its existing location).
- Create a virtual env: `python -m venv .venv && .venv\\Scripts\\activate`.
- Install essentials: `pip install pandas pyarrow polars python-dateutil`.
- Open the docs under `docs/` for schema, prompts, and workflow checklists.

## Repo Layout
- `docs/DATASET_CARD.md` — schema, sample rows, parsing rules, and data handling notes.
- `docs/LLM_PROMPT_RECIPES.md` — reusable system prompt and few-shot examples for the LLM.
- `docs/LLM_WORKFLOW_CHECKLIST.md` — step-by-step EDA → data engineering → feature engineering plan.
- `.gitignore` — ignores large data artifacts and env files.

## Working With an LLM
1) Start every session by pasting the system prompt from `docs/LLM_PROMPT_RECIPES.md`.  
2) Remind the model where the CSV lives and that columns/types are defined in `docs/DATASET_CARD.md`.  
3) Use the workflow checklist as the execution plan; ask for JSON outputs when possible for traceability.  
4) Keep a log of decisions, validation results, and drift/quality findings for future runs.

## Next Steps
- Run initial data quality checks (schema, date parsing, missing coords, duplicates by ID).
- Materialize a typed, partitioned Parquet copy under `data/` for faster iteration.
- Add notebooks or scripts once the exploratory plan is finalized.
