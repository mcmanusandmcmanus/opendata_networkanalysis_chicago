# LLM Workflow Checklist — EDA → Data Engineering → Feature Engineering

Use this step-by-step plan when pairing with an LLM. Share the system prompt from `LLM_PROMPT_RECIPES.md` first, then walk through these stages.

## 0) Context & Safety
- Confirm file path: `data/chicago_crime_odp_2022_202511.csv` (or supplied path), schema from `DATASET_CARD.md`.
- Remind the model: Block is coarse and not for exact address inference.

## 1) Ingest & Validation
- Load CSV with explicit schema: strings for IDs/codes, booleans for Arrest/Domestic, tz-aware datetimes (America/Chicago).
- Run checks: expected columns, non-null `ID`/`Date`/`Primary Type`, `Year` equals parsed Date year, valid lat/long ranges.
- Deduplicate by `ID`, keeping the latest `Updated On`; report counts removed.
- Persist a typed Parquet copy partitioned by `Year` (and month if feasible); keep raw CSV untouched.

## 2) Profiling & Quality Report
- Missingness per column; rate of missing coords.
- Cardinality of key categoricals (Primary Type, FBI Code, District, Ward, Community Area); list unknown values.
- Row counts by year/month/day; highlight spikes/drops.
- Arrest/domestic rates overall and by year.
- Output a short JSON report summarizing findings.

## 3) Core EDA
- Time patterns: incidents by hour, day-of-week, month; arrest rate over time.
- Category patterns: counts and arrest rates by Primary Type and FBI Code.
- Geography: counts by District, Ward, Community Area; hexbin/binned lat/long counts; percent missing coords.
- Consistency: check `Year` vs `Date`, extreme/invalid coordinates, outlier days.

## 4) Data Engineering
- Normalize text categories (uppercase trim); standardize nulls to NA.
- Generate tidy tables: incidents table; lookup tables (District, Ward, Community Area) if available; keep metadata (validation summaries).
- Ensure privacy: keep Block masked; consider rounding lat/long to 3–4 decimals or grid IDs for sharing.
- Version artifacts: raw → cleaned → feature-ready Parquet.

## 5) Feature Engineering (general-purpose)
- Temporal: hour, day, day-of-week, weekend flag, month, quarter, season; days since dataset start/end; holiday flag if calendar available.
- Location: district, ward, community area; binned lat/long or grid_id; distance to city center (optional).
- Crime taxonomy: primary type; FBI code; derived buckets (violent/property/other).
- Outcomes/flags: arrest, domestic.
- Recency/rolling: counts in past 7/30 days by district/category; arrest rate in past windows (no future leakage).
- Interaction terms: category × time (e.g., battery at night), category × location (e.g., theft in downtown wards).

## 6) Splitting & Evaluation Prep
- Use time-based split (train on earlier dates, validate on later). Avoid random splits to prevent leakage.
- When building rolling features, ensure they only use past data relative to the prediction time.
- Define baseline tasks (e.g., arrest likelihood, incident volume forecasting) and metrics (PR-AUC/ROC for classification; MAE/MAPE for counts).

## 7) Reporting & Handoff
- Produce concise artifacts: data quality report, EDA highlights, feature list with definitions, and any drift observations.
- Note any gaps (e.g., missing coords rate, unknown codes) and recommended remediations.
- Keep a run log: commands/prompts executed, parameters used, and output paths.
