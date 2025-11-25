# Chicago Crime ODP CSV (2022–2025) — Data Card

## File
- Name: `chicago_crime_odp_2022_202511.csv` (≈273 MB), comma-delimited, double-quoted fields.
- Storage: keep under `data/` (ignored by git) or reference an external path.
- Each row = one reported incident in Chicago with time, location (masked block), classification (IUCR + FBI code), and disposition flags (`Arrest`, `Domestic`).
- Timestamps are in local 12-hour format with AM/PM (America/Chicago). Convert to offset-aware datetimes before use.
- Boolean fields are the strings `true`/`false` (lowercase).

## Schema
- `ID` (string/int): Unique event id.
- `Case Number` (string): CPD case identifier.
- `Date` (string): Incident datetime, e.g., `11/16/2025 11:00:00 PM`.
- `Block` (string): Masked block address, e.g., `049XX S PRAIRIE AVE`.
- `IUCR` (string): Illinois Uniform Crime Reporting code.
- `Primary Type` (string): Top-level offense category, uppercase.
- `Description` (string): Sub-category description.
- `Location Description` (string): Venue/context (e.g., `RESIDENCE`, `STREET`).
- `Arrest` (string -> bool): `true` if an arrest was made.
- `Domestic` (string -> bool): `true` if domestic-related.
- `Beat` (string): Police beat number.
- `District` (string): Police district code.
- `Ward` (string): City ward number (as string).
- `Community Area` (string): Chicago community area number.
- `FBI Code` (string): FBI UCR code.
- `X Coordinate` / `Y Coordinate` (string/float): State Plane coordinates.
- `Year` (int): Calendar year of incident.
- `Updated On` (string): Last update timestamp for the record.
- `Latitude` / `Longitude` (float): WGS84 coordinates.
- `Location` (string): Tuple representation `(<lat>, <long>)`.

## Sample Rows (first 4)
```
"14031121","JJ492014","11/16/2025 11:00:00 PM","049XX S PRAIRIE AVE","0810","THEFT","OVER $500","RESIDENCE","false","true","0224","002","3","38","06","1178876","1872367","2025","11/22/2025 03:41:52 PM","41.805072258","-87.619480201","(41.805072258, -87.619480201)"
"14032654","JJ494065","11/16/2025 12:00:00 AM","052XX N HARLEM AVE","0460","BATTERY","SIMPLE","COMMERCIAL / BUSINESS OFFICE","false","false","1613","016","41","10","08B","1127352","1933963","2025","11/23/2025 03:42:58 PM","41.975119511","-87.807066347","(41.975119511, -87.807066347)"
"14030856","JJ491909","11/16/2025 12:00:00 AM","080XX S CRANDON AVE","143A","WEAPONS VIOLATION","UNLAWFUL POSSESSION - HANDGUN","VEHICLE NON-COMMERCIAL","false","false","0414","004","7","46","15","1192920","1852209","2025","11/23/2025 03:42:58 PM","41.749425665","-87.568631033","(41.749425665, -87.568631033)"
"14030230","JJ491043","11/16/2025 12:00:00 AM","080XX S TALMAN AVE","1320","CRIMINAL DAMAGE","TO VEHICLE","STREET","false","false","0835","008","18","70","14","1160113","1851060","2025","11/23/2025 03:42:58 PM","41.747009861","-87.688880694","(41.747009861, -87.688880694)"
```

## Data Handling Notes
- Trim surrounding quotes; comma is the delimiter.
- Convert `Arrest`/`Domestic` to booleans; treat blanks as null.
- Parse `Date`/`Updated On` to timezone-aware datetimes (America/Chicago).
- `Block` is coarse for privacy; avoid inferring exact addresses.
- Expect occasional missing coordinates; prefer `Latitude`/`Longitude` over `X/Y` if present.
- Deduplicate on `ID` if needed; `Updated On` can be used to keep the latest revision.

## Suggested LLM Tasks
- Structured Q&A: Filter incidents by date range, offense, arrest flag, district, ward, or community area.
- Aggregations: Counts by `Primary Type`, time-of-day/day-of-week, arrest rate by category, domestic vs non-domestic splits.
- Geospatial summaries: Top blocks or community areas for a category; approximate clustering by lat/long bins.
- Data quality checks: Flag rows with missing coords, invalid dates, or mismatched `Year` vs `Date`.
- Record rendering: Generate concise human-readable summaries for incidents (e.g., “THEFT over $500 at 049XX S PRAIRIE AVE on 2025-11-16 23:00, domestic, no arrest”).

## Output Guidelines for Downstream Models
- Prefer JSON for machine consumption; include filters used and counts (e.g., `{"query":"theft 2025-11-16","rows": [...], "total": 12}`).
- When aggregating, include denominator and percentage where relevant.
- Always state if results are based on a subset/sample vs the full file.
