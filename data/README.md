# Data

This project uses public, aggregate data from the Centers for Medicare & Medicaid Services. Raw and processed files are intentionally excluded from version control so the repository remains lightweight and the analysis can be reproduced from authoritative sources.

Run `python src/retrieve_public_data.py` to retrieve and prepare the source data. The modeling pipeline expects the resulting panel at:

```text
data/processed/analytic_panel_2018_2025.csv.gz
```

CMS-suppressed utilization is treated as missing rather than zero. The data cannot identify diagnoses, indications, prescriber intent, inventory, dispensing delays, stockouts, or individual patients.
