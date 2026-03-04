# Data Directory

This folder stores project data artifacts.

## Expected contents
- Raw datasets (if externally downloaded)
- Intermediate processed tables (optional)
- Data dictionaries / metadata files

## Current project behavior
The notebooks currently load the Breast Cancer Wisconsin (Diagnostic) dataset directly from `scikit-learn` (`load_breast_cancer`), so no local raw dataset file is required to run the current workflow.

## Suggested convention
- `raw/` for untouched inputs
- `processed/` for cleaned or engineered outputs
- Include a short provenance note for each file added
