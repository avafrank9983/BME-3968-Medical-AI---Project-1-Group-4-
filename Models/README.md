# Models Directory

This folder stores trained model artifacts and related metadata.

## Expected contents
- Serialized model files (`.pkl`, `.joblib`, etc.)
- Optional scaler/transform artifacts
- Run metadata (date, seed, feature set, performance)

## Naming suggestion
`<model_name>_<feature_set>_<yyyymmdd>.joblib`

Example:
`logistic_regression_all_features_20260304.joblib`
