# SRC Directory

This folder contains runnable Python source code for a full classical-ML workflow.

## Current files
- `data_utils.py`: dataset loading and simple feature engineering helpers.
- `train.py`: end-to-end training pipeline with cross-validation and hyperparameter tuning.
- `train_cv_tuned.py`: thin entrypoint wrapper to run `train.py`.
- `inference.py`: batch inference utility for CSV inputs.

## How to run
From the project root:

```bash
python SRC/train_cv_tuned.py
```

This will generate:
- `Results/modeling_tables/*.csv` (CV/tuning/test summaries)
- `Results/modeling_figures/tuned_roc_comparison.png`
- `Results/modeling_figures/tuned_pr_comparison.png`
- `Models/best_model.joblib`
- `Models/best_model_metadata.json`

## Inference

Run batch inference on a CSV containing the original 30 Breast Cancer Wisconsin feature columns:

```bash
python SRC/inference.py \
	--model Models/best_model.joblib \
	--input Data/new_patients.csv \
	--output Results/predictions/new_patients_predictions.csv \
	--threshold 0.5
```

Notes:
- `--threshold` is optional and defaults to `0.5`.
- Input columns are validated against the expected raw dataset schema before prediction.
