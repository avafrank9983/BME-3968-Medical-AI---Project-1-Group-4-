# Breast Cancer Diagnostic Prediction

BME 3968 — Medical AI — Project 1 (Group 4)

This repository analyzes the Breast Cancer Wisconsin (Diagnostic) dataset and compares multiple machine learning models for benign vs malignant tumor classification.

## Team

- Ava Frank: Created the GitHub repository and wrote the Methods, Results, and Discussion sections of the report.
- Carolina Horey: Wrote the Literature Review and contributed to the Abstract.
- Ayushi Elhence: Wrote the Introduction and contributed to the Abstract.

## Project Overview

The project has two main goals:

1. Perform clinically motivated exploratory data analysis (EDA) on nuclear morphology features.
2. Train and compare baseline ML classifiers, then evaluate cross-validation, hyperparameter tuning, and reduced-feature behavior.

## Clinical Motivation

Breast cancer screening and triage benefit from high-sensitivity diagnostic tools. In this project, malignant cases are treated as the positive class during modeling so that recall-oriented evaluation can directly reflect missed-cancer risk.

## Dataset

- Name: Breast Cancer Wisconsin (Diagnostic)
- Samples: 569
- Predictors: 30 numeric morphology features
- Classes: 2 (Benign, Malignant)
- Source interface used in notebooks: `sklearn.datasets.load_breast_cancer`

Features describe cell nucleus morphology (radius, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each represented via mean, standard error, and worst values.

## Repository Structure

```text
.
├── Data/
├── Models/
├── Notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
├── Results/
│   ├── eda_figures/
│   └── modeling_figures/
├── SRC/
├── README.md
└── requirements.txt
```

Notes:

- `Notebooks/eda.ipynb` contains structured EDA with interpretation after visualizations.
- `Notebooks/modeling.ipynb` contains model training, comparison, interpretability, reduced-feature analysis, and CV/tuning summaries.
- `Results/eda_figures` and `Results/modeling_figures` store generated plots.
- `SRC/` contains runnable scripts for feature engineering, CV+tuning training, and inference.
- `Models/` stores serialized model artifacts and metadata from script-based training.

## Reproducibility

Both notebooks are set up to run top-to-bottom and use a fixed seed (`np.random.seed(42)`) for reproducibility.

## Quick Start

1. Create and activate a virtual environment.
2. Install required packages.
3. Open the notebooks and run all cells.

### Example setup (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Run notebooks in this order:

1. `Notebooks/eda.ipynb`
2. `Notebooks/modeling.ipynb`

Optional script pipeline run:

```bash
python SRC/train_cv_tuned.py
```

Optional batch inference run:

```bash
python SRC/inference.py \
	--model Models/best_model.joblib \
	--input Data/new_patients.csv \
	--output Results/predictions/new_patients_predictions.csv \
	--threshold 0.5
```

## Current Outputs

Examples of generated artifacts:

- EDA: class distribution, feature boxplots/histograms, correlation heatmap, PCA scatter.
- Modeling: confusion matrices, ROC/PR comparison curves, feature importance comparison, CV/tuning summary tables.

Saved figures are written to:

- `Results/eda_figures/`
- `Results/modeling_figures/`

Saved tables and model artifacts:

- `Results/modeling_tables/` (CV, tuning, and test summaries)
- `Models/best_model.joblib`
- `Models/best_model_metadata.json`

## Key Findings (Current Version)

- No missing values in the dataset.
- Strong discriminative signal in morphology-driven size/irregularity features.
- High multicollinearity among geometric features (for example radius/perimeter/area families).
- Multiple models achieve strong classification performance.
- A reduced top-feature subset can still maintain competitive performance.

## Next Improvements

- Add external validation and calibration analysis.