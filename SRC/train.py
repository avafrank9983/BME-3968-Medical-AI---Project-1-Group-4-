import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_breast_cancer_dataset


RANDOM_STATE = 42
sns.set_style("whitegrid")
np.random.seed(RANDOM_STATE)


def build_model_specs() -> dict[str, dict]:
    return {
        "Logistic Regression": {
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                steps=[
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_split": [2, 5],
            },
        },
        "Support Vector Machine": {
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__C": [0.5, 1.0, 2.0],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__kernel": ["rbf"],
            },
        },
        "Gradient Boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },
        "K-Nearest Neighbors": {
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
    }


def evaluate_on_test_set(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_malignant": precision_score(y_true, y_pred, zero_division=0),
        "recall_malignant": recall_score(y_true, y_pred, zero_division=0),
        "f1_malignant": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "Results" / "modeling_tables"
    figure_dir = project_root / "Results" / "modeling_figures"
    model_dir = project_root / "Models"

    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_breast_cancer_dataset(add_features=True)

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model_specs = build_model_specs()
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_rows = []
    tuning_rows = []
    test_rows = []
    best_estimators: dict[str, Pipeline] = {}
    probability_outputs: dict[str, np.ndarray] = {}

    for model_name, model_spec in model_specs.items():
        pipeline = model_spec["pipeline"]

        cv_scores = cross_validate(
            pipeline,
            train_features,
            train_target,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
        )

        cv_rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": np.mean(cv_scores["test_accuracy"]),
                "cv_precision_mean": np.mean(cv_scores["test_precision"]),
                "cv_recall_mean": np.mean(cv_scores["test_recall"]),
                "cv_f1_mean": np.mean(cv_scores["test_f1"]),
                "cv_roc_auc_mean": np.mean(cv_scores["test_roc_auc"]),
            }
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec["param_grid"],
            scoring="roc_auc",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(train_features, train_target)

        best_estimators[model_name] = grid_search.best_estimator_

        tuning_rows.append(
            {
                "model": model_name,
                "best_cv_roc_auc": grid_search.best_score_,
                "best_params": str(grid_search.best_params_),
            }
        )

        test_predictions = grid_search.best_estimator_.predict(test_features)
        test_probabilities = grid_search.best_estimator_.predict_proba(test_features)[:, 1]
        probability_outputs[model_name] = test_probabilities

        test_metrics = evaluate_on_test_set(test_target, test_predictions, test_probabilities)
        test_metrics["model"] = model_name
        test_rows.append(test_metrics)

    cv_summary = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    tuning_summary = pd.DataFrame(tuning_rows).sort_values("best_cv_roc_auc", ascending=False)
    test_summary = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)

    cv_summary.to_csv(results_dir / "cv_summary.csv", index=False)
    tuning_summary.to_csv(results_dir / "tuning_summary.csv", index=False)
    test_summary.to_csv(results_dir / "test_summary.csv", index=False)

    best_model_name = test_summary.iloc[0]["model"]
    best_model = best_estimators[best_model_name]

    joblib.dump(best_model, model_dir / "best_model.joblib")
    metadata = {
        "best_model": best_model_name,
        "test_metrics": test_summary.iloc[0].to_dict(),
    }
    (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 6), dpi=130)
    for model_name, probabilities in probability_outputs.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(test_target, probabilities)
        auc_value = roc_auc_score(test_target, probabilities)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Tuned Model ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "tuned_roc_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6), dpi=130)
    for model_name, probabilities in probability_outputs.items():
        precision, recall, _ = precision_recall_curve(test_target, probabilities)
        ap_value = average_precision_score(test_target, probabilities)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap_value:.3f})")

    plt.title("Tuned Model Precision-Recall Curves")
    plt.xlabel("Recall (Malignant)")
    plt.ylabel("Precision (Malignant)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "tuned_pr_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    print("Saved cross-validation, tuning, and test summaries to Results/modeling_tables")
    print(f"Best model: {best_model_name} (saved to Models/best_model.joblib)")


if __name__ == "__main__":
    main()
