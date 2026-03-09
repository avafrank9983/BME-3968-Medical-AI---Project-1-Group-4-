import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

from data_utils import add_engineered_features


def _get_expected_feature_columns() -> list[str]:
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.feature_names)


def _validate_input_columns(patient_table: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in expected_columns if column not in patient_table.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "Input CSV is missing required raw feature columns: "
            f"{missing_text}."
        )

    # Keep only expected raw features and preserve training-time column order.
    return patient_table[expected_columns].copy()


def run_inference(model_path: Path, input_csv: Path, output_csv: Path, threshold: float = 0.5) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    model = joblib.load(model_path)
    patient_table = pd.read_csv(input_csv)
    expected_columns = _get_expected_feature_columns()
    aligned_raw_table = _validate_input_columns(patient_table, expected_columns)

    enriched_table = add_engineered_features(aligned_raw_table)
    probability_malignant = model.predict_proba(enriched_table)[:, 1]
    predicted_label = (probability_malignant >= threshold).astype(int)

    output_table = patient_table.copy()
    output_table["predicted_malignant"] = predicted_label
    output_table["probability_malignant"] = probability_malignant
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained breast cancer model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to serialized model file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV with feature columns.")
    parser.add_argument("--output", type=Path, required=True, help="Path for output CSV predictions.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for malignant class prediction (default: 0.5).",
    )

    arguments = parser.parse_args()
    run_inference(arguments.model, arguments.input, arguments.output, threshold=arguments.threshold)
