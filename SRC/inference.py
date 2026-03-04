import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_utils import add_engineered_features


def run_inference(model_path: Path, input_csv: Path, output_csv: Path) -> None:
    model = joblib.load(model_path)
    patient_table = pd.read_csv(input_csv)

    enriched_table = add_engineered_features(patient_table)
    probability_malignant = model.predict_proba(enriched_table)[:, 1]
    predicted_label = (probability_malignant >= 0.5).astype(int)

    output_table = patient_table.copy()
    output_table["predicted_malignant"] = predicted_label
    output_table["probability_malignant"] = probability_malignant
    output_table.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained breast cancer model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to serialized model file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV with feature columns.")
    parser.add_argument("--output", type=Path, required=True, help="Path for output CSV predictions.")

    arguments = parser.parse_args()
    run_inference(arguments.model, arguments.input, arguments.output)
