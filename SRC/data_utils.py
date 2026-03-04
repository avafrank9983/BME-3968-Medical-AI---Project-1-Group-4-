import pandas as pd
from sklearn.datasets import load_breast_cancer


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    engineered = features.copy()
    epsilon = 1e-8

    engineered["area_perimeter_ratio"] = engineered["mean area"] / (engineered["mean perimeter"] + epsilon)
    engineered["compactness_concavity_interaction"] = engineered["mean compactness"] * engineered["mean concavity"]
    engineered["worst_radius_texture_interaction"] = engineered["worst radius"] * engineered["worst texture"]
    engineered["concavity_point_density"] = engineered["mean concavity"] / (engineered["mean concave points"] + epsilon)

    return engineered


def load_breast_cancer_dataset(add_features: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()

    feature_frame = frame.drop(columns=["target"])
    target = frame["target"].copy()
    malignant_target = (target == 0).astype(int)

    if add_features:
        feature_frame = add_engineered_features(feature_frame)

    return feature_frame, malignant_target
