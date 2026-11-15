import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN, build_preprocessor


def build_random_forest_pipeline(
    n_estimators: int = 30,
    max_depth: int = 8,
    min_samples_leaf: int = 4,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_and_evaluate():
    train_data_path = PROJECT_ROOT / "data" / "processed" / "cinema_ticket_features.csv"
    test_data_path = PROJECT_ROOT / "data" / "processed" / "test_set.csv"

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    X_train = train_df[ALL_MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[ALL_MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("MAE:", f"{mae:.2f}")
    print("RMSE:", f"{rmse:.2f}")
    print("R2 Score:", f"{r2:.4f}")

    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1][:5]

    for rank, idx in enumerate(order, 1):
        clean_name = feature_names[idx].split("__")[-1]
        print(f"{rank}. {clean_name}: {importances[idx]:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
