import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN, build_preprocessor


def build_linear_regression_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LinearRegression()
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

    pipeline = build_linear_regression_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("MAE:", f"{mae:.2f}")
    print("RMSE:", f"{rmse:.2f}")
    print("R2 Score:", f"{r2:.4f}")

    sample_input = X_test.head(3)
    sample_preds = pipeline.predict(sample_input)
    for i, (pred, actual) in enumerate(zip(sample_preds, y_test.head(3))):
        print(f"Screening {i+1}: Predicted = {pred:.1f} | Actual = {actual}")


if __name__ == "__main__":
    train_and_evaluate()
