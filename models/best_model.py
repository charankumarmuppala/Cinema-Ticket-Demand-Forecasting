import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.pipeline import Pipeline
from models.xgboost_model import build_xgboost_pipeline
from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN


def load_model() -> Pipeline:
    train_path = PROJECT_ROOT / "data" / "processed" / "cinema_ticket_features.csv"
    train_df = pd.read_csv(train_path)
    pipeline = build_xgboost_pipeline()
    pipeline.fit(train_df[ALL_MODEL_FEATURES], train_df[TARGET_COLUMN])
    return pipeline


def predict_tickets(
    film_code: int = 1494,
    cinema_code: int = 466,
    ticket_price: float = 60000.0,
    capacity: float = 350.0,
    tickets_out: int = 0,
    show_time: int = 2,
    month: int = 6,
    quarter: int = 2,
    day: int = 13,
    day_of_week: int = 2,
) -> dict:
    model = load_model()
    is_weekend = 1 if day_of_week in (5, 6) else 0

    input_row = {
        "ticket_price": float(ticket_price),
        "capacity": float(capacity),
        "tickets_out": int(tickets_out),
        "show_time": int(show_time),
        "month": int(month),
        "quarter": int(quarter),
        "day": int(day),
        "is_weekend": int(is_weekend),
        "film_code": int(film_code),
        "day_of_week": int(day_of_week),
        "cinema_code": int(cinema_code),
    }

    input_df = pd.DataFrame([input_row])[ALL_MODEL_FEATURES]
    raw_pred = model.predict(input_df)[0]
    predicted_tickets = max(0, round(float(raw_pred)))
    predicted_tickets = min(predicted_tickets, int(capacity))

    occupancy_pct = (predicted_tickets / capacity) * 100.0 if capacity > 0 else 0.0
    estimated_revenue = predicted_tickets * ticket_price

    return {
        "predicted_tickets_sold": predicted_tickets,
        "capacity": int(capacity),
        "occupancy_rate_percent": round(occupancy_pct, 2),
        "estimated_revenue": round(estimated_revenue, 2),
        "inputs": input_row,
    }


if __name__ == "__main__":
    result = predict_tickets()
    print("Predicted Tickets Sold:", result["predicted_tickets_sold"])
    print("Auditorium Capacity:", result["capacity"])
    print("Projected Occupancy:", f"{result['occupancy_rate_percent']}%")
    print("Estimated Revenue:", f"${result['estimated_revenue']:,.2f}")
