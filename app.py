import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from theme import apply_theme
from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN

COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
CLEANING_SUMMARY_JSON_PATH = Path("reports/cleaning_summary.json")
FEATURED_DATA_PATH = Path("data/processed/cinema_ticket_features.csv")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_best_model():
    from models.xgboost_model import build_xgboost_pipeline
    if FEATURED_DATA_PATH.exists():
        df = pd.read_csv(FEATURED_DATA_PATH)
        pipe = build_xgboost_pipeline()
        pipe.fit(df[ALL_MODEL_FEATURES], df[TARGET_COLUMN])
        return pipe
    return None


@st.cache_data
def load_comparison_report() -> pd.DataFrame:
    if not COMPARISON_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_CSV_PATH)


@st.cache_data
def load_cleaning_summary() -> dict:
    if not CLEANING_SUMMARY_JSON_PATH.exists():
        return {}
    with open(CLEANING_SUMMARY_JSON_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dropdown_options() -> tuple:
    if not FEATURED_DATA_PATH.exists():
        return [], []
    df = pd.read_csv(FEATURED_DATA_PATH, usecols=["film_code", "cinema_code"])
    film_codes = sorted(df["film_code"].unique().tolist())
    cinema_codes = sorted(df["cinema_code"].unique().tolist())
    return film_codes, cinema_codes


def render_overview_tab():
    st.header("Ticket Sales Forecasting & Demand Analysis")
    st.markdown(
        "An end-to-end regression pipeline that predicts **tickets sold** for a "
        "cinema screening, using only information available before the show "
        "happens (schedule, pricing, venue, and calendar features) -- with "
        "leakage features like `total_sales` and `occu_perc` explicitly excluded "
        "from the model."
    )

    cleaning = load_cleaning_summary()
    comparison_df = load_comparison_report()

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of records (raw)", f"{cleaning.get('rows_before_cleaning', 'N/A'):,}"
                   if cleaning else "N/A")
    with col2:
        st.metric("Number of features (raw)", len(cleaning.get("columns", [])) or "N/A")

    st.subheader("Data Cleaning Summary")
    if cleaning:
        c1, c2, c3 = st.columns(3)
        c1.metric("Duplicates removed", cleaning.get("duplicates_removed", "N/A"))
        c2.metric("Invalid rows removed", cleaning.get("invalid_rows_removed_total", "N/A"))
        missing_handled = (
            cleaning.get("missing_capacity_imputed_per_cinema_median", 0)
            + cleaning.get("missing_capacity_dropped_no_reference", 0)
        )
        c3.metric("Missing values handled", missing_handled)
        st.caption(
            f"Rows before cleaning: {cleaning.get('rows_before_cleaning'):,} -> "
            f"Rows after cleaning: {cleaning.get('rows_after_cleaning'):,} "
            f"({cleaning.get('total_rows_removed')} removed, "
            f"{cleaning.get('total_rows_removed') / cleaning.get('rows_before_cleaning') * 100:.2f}%)."
        )
    else:
        st.warning("Cleaning summary not found. Run src/data_cleaning.py first.")

    st.subheader("Final Feature List (used by the model)")
    st.write(", ".join(f"`{f}`" for f in ALL_MODEL_FEATURES))
    st.caption(
        "Excludes leakage features (`total_sales`, `ticket_use`, `occu_perc`) which "
        "are mathematically derived from the target."
    )

    st.subheader("Best Performing Model")
    if not comparison_df.empty:
        best_row = comparison_df.sort_values("test_rmse").iloc[0]
        st.success(f"**{best_row['model']}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{best_row['test_mae']:.2f}")
        m2.metric("RMSE", f"{best_row['test_rmse']:.2f}")
        m3.metric("R2 Score", f"{best_row['test_r2']:.3f}")
    else:
        st.warning("Model comparison report not found.")


@st.cache_data
def get_feature_importance_figure(dark_mode: bool = False):
    from models.random_forest import build_random_forest_pipeline
    from models.xgboost_model import build_xgboost_pipeline

    if not FEATURED_DATA_PATH.exists():
        return None

    df = pd.read_csv(FEATURED_DATA_PATH)
    sample = df.sample(n=min(10000, len(df)), random_state=42)

    rf = build_random_forest_pipeline(n_estimators=30, max_depth=8)
    rf.fit(sample[ALL_MODEL_FEATURES], sample[TARGET_COLUMN])

    xgb_pipe = build_xgboost_pipeline(n_estimators=100, max_depth=5)
    xgb_pipe.fit(sample[ALL_MODEL_FEATURES], sample[TARGET_COLUMN])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Random Forest: Top 15 Features", "XGBoost: Top 15 Features"],
        horizontal_spacing=0.15,
    )

    def clean_name(raw: str) -> str:
        for prefix in ["num__", "dow_onehot__", "film_onehot__", "cinema_freq__"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        return raw.replace("_freq", " (freq)").replace("_code_", ": ")

    rf_color = "#888890" if dark_mode else "#5B574D"
    xgb_color = "#FFC21A"
    border_color = "#2E2E34" if dark_mode else "#141414"

    for idx, (pipe, col_color) in enumerate([(rf, rf_color), (xgb_pipe, xgb_color)], 1):
        prep = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]
        names = [clean_name(n) for n in prep.get_feature_names_out()]
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1][:15]
        top_names = [names[j] for j in order][::-1]
        top_imps = imps[order][::-1]

        fig.add_trace(go.Bar(
            x=top_imps,
            y=top_names,
            orientation="h",
            marker=dict(color=col_color, line=dict(color=border_color, width=1)),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            showlegend=False,
        ), row=1, col=idx)

    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4F4F6" if dark_mode else "#141414"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2E2E34" if dark_mode else "#E8E2D3")
    fig.update_yaxes(showgrid=False)
    return fig


@st.cache_data
def get_model_comparison_figure(dark_mode: bool = False):
    df = load_comparison_report()
    if df.empty:
        return None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE (lower is better)", "RMSE (lower is better)", "R2 (higher is better)"],
        horizontal_spacing=0.08,
    )

    colors = [
        "#FFC21A" if m == "XGBoost" else ("#888890" if dark_mode else "#5B574D") if m == "Random Forest" else ("#2E2E34" if dark_mode else "#EFEADF")
        for m in df["model"]
    ]
    border_color = "#2E2E34" if dark_mode else "#141414"

    for i, (col, fmt) in enumerate([("test_mae", ".2f"), ("test_rmse", ".2f"), ("test_r2", ".3f")], 1):
        fig.add_trace(go.Bar(
            x=df["model"],
            y=df[col],
            marker=dict(color=colors, line=dict(color=border_color, width=1)),
            text=[f"{v:{fmt}}" for v in df[col]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + col.replace("test_", "").upper() + ": %{y:" + fmt + "}<extra></extra>",
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4F4F6" if dark_mode else "#141414"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=380,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2E2E34" if dark_mode else "#E8E2D3")
    return fig


@st.cache_data
def get_predicted_vs_actual_figure(dark_mode: bool = False):
    test_path = Path("data/processed/test_set.csv")
    if not test_path.exists():
        return None

    model = load_best_model()
    if model is None:
        return None

    test_df = pd.read_csv(test_path)
    sample_df = test_df.sample(n=min(3000, len(test_df)), random_state=42)
    X_samp = sample_df.drop(columns=[TARGET_COLUMN])
    y_samp = sample_df[TARGET_COLUMN].values

    preds = model.predict(X_samp)
    abs_errors = np.abs(y_samp - preds)
    max_val = max(float(y_samp.max()), float(preds.max()))

    marker_color = "#FFC21A" if dark_mode else "#141414"
    line_color = "#FF6B4A" if dark_mode else "#B84A00"
    grid_color = "#2E2E34" if dark_mode else "#E8E2D3"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_samp,
        y=preds,
        mode="markers",
        marker=dict(size=6, color=marker_color, opacity=0.4, line=dict(width=0)),
        name="Screenings",
        customdata=abs_errors,
        hovertemplate="<b>Screening</b><br>Actual: %{x}<br>Predicted: %{y:.0f}<br>Abs Error: %{customdata:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color=line_color, dash="dash", width=2),
        name="Perfect prediction (y = x)",
        hoverinfo="name",
    ))
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4F4F6" if dark_mode else "#141414"),
        xaxis=dict(title="Actual tickets sold", showgrid=True, gridcolor=grid_color),
        yaxis=dict(title="Predicted tickets sold", showgrid=True, gridcolor=grid_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=40),
        height=520,
    )
    return fig


def render_analysis_tab(dark_mode: bool = False):
    st.header("Data Analysis & Model Insights")

    st.subheader("Feature Importance")
    fig_fi = get_feature_importance_figure(dark_mode=dark_mode)
    if fig_fi is not None:
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(
            "Shows which inputs drive each tree-based model's predictions most. "
            "show_time and capacity dominate in both Random Forest and XGBoost."
        )

    st.subheader("Model Comparison")
    fig_mc = get_model_comparison_figure(dark_mode=dark_mode)
    if fig_mc is not None:
        st.plotly_chart(fig_mc, use_container_width=True)
        st.caption(
            "Side-by-side MAE, RMSE, and R2 across all three models on the same held-out test set."
        )

    st.subheader("Predicted vs Actual (Best Model)")
    fig_pva = get_predicted_vs_actual_figure(dark_mode=dark_mode)
    if fig_pva is not None:
        st.plotly_chart(fig_pva, use_container_width=True)
        st.caption(
            "Each point is one test-set screening: x-axis is true tickets sold, y-axis is model prediction."
        )


def build_input_dataframe(film_code, cinema_code, ticket_price, capacity,
                           tickets_out, show_time, month, quarter, day,
                           day_of_week) -> pd.DataFrame:
    is_weekend = 1 if day_of_week in (5, 6) else 0

    row = {
        "ticket_price": ticket_price,
        "capacity": capacity,
        "tickets_out": tickets_out,
        "show_time": show_time,
        "month": month,
        "quarter": quarter,
        "day": day,
        "is_weekend": is_weekend,
        "film_code": film_code,
        "day_of_week": day_of_week,
        "cinema_code": cinema_code,
    }
    return pd.DataFrame([row])[ALL_MODEL_FEATURES]


def render_prediction_tab():
    st.header("Ticket Sales Prediction")
    st.markdown("Fill in the screening details below to get a predicted ticket count.")

    model = load_best_model()
    film_codes, cinema_codes = load_dropdown_options()

    if model is None:
        st.error("No model available.")
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            film_code = st.selectbox("Film Code", options=film_codes if film_codes else [0])
            cinema_code = st.selectbox("Cinema Code", options=cinema_codes if cinema_codes else [0])
            ticket_price = st.number_input("Ticket Price", min_value=1.0, max_value=1_000_000.0, value=80000.0, step=1000.0)
            capacity = st.number_input("Capacity", min_value=1.0, max_value=5000.0, value=300.0, step=1.0)
            tickets_out = st.number_input("Tickets Out", min_value=0, max_value=5000, value=0, step=1)

        with col2:
            show_time = st.number_input("Show Time", min_value=1, max_value=60, value=4, step=1)
            month = st.selectbox("Month", options=list(range(1, 13)), index=4)
            quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=1)
            day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
            day_of_week_name = st.selectbox("Day of Week", options=DAY_NAMES, index=1)
            day_of_week = DAY_NAMES.index(day_of_week_name)

            is_weekend_display = "Yes" if day_of_week in (5, 6) else "No"
            st.text_input("is_weekend", value=is_weekend_display, disabled=True)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        if ticket_price <= 0:
            st.error("Ticket price must be greater than 0.")
            return
        if capacity <= 0:
            st.error("Capacity must be greater than 0.")
            return
        if tickets_out < 0:
            st.error("Tickets out cannot be negative.")
            return

        try:
            input_df = build_input_dataframe(
                film_code, cinema_code, ticket_price, capacity, tickets_out,
                show_time, month, quarter, day, day_of_week,
            )
            prediction = model.predict(input_df)[0]
            prediction = max(0, round(float(prediction)))

            st.markdown("### Result")
            st.metric(label="Predicted Tickets Sold", value=f"{prediction:,}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def main():
    st.set_page_config(
        page_title="Ticket Sales Forecasting",
        layout="wide",
    )
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    col_title, col_btn = st.columns([8, 2])
    with col_title:
        st.title("Ticket Sales Forecasting & Demand Analysis")
    with col_btn:
        st.markdown('<div class="theme-toggle-container">', unsafe_allow_html=True)
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
        st.markdown('</div>', unsafe_allow_html=True)

    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    apply_theme(dark_mode=st.session_state.dark_mode)

    tab1, tab2, tab3 = st.tabs(["Project Overview", "Data Analysis", "Ticket Sales Prediction"])

    with tab1:
        render_overview_tab()
    with tab2:
        render_analysis_tab(dark_mode=st.session_state.dark_mode)
    with tab3:
        render_prediction_tab()


if __name__ == "__main__":
    main()
