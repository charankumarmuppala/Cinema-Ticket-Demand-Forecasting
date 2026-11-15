# Cinema Ticket Demand Forecasting & Analysis

A comprehensive **Data Science** project that analyzes historical cinema ticket sales and builds predictive models to forecast **movie ticket demand** using machine learning and an interactive Streamlit dashboard.

The project explores the factors influencing ticket sales, compares multiple regression models, and provides real-time ticket demand predictions through a production-ready web application.

---

## Live Dashboard

**Streamlit App:**  
https://cinematicketdemandforecasting.streamlit.app/

---

## Project Objectives

* Analyze historical cinema ticket sales data
* Perform data cleaning, preprocessing, and feature engineering
* Compare multiple regression models for ticket demand prediction
* Evaluate model performance using standard regression metrics
* Deploy an interactive Streamlit dashboard for real-time predictions

---

## Features

### Module 1 — Data Analysis

* Dataset Overview
* Data Cleaning Summary
* Feature Engineering
* Missing Value Handling
* Data Validation

### Module 2 — Model Evaluation

* Linear Regression
* Random Forest Regression
* XGBoost Regression
* Model Performance Comparison
* Feature Importance Analysis
* Predicted vs Actual Visualization

### Module 3 — Ticket Sales Prediction

* Interactive prediction form
* Real-time ticket sales forecasting
* Automatic preprocessing pipeline
* XGBoost-powered predictions

### Interactive Dashboard

* Project overview
* Data analysis and model comparison
* Feature importance visualization
* Predicted vs Actual analysis
* Real-time ticket sales prediction

---

## Project Structure

```text
Cinema-Ticket-Demand-Forecasting/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── cinema_ticket.csv
│   │
│   └── processed/
│       ├── cinema_ticket_cleaned.csv
│       ├── cinema_ticket_features.csv
│       └── test_set.csv
│
├── models/
│   ├── __init__.py
│   ├── best_model.py
│   ├── linear_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── reports/
│   ├── best_model_predicted_vs_actual.png
│   ├── cleaning_summary.json
│   ├── cleaning_summary.txt
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   └── model_comparison_chart.png
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── train.py
│
├── .gitignore
├── app.py
├── requirements.txt
├── theme.py
└── README.md
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Plotly
* Matplotlib
* Joblib

---

## Model Performance

The project compares three regression models:

* Linear Regression
* Random Forest Regression
* XGBoost Regression

### Best Model (XGBoost)

* **R² Score:** **0.81**
* **RMSE:** **118.59**
* **MAE:** **65.82**

XGBoost achieved the best predictive performance and was deployed in the Streamlit application for real-time inference.

---

## Dataset

The project uses a real-world cinema ticket sales dataset containing **142,524 screening records**.

The dataset includes:

* Film information
* Cinema information
* Ticket sales
* Ticket price
* Capacity
* Show timings
* Occupancy percentage
* Calendar features
* Date information

After preprocessing:

* Duplicate rows removed
* Missing values handled
* Invalid records cleaned
* Feature engineering performed
* Machine learning-ready dataset generated

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/charankumarmuppala/Cinema-Ticket-Demand-Forecasting.git
cd Cinema-Ticket-Demand-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## Future Improvements

* Time-series forecasting models
* Holiday and festival impact analysis
* Movie genre and cast-based demand prediction
* External event and weather integration
* Hyperparameter optimization using Bayesian Optimization
* Deep Learning-based demand forecasting

---

## Author

**Charan Kumar Muppala**

* GitHub: [@charankumarmuppala](https://github.com/charankumarmuppala)
* Email: charankumarmuppala@gmail.com

---

## License

This project is intended for educational and research purposes.
