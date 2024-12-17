# Food Price Inflation Prediction with Machine Learning and Plotly Dashboard

## Project Overview

This project analyzes and predicts food price inflation using **machine learning models**. A Plotly-powered interactive **dashboard** has been designed for end-users to explore food price trends, predictions, and contributing economic factors.

Key highlights:
- **Data Cleaning and Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering** (Lag features, rolling averages)
- **Model Development and Evaluation**: Ridge Regression, Random Forest, XGBoost, Lasso Regression
- **Interactive Plotly Dashboard** for visualizing historical data and predictions.

---

## Table of Contents

1. [Project Workflow](#project-workflow)
2. [Dataset Description](#dataset-description)
3. [Key Findings](#key-findings)
4. [How to Run the Code](#how-to-run-the-code)
5. [Model Comparison](#model-comparison)
6. [Dashboard Features](#dashboard-features)
7. [Code Highlights](#code-highlights)
8. [Conclusion](#conclusion)

---

## Project Workflow

1. **Loading Datasets**: Combined *Food Price Index* and *World Development Indicators (WDI)* datasets.
2. **Data Cleaning**:
   - Handling missing values using medians and averages.
   - Outlier detection using **Z-Score** and **IQR Method**.
   - Applied **log transformation** to normalize skewed data.
3. **Feature Engineering**:
   - Lagging features: `inflation_lag_1`, `inflation_lag_3`, `inflation_lag_6`.
   - Rolling averages: `inflation_rolling_3`, `inflation_rolling_6`.
4. **Exploratory Data Analysis**:
   - Trends over time, seasonal patterns, and correlation heatmaps.
5. **Model Development**:
   - Trained Ridge Regression, Random Forest, XGBoost, and Lasso models.
   - Fine-tuned Random Forest and Ridge models using GridSearchCV.
6. **Model Evaluation**:
   - Compared models using **RMSE**, **R² Score**, and **MAE** on test data.
7. **Interactive Dashboard**:
   - Built a user-friendly dashboard to show food price trends and predictions.

---

## Dataset Description

### Datasets Used:
1. **Food Price Dataset**:
   - Contains food price indices (opening/closing) and inflation values from 2007 to 2023.
2. **WDI Dataset**:
   - Includes GDP, tax revenue, and debt service indicators for various countries.

### Common Countries:
- 36 countries were used (e.g., Afghanistan, Bangladesh, Indonesia, Kenya).

### Features Created:
- `inflation_lag_1`, `inflation_lag_3`, `inflation_lag_6`: Previous values of inflation.
- `inflation_rolling_3`, `inflation_rolling_6`: Rolling averages for smoothing trends.

---

## Key Findings

1. **Seasonal Patterns**: Higher inflation observed during spring months (March-May).
2. **Trends**:
   - A sharp rise in 2008 followed by volatility in inflation.
   - Peak inflation between **2020-2022** due to global disruptions.
3. **Model Performance**:
   - **Random Forest** and **Ridge Regression** were the top-performing models:
     - Ridge Regression (Test R²: 87.55%, RMSE: 0.4420).
     - Random Forest (Test R²: 84.38%, RMSE: 0.4952).

---

## How to Run the Code

### 1. Prerequisites

Install the required libraries:
```bash
pip install pandas numpy scikit-learn statsmodels dash dash-bootstrap-components plotly matplotlib seaborn xgboost
```

### 2. Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/food-price-prediction.git
   cd food-price-prediction
   ```

2. **Run the Jupyter Notebook** for data preprocessing and model training:
   ```bash
   jupyter notebook
   ```

3. **Run the Dashboard**:
   - The dashboard code is in `dashboard_plotly.py`.
   - Execute the following:
     ```bash
     python dashboard_plotly.py
     ```

   - Open your browser and navigate to `http://127.0.0.1:8050/`.

---

## Model Comparison

| Model                | Training RMSE | Test RMSE | Test R² Score | Test MAE |
|----------------------|---------------|-----------|---------------|----------|
| **Ridge Regression** | 0.4993        | 0.4420    | 0.8755        | 0.3202   |
| **Random Forest**    | 0.1701        | 0.4952    | 0.8438        | 0.3273   |
| XGBoost              | 0.1756        | 0.6215    | 0.7539        | 0.4560   |
| Lasso Regression     | 0.5130        | 0.4682    | 0.8603        | 0.3422   |

### Best Model:
- **Ridge Regression**: Best R² Score and lowest RMSE on test data.

---

## Dashboard Features

The Plotly Dashboard provides:
1. **Dataset Selection**:
   - Dropdown to choose **country** and **prediction model**.
   - Date range slider for time customization.
2. **Visualizations**:
   - Historical and predicted trends in food price indices.
   - Interactive line charts with hover tooltips for exact values.
   - Comparative charts showing the impact of GDP, tax revenue, etc.
3. **Model Performance Metrics**:
   - Display RMSE, R² Score, and MAE for the selected model.
4. **Download Options**:
   - Export predictions as CSV or PDF.

---

## Code Highlights

- **Feature Engineering**:
   ```python
   # Adding Lag Features
   food_price_data['inflation_lag_3'] = food_price_data['inflation'].shift(3)
   ```

- **Model Training**:
   ```python
   from sklearn.linear_model import Ridge
   ridge = Ridge(alpha=1)
   ridge.fit(X_train, y_train)
   ```

- **Plotly Visualization**:
   ```python
   fig = px.line(df, x='year', y='predictions', title='Predicted Trends')
   ```

---

## Conclusion

This project successfully:
- Identified trends and seasonal patterns in food price inflation.
- Built and compared machine learning models for accurate predictions.
- Designed an interactive dashboard to provide valuable insights to users.

### Next Steps:
1. Add **ARIMA** for time series forecasting.
2. Improve feature selection and fine-tuning for better accuracy.
3. Extend dashboard to allow real-time data updates.

---
