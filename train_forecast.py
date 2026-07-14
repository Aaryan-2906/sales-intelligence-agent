import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------
# 1. Load and aggregate data
# ---------------------------------------------------------
# We forecast at region+category+day granularity — daily totals
# are more stable and business-meaningful than per-SKU noise.

conn = sqlite3.connect("data/sales.db")
df = pd.read_sql_query("""
    SELECT date, region, category, channel,
           SUM(units_sold) AS units_sold,
           AVG(price_unit) AS price_unit,
           MAX(promotion_flag) AS promotion_flag,
           AVG(stock_available) AS stock_available
    FROM sales
    GROUP BY date, region, category, channel
""", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["region", "category", "channel", "date"]).reset_index(drop=True)

print(f"Loaded {len(df)} rows after aggregation")


# ---------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------

# Calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# Lag + rolling features, computed PER region+category+channel group
# so we don't leak information across unrelated series
group_cols = ["region", "category", "channel"]

df["lag_7"] = df.groupby(group_cols)["units_sold"].shift(7)
df["lag_14"] = df.groupby(group_cols)["units_sold"].shift(14)
df["lag_28"] = df.groupby(group_cols)["units_sold"].shift(28)

df["rolling_mean_7"] = (
    df.groupby(group_cols)["units_sold"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)
df["rolling_std_7"] = (
    df.groupby(group_cols)["units_sold"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

# Drop rows with NaN from lag/rolling window (early rows in each group)
df = df.dropna().reset_index(drop=True)
print(f"{len(df)} rows remain after dropping lag warm-up NaNs")

# Encode categoricals
encoders = {}
for col in ["region", "category", "channel"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "day_of_week", "month", "is_weekend", "is_month_start", "is_month_end",
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_std_7",
    "price_unit", "promotion_flag", "stock_available",
    "region_enc", "category_enc", "channel_enc"
]

X = df[feature_cols]
y = df["units_sold"]


# ---------------------------------------------------------
# 3. Time-based train/test split (never random for time series!)
# ---------------------------------------------------------
split_date = pd.Timestamp("2024-01-01")
train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)} rows (before {split_date.date()})")
print(f"Test:  {len(X_test)} rows (on/after {split_date.date()})")


# ---------------------------------------------------------
# 4. Train and compare three models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
    trained_models[name] = model

print("\n=== MODEL COMPARISON ===")
for name, metrics in results.items():
    print(f"{name:20s}  MAE: {metrics['MAE']:8.2f}   RMSE: {metrics['RMSE']:8.2f}")

# Pick the best model by MAE
best_model_name = min(results, key=lambda k: results[k]["MAE"])
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------
# 5. Save best model + encoders + feature list + test residuals
#    (residuals needed later for anomaly detection thresholds)
# ---------------------------------------------------------
joblib.dump(best_model, "forecast_model.pkl")
joblib.dump(encoders, "forecast_encoders.pkl")
joblib.dump(feature_cols, "forecast_features.pkl")

test_preds = best_model.predict(X_test)
residuals = y_test.values - test_preds
joblib.dump({"mean": residuals.mean(), "std": residuals.std()}, "forecast_residuals.pkl")

print("\nSaved model, encoders, feature list, and residual stats.")