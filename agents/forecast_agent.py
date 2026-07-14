import sqlite3
import joblib
import pandas as pd
import numpy as np

# Load saved artifacts once at import time
model = joblib.load("forecast_model.pkl")
encoders = joblib.load("forecast_encoders.pkl")
feature_cols = joblib.load("forecast_features.pkl")
residual_stats = joblib.load("forecast_residuals.pkl")

RESIDUAL_STD = residual_stats["std"]
ANOMALY_THRESHOLD = 2 * RESIDUAL_STD  # flag if actual deviates more than 2 std from prediction


def _build_features_for_range(region: str, category: str, channel: str, start_date: str, end_date: str):
    """Pulls raw data and rebuilds the same features used in training, for a specific slice."""
    conn = sqlite3.connect("data/sales.db")
    df = pd.read_sql_query("""
        SELECT date, region, category, channel,
               SUM(units_sold) AS units_sold,
               AVG(price_unit) AS price_unit,
               MAX(promotion_flag) AS promotion_flag,
               AVG(stock_available) AS stock_available
        FROM sales
        WHERE region = ? AND category = ? AND channel = ?
        GROUP BY date, region, category, channel
        ORDER BY date
    """, conn, params=[region, category, channel])
    conn.close()

    df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Lag + rolling features (need full history before start_date to compute these correctly)
    df["lag_7"] = df["units_sold"].shift(7)
    df["lag_14"] = df["units_sold"].shift(14)
    df["lag_28"] = df["units_sold"].shift(28)
    df["rolling_mean_7"] = df["units_sold"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["units_sold"].shift(1).rolling(7).std()

    # Encode categoricals using the SAME encoders from training
    df["region_enc"] = encoders["region"].transform(df["region"])
    df["category_enc"] = encoders["category"].transform(df["category"])
    df["channel_enc"] = encoders["channel"].transform(df["channel"])

    # Now filter down to just the requested date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    result = df[mask].dropna(subset=feature_cols).reset_index(drop=True)
    return result


def forecast_agent(region: str, category: str, channel: str, start_date: str, end_date: str):
    """
    Returns actual vs predicted units_sold for the given slice, with anomaly flags.
    """
    df = _build_features_for_range(region, category, channel, start_date, end_date)

    if len(df) == 0:
        return {"error": "No data available for this combination (or insufficient history for lag features)."}

    X = df[feature_cols]
    predictions = model.predict(X)

    df["predicted_units_sold"] = predictions
    df["deviation"] = df["units_sold"] - df["predicted_units_sold"]
    df["is_anomaly"] = df["deviation"].abs() > ANOMALY_THRESHOLD

    output_rows = df[["date", "units_sold", "predicted_units_sold", "deviation", "is_anomaly"]].copy()
    output_rows["date"] = output_rows["date"].dt.strftime("%Y-%m-%d")

    anomalies = output_rows[output_rows["is_anomaly"]]

    return {
        "rows": output_rows.to_dict("records"),
        "anomaly_count": int(len(anomalies)),
        "anomaly_dates": anomalies["date"].tolist(),
        "anomaly_threshold": float(round(ANOMALY_THRESHOLD, 2)),
        "mean_deviation": float(round(output_rows["deviation"].mean(), 2)),
    }

if __name__ == "__main__":
    result = forecast_agent(
        region="PL-South",
        category="Yogurt",
        channel="Retail",
        start_date="2024-10-01",
        end_date="2024-12-31"
    )
    print("Anomaly count:", result.get("anomaly_count"))
    print("Anomaly dates:", result.get("anomaly_dates"))
    print("Mean deviation:", result.get("mean_deviation"))
    print("Sample rows:", result.get("rows", [])[:5])