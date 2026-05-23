
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["hour"]       = df["timestamp"].dt.hour
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month
    df["dayofyear"]  = df["timestamp"].dt.dayofyear
    df["lag_1"]      = df["aqi"].shift(1)
    df["lag_6"]      = df["aqi"].shift(6)
    df["lag_24"]     = df["aqi"].shift(24)
    df["rolling_6"]  = df["aqi"].rolling(6).mean()
    df["rolling_24"] = df["aqi"].rolling(24).mean()
    return df.dropna()


def forecast_city_aqi(readings: list[dict], city_name: str, days: int = 7) -> list[dict]:
   
    if len(readings) < 50:
        logger.warning(f"Insufficient data for {city_name} ({len(readings)} records). Using trend forecast.")
        return _trend_forecast(readings, days)

    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Try XGBoost first (better for non-linear patterns), fall back to RF
    try:
        return _xgboost_forecast(df, city_name, days)
    except Exception as e:
        logger.warning(f"XGBoost failed for {city_name}: {e}. Trying Random Forest.")
        try:
            return _rf_forecast(df, city_name, days)
        except Exception as e2:
            logger.error(f"RF also failed for {city_name}: {e2}. Using trend.")
            return _trend_forecast(readings, days)


def _xgboost_forecast(df: pd.DataFrame, city_name: str, days: int) -> list[dict]:
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    df_feat = _build_features(df)
    feature_cols = ["hour", "dayofweek", "month", "dayofyear", "lag_1", "lag_6", "lag_24", "rolling_6", "rolling_24"]

    X = df_feat[feature_cols]
    y = df_feat["aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    logger.info(f"XGBoost MAE for {city_name}: {mae:.2f}")

    return _generate_predictions(model, df_feat, feature_cols, days, "XGBoost")


def _rf_forecast(df: pd.DataFrame, city_name: str, days: int) -> list[dict]:
    from sklearn.ensemble import RandomForestRegressor

    df_feat = _build_features(df)
    feature_cols = ["hour", "dayofweek", "month", "dayofyear", "lag_1", "lag_6", "lag_24", "rolling_6", "rolling_24"]

    X = df_feat[feature_cols]
    y = df_feat["aqi"]

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return _generate_predictions(model, df_feat, feature_cols, days, "RandomForest")


def _generate_predictions(model, df_feat: pd.DataFrame, feature_cols: list, days: int, model_name: str) -> list[dict]:
   
    predictions = []
    last_row = df_feat.iloc[-1].copy()

    for day_offset in range(1, days + 1):
        target_date = date.today() + timedelta(days=day_offset)
        target_dt   = datetime.combine(target_date, datetime.min.time().replace(hour=12))  # midday

        # Update time features
        last_row["hour"]      = 12
        last_row["dayofweek"] = target_dt.weekday()
        last_row["month"]     = target_dt.month
        last_row["dayofyear"] = target_dt.timetuple().tm_yday

        X_pred = last_row[feature_cols].values.reshape(1, -1)
        pred_aqi = float(model.predict(X_pred)[0])
        pred_aqi = round(max(10, min(500, pred_aqi)), 1)  # Physical bounds

        predictions.append({
            "predicted_date": target_date,
            "predicted_aqi":  pred_aqi,
            "model_used":     model_name,
        })

        # Shift lags for next iteration
        last_row["lag_24"]    = last_row["lag_6"]
        last_row["lag_6"]     = last_row["lag_1"]
        last_row["lag_1"]     = pred_aqi
        last_row["rolling_6"] = (last_row["rolling_6"] * 5 + pred_aqi) / 6
        last_row["rolling_24"]= (last_row["rolling_24"] * 23 + pred_aqi) / 24

    return predictions


def _trend_forecast(readings: list[dict], days: int) -> list[dict]:
    
    if not readings:
        base_aqi = 100.0
    else:
        recent = [r["aqi"] for r in readings[-10:] if r.get("aqi")]
        base_aqi = np.mean(recent) if recent else 100.0

    predictions = []
    for day_offset in range(1, days + 1):
        noise = np.random.normal(0, 10)
        predictions.append({
            "predicted_date": date.today() + timedelta(days=day_offset),
            "predicted_aqi":  round(max(10, base_aqi + noise), 1),
            "model_used":     "LinearTrend",
        })
    return predictions


def get_city_forecast_summary(predictions: list[dict]) -> dict:
   
    if not predictions:
        return {}
    aqis = [p["predicted_aqi"] for p in predictions]
    return {
        "min_aqi":  round(min(aqis), 1),
        "max_aqi":  round(max(aqis), 1),
        "avg_aqi":  round(np.mean(aqis), 1),
        "worst_day": predictions[int(np.argmax(aqis))]["predicted_date"].isoformat(),
        "best_day":  predictions[int(np.argmin(aqis))]["predicted_date"].isoformat(),
    }
