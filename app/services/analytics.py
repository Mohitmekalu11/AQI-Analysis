

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_city_ranking(readings_data: list[dict]) -> list[dict]:
    """
    Daily city ranking by average AQI.
    Returns sorted list from cleanest to most polluted.
    """
    if not readings_data:
        return []

    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Last 24 hours only
    cutoff = datetime.utcnow() - timedelta(hours=24)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return []

    ranking = (
        df.groupby("city_name")["aqi"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_aqi", "min": "min_aqi", "max": "max_aqi", "count": "readings"})
        .sort_values("avg_aqi")
    )

    ranking["avg_aqi"] = ranking["avg_aqi"].round(1)
    ranking["rank"]    = range(1, len(ranking) + 1)
    ranking["category"] = ranking["avg_aqi"].apply(_classify_aqi)

    return ranking.to_dict(orient="records")


def get_aqi_trend(readings_data: list[dict], city_name: str, hours: int = 168) -> dict:
    """
    Hourly AQI trend for a city over the last N hours.
    Default 168h = 7 days.
    """
    if not readings_data:
        return {"labels": [], "values": [], "categories": []}

    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df = df[(df["city_name"] == city_name) & (df["timestamp"] >= cutoff)]

    if df.empty:
        return {"labels": [], "values": [], "categories": []}

    # Resample to hourly averages for smooth chart
    df = df.set_index("timestamp").sort_index()
    hourly = df["aqi"].resample("1H").mean().dropna()

    return {
        "labels":     [ts.strftime("%b %d %H:%M") for ts in hourly.index],
        "values":     [round(v, 1) for v in hourly.values],
        "categories": [_classify_aqi(v) for v in hourly.values],
    }


def get_city_comparison(readings_data: list[dict], cities: list[str]) -> dict:
    """
    Compare AQI trends for multiple cities side-by-side.
    Returns data formatted for Chart.js multi-line chart.
    """
    if not readings_data or not cities:
        return {}

    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.utcnow() - timedelta(hours=168)
    df = df[df["timestamp"] >= cutoff]

    # Common time index — daily averages for comparison
    result = {"labels": [], "datasets": []}
    all_dates = pd.date_range(
        start=cutoff.date(),
        end=datetime.utcnow().date(),
        freq="D"
    )
    result["labels"] = [d.strftime("%b %d") for d in all_dates]

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
              "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]

    for i, city in enumerate(cities):
        city_df = df[df["city_name"] == city].copy()
        if city_df.empty:
            continue

        city_df = city_df.set_index("timestamp")
        daily = city_df["aqi"].resample("1D").mean().reindex(all_dates).fillna(method="ffill")

        result["datasets"].append({
            "label":           city,
            "data":            [round(v, 1) if not np.isnan(v) else None for v in daily.values],
            "borderColor":     colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "20",
            "tension":         0.4,
        })

    return result


def get_pollutant_breakdown(readings_data: list[dict], city_name: str) -> dict:
    """Average pollutant levels for a city (last 24h)."""
    if not readings_data:
        return {}

    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.utcnow() - timedelta(hours=24)
    df = df[(df["city_name"] == city_name) & (df["timestamp"] >= cutoff)]

    if df.empty:
        return {}

    pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    result = {}
    for p in pollutants:
        if p in df.columns:
            val = df[p].dropna().mean()
            result[p] = round(float(val), 2) if not np.isnan(val) else None

    return result


def check_alerts(readings_data: list[dict], threshold: int = 200) -> list[dict]:
    """Return cities currently exceeding AQI threshold."""
    if not readings_data:
        return []

    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.utcnow() - timedelta(hours=1)
    recent = df[df["timestamp"] >= cutoff]

    if recent.empty:
        return []

    alerts = []
    for city, group in recent.groupby("city_name"):
        avg_aqi = group["aqi"].mean()
        if avg_aqi > threshold:
            alerts.append({
                "city_name": city,
                "avg_aqi":   round(avg_aqi, 1),
                "category":  _classify_aqi(avg_aqi),
                "severity":  "critical" if avg_aqi > 300 else "warning",
            })

    return sorted(alerts, key=lambda x: x["avg_aqi"], reverse=True)


def _classify_aqi(aqi: float) -> str:
    if aqi is None or np.isnan(aqi): return "Unknown"
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"
