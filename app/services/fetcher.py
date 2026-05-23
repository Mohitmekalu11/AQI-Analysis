
import logging
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# OpenAQ v2 — free, public API
OPENAQ_BASE = "https://api.openaq.io/v2"

# City → OpenAQ location IDs (pre-mapped for reliability)
CITY_OPENAQ_MAP = {
    "Delhi":     {"country": "IN", "city": "Delhi"},
    "Mumbai":    {"country": "IN", "city": "Mumbai"},
    "Nagpur":    {"country": "IN", "city": "Nagpur"},
    "Pune":      {"country": "IN", "city": "Pune"},
    "Bengaluru": {"country": "IN", "city": "Bengaluru"},
    "Chennai":   {"country": "IN", "city": "Chennai"},
    "Hyderabad": {"country": "IN", "city": "Hyderabad"},
    "Kolkata":   {"country": "IN", "city": "Kolkata"},
    "Ahmedabad": {"country": "IN", "city": "Ahmedabad"},
    "Jaipur":    {"country": "IN", "city": "Jaipur"},
}

# Realistic AQI ranges per city (based on actual historical data)
CITY_AQI_PROFILES = {
    "Delhi":     (120, 320, 25),   # (min, max, std)
    "Mumbai":    (60, 180, 20),
    "Nagpur":    (50, 160, 18),
    "Pune":      (45, 150, 15),
    "Bengaluru": (40, 130, 15),
    "Chennai":   (50, 140, 18),
    "Hyderabad": (55, 160, 20),
    "Kolkata":   (90, 250, 25),
    "Ahmedabad": (80, 220, 22),
    "Jaipur":    (70, 200, 20),
}


def fetch_city_aqi(city_name: str) -> dict | None:
    """
    Fetch latest AQI data for a city from OpenAQ v2.
    Returns cleaned dict or None on failure.
    """
    config = CITY_OPENAQ_MAP.get(city_name)
    if not config:
        logger.warning(f"No OpenAQ config for city: {city_name}")
        return None

    try:
        params = {
            "city":    config["city"],
            "country": config["country"],
            "limit":   100,
            "order_by": "lastUpdated",
            "sort":    "desc",
        }
        resp = requests.get(
            f"{OPENAQ_BASE}/latest",
            params=params,
            timeout=10,
            headers={"User-Agent": "AQI-Platform/1.0 (educational project)"}
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            logger.info(f"No OpenAQ results for {city_name}, using simulation.")
            return _simulate_reading(city_name)

        # Aggregate pollutant values from all stations in the city
        pollutants = {"pm25": [], "pm10": [], "co": [], "no2": [], "so2": [], "o3": []}

        for station in data["results"]:
            for measurement in station.get("measurements", []):
                param = measurement.get("parameter", "").lower().replace(".", "")
                value = measurement.get("value")
                if value is not None and value >= 0:
                    if param == "pm25":   pollutants["pm25"].append(value)
                    elif param == "pm10": pollutants["pm10"].append(value)
                    elif param == "co":   pollutants["co"].append(value)
                    elif param == "no2":  pollutants["no2"].append(value)
                    elif param == "so2":  pollutants["so2"].append(value)
                    elif param == "o3":   pollutants["o3"].append(value)

        averaged = {k: round(float(np.mean(v)), 2) if v else None for k, v in pollutants.items()}
        averaged["aqi"] = _calculate_aqi(averaged.get("pm25"), averaged.get("pm10"))
        averaged["source"] = "OpenAQ"
        averaged["timestamp"] = datetime.utcnow()

        return _clean_reading(averaged)

    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAQ fetch failed for {city_name}: {e}")
        return _simulate_reading(city_name)


def _calculate_aqi(pm25: float | None, pm10: float | None) -> float | None:
    """
    US EPA AQI formula for PM2.5 (most commonly used in India too).
    Breakpoints: https://www.airnow.gov/aqi/aqi-calculator-concentration/
    """
    if pm25 is None:
        return None

    # PM2.5 AQI breakpoints
    bp = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]

    for c_low, c_high, i_low, i_high in bp:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)
    return min(500.0, round(pm25 * 2, 1))


def _simulate_reading(city_name: str) -> dict:
    """
    Generate realistic simulated AQI data when API is unavailable.
    Uses historical profiles with seasonal variation.
    """
    profile = CITY_AQI_PROFILES.get(city_name, (60, 200, 20))
    mn, mx, std = profile

    # Add time-of-day variation (worse in morning/evening rush hours)
    hour = datetime.now().hour
    time_factor = 1.3 if (7 <= hour <= 10 or 17 <= hour <= 21) else 1.0

    base_aqi = random.gauss((mn + mx) / 2, std) * time_factor
    aqi = round(max(mn * 0.7, min(mx * 1.1, base_aqi)), 1)

    # Derive pollutants from AQI (approximate real-world ratios)
    pm25  = round(aqi * 0.35 + random.gauss(0, 3), 2)
    pm10  = round(pm25 * 1.8 + random.gauss(0, 5), 2)
    no2   = round(aqi * 0.15 + random.gauss(0, 2), 2)
    so2   = round(aqi * 0.05 + random.gauss(0, 1), 2)
    co    = round(aqi * 0.8 + random.gauss(0, 10), 2)
    o3    = round(max(10, aqi * 0.12 + random.gauss(0, 2)), 2)

    return {
        "aqi": aqi, "pm25": max(0, pm25), "pm10": max(0, pm10),
        "no2": max(0, no2), "so2": max(0, so2),
        "co": max(0, co), "o3": max(0, o3),
        "source": "Simulated", "timestamp": datetime.utcnow(),
    }


def _clean_reading(data: dict) -> dict:
    """
    Data Cleaning Pipeline using Pandas.
    Handles: missing values, outliers, unit normalization.
    """
    df = pd.DataFrame([data])

    numeric_cols = ["aqi", "pm25", "pm10", "co", "no2", "so2", "o3"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Remove physically impossible negatives
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)

    # Cap extreme outliers at 99th percentile thresholds (domain knowledge)
    caps = {"aqi": 500, "pm25": 500, "pm10": 600, "co": 10000, "no2": 400, "so2": 350, "o3": 300}
    for col, cap in caps.items():
        df[col] = df[col].clip(upper=cap)

    # Fill NaN with None (JSON-serializable)
    result = df.iloc[0].to_dict()
    for col in numeric_cols:
        val = result.get(col)
        if pd.isna(val):
            result[col] = None
        else:
            result[col] = round(float(val), 2)

    return result


def generate_historical_data(city_name: str, days: int = 30) -> list[dict]:
    """
    Generate 30 days of historical data for a city.
    Used for initial DB seeding and ML training.
    """
    readings = []
    profile = CITY_AQI_PROFILES.get(city_name, (60, 200, 20))
    mn, mx, std = profile

    for i in range(days * 24):  # Hourly readings
        ts = datetime.utcnow() - timedelta(hours=i)
        hour = ts.hour

        # Seasonal + diurnal variation
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * ts.timetuple().tm_yday / 365)
        diurnal  = 1.2 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0.9

        base = ((mn + mx) / 2) * seasonal * diurnal
        aqi  = round(max(mn * 0.5, min(mx * 1.2, random.gauss(base, std))), 1)
        pm25 = round(aqi * 0.35 + random.gauss(0, 3), 2)

        readings.append({
            "aqi": aqi,
            "pm25": max(0, pm25),
            "pm10": round(pm25 * 1.8, 2),
            "no2":  round(aqi * 0.15, 2),
            "so2":  round(aqi * 0.05, 2),
            "co":   round(aqi * 0.8, 2),
            "o3":   round(max(10, aqi * 0.12), 2),
            "source": "Historical",
            "timestamp": ts,
        })

    return readings
