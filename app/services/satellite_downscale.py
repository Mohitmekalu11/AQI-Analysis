
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ── CPCB Station Metadata ──────────────────────────────────────────────────────
# Real CPCB monitoring stations with approximate coordinates
CPCB_STATIONS = {
    "Delhi": [
        {"station_id": "DPCC_001", "name": "Anand Vihar",       "lat": 28.6469, "lon": 77.3160, "zone": "industrial"},
        {"station_id": "DPCC_002", "name": "IGI Airport T3",    "lat": 28.5562, "lon": 77.1000, "zone": "residential"},
        {"station_id": "DPCC_003", "name": "Punjabi Bagh",      "lat": 28.6742, "lon": 77.1311, "zone": "residential"},
        {"station_id": "DPCC_004", "name": "RK Puram",          "lat": 28.5651, "lon": 77.1830, "zone": "residential"},
        {"station_id": "DPCC_005", "name": "Okhla Phase 2",     "lat": 28.5355, "lon": 77.2720, "zone": "industrial"},
    ],
    "Mumbai": [
        {"station_id": "MPCB_001", "name": "Bandra Kurla Complex", "lat": 19.0596, "lon": 72.8656, "zone": "commercial"},
        {"station_id": "MPCB_002", "name": "Chembur",              "lat": 19.0522, "lon": 72.8992, "zone": "industrial"},
        {"station_id": "MPCB_003", "name": "Worli",                "lat": 18.9987, "lon": 72.8156, "zone": "residential"},
    ],
    "Bengaluru": [
        {"station_id": "KSPCB_001", "name": "Silk Board",   "lat": 12.9176, "lon": 77.6237, "zone": "traffic"},
        {"station_id": "KSPCB_002", "name": "Hebbal",       "lat": 13.0358, "lon": 77.5970, "zone": "residential"},
    ],
    "Kolkata": [
        {"station_id": "WBPCB_001", "name": "Rabindra Sarani", "lat": 22.5893, "lon": 88.3534, "zone": "commercial"},
        {"station_id": "WBPCB_002", "name": "Jadavpur",        "lat": 22.4990, "lon": 88.3712, "zone": "residential"},
    ],
    "Chennai": [
        {"station_id": "TNPCB_001", "name": "Alandur",     "lat": 13.0003, "lon": 80.2055, "zone": "residential"},
        {"station_id": "TNPCB_002", "name": "Manali",      "lat": 13.1673, "lon": 80.2714, "zone": "industrial"},
    ],
    "Hyderabad": [
        {"station_id": "TSPCB_001", "name": "Sanathnagar",    "lat": 17.4490, "lon": 78.4500, "zone": "industrial"},
        {"station_id": "TSPCB_002", "name": "Somajiguda",     "lat": 17.4239, "lon": 78.4738, "zone": "commercial"},
    ],
    "Pune": [
        {"station_id": "MPCB_P01",  "name": "Bhosari",        "lat": 18.6298, "lon": 73.8552, "zone": "industrial"},
        {"station_id": "MPCB_P02",  "name": "Shivajinagar",   "lat": 18.5308, "lon": 73.8475, "zone": "residential"},
    ],
    "Ahmedabad": [
        {"station_id": "GPCB_001",  "name": "Maninagar",      "lat": 22.9960, "lon": 72.6023, "zone": "residential"},
        {"station_id": "GPCB_002",  "name": "Vatva GIDC",     "lat": 22.9500, "lon": 72.6400, "zone": "industrial"},
    ],
    "Jaipur": [
        {"station_id": "RSPCB_001", "name": "Mansarovar",     "lat": 26.8665, "lon": 75.7700, "zone": "residential"},
    ],
    "Nagpur": [
        {"station_id": "MPCB_N01",  "name": "Civil Lines",    "lat": 21.1562, "lon": 79.0849, "zone": "residential"},
    ],
}


# ── Sentinel-5P / VEDAS Satellite Data Simulation ─────────────────────────────
# In production: replace with actual VEDAS API calls to vedas.sac.gov.in
# or Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu)

def fetch_satellite_no2(city: str, date: datetime = None) -> dict:
    """
    Fetch tropospheric NO₂ column density from satellite.

    In production: call VEDAS SAC ISRO API or Copernicus Sentinel-5P TROPOMI.
    Here we simulate realistic values for Indian cities based on published data.

    Returns:
        {
          city, date, no2_column_density (µmol/m²),
          spatial_resolution_km, data_source,
          satellite_pixels: [{lat, lon, no2}]  — coarse 7x7 km grid
        }
    """
    if date is None:
        date = datetime.utcnow()

    # Realistic tropospheric NO₂ column densities for Indian cities (µmol/m²)
    # Based on Sentinel-5P TROPOMI published data
    CITY_NO2_PROFILES = {
        "Delhi":     (0.0012, 0.0045, 0.0008),  # (min, max, std) — very high
        "Mumbai":    (0.0006, 0.0025, 0.0005),
        "Kolkata":   (0.0010, 0.0038, 0.0007),
        "Bengaluru": (0.0004, 0.0018, 0.0004),
        "Chennai":   (0.0005, 0.0020, 0.0004),
        "Hyderabad": (0.0006, 0.0022, 0.0005),
        "Pune":      (0.0004, 0.0018, 0.0004),
        "Ahmedabad": (0.0007, 0.0030, 0.0006),
        "Jaipur":    (0.0005, 0.0022, 0.0005),
        "Nagpur":    (0.0004, 0.0016, 0.0004),
    }

    profile = CITY_NO2_PROFILES.get(city, (0.0005, 0.0020, 0.0005))
    mn, mx, std = profile

    # Seasonal effect: higher in winter (Oct-Feb) due to temperature inversion
    month = date.month
    seasonal_factor = 1.4 if month in [11, 12, 1, 2] else (0.8 if month in [6, 7, 8] else 1.0)

    base_no2 = random.gauss((mn + mx) / 2, std) * seasonal_factor
    city_no2 = round(max(mn * 0.5, min(mx * 1.3, base_no2)), 6)

    # Generate a coarse satellite grid (7 km pixels) around the city
    stations = CPCB_STATIONS.get(city, [])
    if stations:
        center_lat = np.mean([s["lat"] for s in stations])
        center_lon = np.mean([s["lon"] for s in stations])
    else:
        center_lat, center_lon = 20.5937, 78.9629  # India center fallback

    # 5x5 pixel grid, ~0.063° ≈ 7km resolution
    satellite_pixels = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pixel_no2 = city_no2 * (1 + random.gauss(0, 0.15))
            # Industrial zones have higher NO2
            pixel_no2 = round(max(0, pixel_no2), 6)
            satellite_pixels.append({
                "lat": round(center_lat + i * 0.063, 4),
                "lon": round(center_lon + j * 0.063, 4),
                "no2_column": pixel_no2,
                "qa_value": round(random.uniform(0.5, 1.0), 2),  # Quality flag
            })

    return {
        "city":                  city,
        "date":                  date.strftime("%Y-%m-%d"),
        "no2_column_density":    city_no2,
        "no2_unit":              "µmol/m²",
        "spatial_resolution_km": 7.0,
        "data_source":           "Sentinel-5P TROPOMI (simulated VEDAS data)",
        "satellite":             "Sentinel-5P",
        "product":               "L2__NO2___",
        "satellite_pixels":      satellite_pixels,
        "pixel_count":           len(satellite_pixels),
        "seasonal_factor":       round(seasonal_factor, 2),
    }


def fetch_cpcb_ground_data(city: str) -> list[dict]:
    """
    Fetch ground-level NO₂ from CPCB stations.

    In production: use CPCB Central Control Room API
    (https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository)

    Returns list of station readings with NO₂ in µg/m³
    """
    stations = CPCB_STATIONS.get(city, [])
    if not stations:
        return []

    # Realistic ground NO₂ ranges by zone type (µg/m³)
    ZONE_NO2 = {
        "industrial":  (40, 120, 15),
        "traffic":     (50, 150, 20),
        "commercial":  (30, 100, 12),
        "residential": (20, 80, 10),
    }

    # Delhi multiplier (worst city)
    city_factor = 1.5 if city == "Delhi" else 1.0

    readings = []
    for station in stations:
        mn, mx, std = ZONE_NO2.get(station["zone"], (25, 90, 12))
        no2 = round(max(5, random.gauss((mn + mx) / 2, std) * city_factor), 1)

        readings.append({
            "station_id":   station["station_id"],
            "station_name": station["name"],
            "city":         city,
            "lat":          station["lat"],
            "lon":          station["lon"],
            "zone":         station["zone"],
            "no2_ugm3":     no2,
            "pm25_ugm3":    round(max(0, no2 * 2.1 + random.gauss(0, 5)), 1),
            "pm10_ugm3":    round(max(0, no2 * 3.8 + random.gauss(0, 8)), 1),
            "timestamp":    datetime.utcnow().isoformat(),
            "source":       "CPCB CCR",
        })

    return readings


# ── AI/ML Downscaling Pipeline ────────────────────────────────────────────────

def downscale_no2(
    city: str,
    satellite_data: dict,
    ground_data: list[dict],
    method: str = "xgboost",
    target_resolution_km: float = 1.0,
) -> dict:
    """
    SIH1734 Core Algorithm: Downscale satellite NO₂ from 7km → 1km resolution.

    Method:
      1. Use CPCB ground stations as training truth
      2. Build auxiliary features (lat, lon, zone_type, time_of_day, seasonal_factor)
      3. Train ML model to predict fine-scale NO₂ from coarse satellite values
      4. Apply to full grid to produce high-resolution map

    Args:
        method: "xgboost" | "random_forest" | "ann"

    Returns:
        {
          downscaled_pixels: [{lat, lon, no2_estimated}],
          model_metrics: {mae, rmse, r2},
          resolution_km: 1.0,
          method_used: str,
        }
    """
    if not ground_data or not satellite_data.get("satellite_pixels"):
        return {"error": "Insufficient data for downscaling"}

    # ── Step 1: Build training dataset ────────────────────────────────────────
    # For each ground station, find the nearest satellite pixel
    training_rows = []
    for station in ground_data:
        # Find nearest coarse pixel
        nearest_no2_col = _find_nearest_pixel(
            station["lat"], station["lon"],
            satellite_data["satellite_pixels"]
        )
        training_rows.append({
            "lat":            station["lat"],
            "lon":            station["lon"],
            "sat_no2_col":    nearest_no2_col,
            "zone_industrial": 1 if station["zone"] == "industrial" else 0,
            "zone_traffic":    1 if station["zone"] == "traffic" else 0,
            "hour":            datetime.utcnow().hour,
            "seasonal_factor": satellite_data["seasonal_factor"],
            "ground_no2":      station["no2_ugm3"],  # TARGET
        })

    df = pd.DataFrame(training_rows)
    feature_cols = ["lat", "lon", "sat_no2_col", "zone_industrial",
                    "zone_traffic", "hour", "seasonal_factor"]
    X = df[feature_cols].values
    y = df["ground_no2"].values

    # ── Step 2: Train ML model ─────────────────────────────────────────────────
    model_fn = {
        "xgboost":       _train_xgboost,
        "random_forest": _train_rf,
        "ann":           _train_ann,
    }.get(method, _train_xgboost)

    model, metrics = model_fn(X, y)

    # ── Step 3: Generate fine-scale grid (1 km) ───────────────────────────────
    pixels = satellite_data["satellite_pixels"]
    center_lat = np.mean([p["lat"] for p in pixels])
    center_lon = np.mean([p["lon"] for p in pixels])

    # 0.009° ≈ 1 km resolution; generate 15x15 = 225 pixels
    fine_grid = []
    step = 0.009
    for i in range(-7, 8):
        for j in range(-7, 8):
            lat = round(center_lat + i * step, 5)
            lon = round(center_lon + j * step, 5)
            nearest_col = _find_nearest_pixel(lat, lon, pixels)

            # Heuristic zone assignment for inference
            is_industrial = 1 if random.random() < 0.2 else 0
            is_traffic     = 1 if random.random() < 0.3 else 0

            X_pred = np.array([[
                lat, lon, nearest_col, is_industrial,
                is_traffic, datetime.utcnow().hour,
                satellite_data["seasonal_factor"]
            ]])
            no2_est = float(model(X_pred))
            no2_est = round(max(0, no2_est + random.gauss(0, 1.5)), 1)

            fine_grid.append({
                "lat":          lat,
                "lon":          lon,
                "no2_ugm3":     no2_est,
                "no2_category": _no2_category(no2_est),
            })

    return {
        "city":              city,
        "downscaled_pixels": fine_grid,
        "pixel_count":       len(fine_grid),
        "resolution_km":     target_resolution_km,
        "method_used":       method,
        "model_metrics":     metrics,
        "coverage_km2":      round(len(fine_grid) * target_resolution_km ** 2, 1),
        "timestamp":         datetime.utcnow().isoformat(),
        "sih_reference":     "SIH1734 — ISRO SAC Downscaling Methodology",
    }


def _find_nearest_pixel(lat: float, lon: float, pixels: list[dict]) -> float:
    """Return NO₂ column density of the closest satellite pixel."""
    if not pixels:
        return 0.001
    distances = [
        ((p["lat"] - lat) ** 2 + (p["lon"] - lon) ** 2, p["no2_column"])
        for p in pixels
    ]
    return min(distances, key=lambda x: x[0])[1]


def _train_xgboost(X, y):
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=42, verbosity=0)
        if len(X) >= 3:
            scores = cross_val_score(model, X, y, cv=min(3, len(X)),
                                     scoring="neg_mean_absolute_error")
            mae = round(-scores.mean(), 2)
        else:
            mae = None
        model.fit(X, y)
        metrics = {"model": "XGBoost", "mae_ugm3": mae, "training_samples": len(X)}
        return model.predict, metrics
    except ImportError:
        return _train_rf(X, y)


def _train_rf(X, y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    from sklearn.metrics import mean_absolute_error
    mae = round(mean_absolute_error(y, model.predict(X)), 2) if len(X) > 1 else None
    metrics = {"model": "RandomForest", "mae_ugm3": mae, "training_samples": len(X)}
    return model.predict, metrics


def _train_ann(X, y):
    """Simple MLP neural network (ANN) as suggested in SIH1734."""
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=42, early_stopping=True)
        model.fit(X_scaled, y)
        from sklearn.metrics import mean_absolute_error
        mae = round(mean_absolute_error(y, model.predict(X_scaled)), 2) if len(X) > 1 else None
        metrics = {"model": "ANN/MLP", "mae_ugm3": mae, "training_samples": len(X)}

        def predict_fn(X_new):
            return model.predict(scaler.transform(X_new))

        return predict_fn, metrics
    except Exception:
        return _train_rf(X, y)


def _no2_category(no2_ugm3: float) -> str:
    """CPCB NO₂ annual standard: 40 µg/m³"""
    if no2_ugm3 < 20:  return "Good"
    if no2_ugm3 < 40:  return "Satisfactory"
    if no2_ugm3 < 80:  return "Moderate"
    if no2_ugm3 < 120: return "Poor"
    if no2_ugm3 < 200: return "Very Poor"
    return "Severe"


# ── Full SIH1734 Pipeline ─────────────────────────────────────────────────────

def run_sih1734_pipeline(city: str, ml_method: str = "xgboost") -> dict:
    """
    End-to-end SIH1734 downscaling pipeline for a city.

    Steps:
      1. Fetch satellite NO₂ (Sentinel-5P / VEDAS simulation)
      2. Fetch CPCB ground station data
      3. Run AI/ML downscaling
      4. Return full result with metrics

    Returns comprehensive dict for API or dashboard display.
    """
    logger.info(f"Running SIH1734 pipeline for {city} using {ml_method}")

    satellite = fetch_satellite_no2(city)
    ground    = fetch_cpcb_ground_data(city)
    downscaled = downscale_no2(city, satellite, ground, method=ml_method)

    # Summary stats on downscaled grid
    if downscaled.get("downscaled_pixels"):
        vals = [p["no2_ugm3"] for p in downscaled["downscaled_pixels"]]
        downscaled["summary"] = {
            "min_no2":  round(min(vals), 1),
            "max_no2":  round(max(vals), 1),
            "mean_no2": round(float(np.mean(vals)), 1),
            "std_no2":  round(float(np.std(vals)), 1),
            "hotspot_count": sum(1 for v in vals if v > 80),
        }

    return {
        "pipeline":   "SIH1734",
        "city":       city,
        "timestamp":  datetime.utcnow().isoformat(),
        "satellite":  satellite,
        "ground_stations": ground,
        "downscaled": downscaled,
        "method":     ml_method,
    }
