"""
REST API Blueprint
==================
All /api/* endpoints. Returns JSON.
Documented with inline comments (Swagger via flask-smorest optional add-on).
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from app import db
from app.models.city import City
from app.models.air_quality import AirQualityReading, AQIPrediction
from app.services.analytics import (
    get_city_ranking, get_aqi_trend,
    get_city_comparison, get_pollutant_breakdown, check_alerts
)
from app.services.ml_forecast import forecast_city_aqi, get_city_forecast_summary

api_bp = Blueprint("api", __name__)


def success(data, status=200):
    return jsonify({"status": "success", "data": data}), status


def error(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# ── GET /api/cities ──────────────────────────────────────────────────────────
@api_bp.route("/cities", methods=["GET"])
def get_cities():
    """List all tracked cities with latest AQI."""
    cities = City.query.filter_by(is_active=True).all()
    result = []
    for city in cities:
        latest = AirQualityReading.query\
            .filter_by(city_id=city.id)\
            .order_by(AirQualityReading.timestamp.desc())\
            .first()

        city_data = city.to_dict()
        city_data["latest_aqi"]      = latest.aqi if latest else None
        city_data["latest_category"] = latest.category if latest else None
        city_data["last_updated"]    = latest.timestamp.isoformat() if latest else None
        result.append(city_data)

    return success(result)


# ── GET /api/aqi?city=Delhi&hours=24 ────────────────────────────────────────
@api_bp.route("/aqi", methods=["GET"])
def get_aqi():
    """Get AQI readings for a city. Optional: hours param (default 24)."""
    city_name = request.args.get("city")
    hours     = int(request.args.get("hours", 24))

    if not city_name:
        return error("city parameter is required")

    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    readings = AirQualityReading.query\
        .filter(AirQualityReading.city_id == city.id,
                AirQualityReading.timestamp >= cutoff)\
        .order_by(AirQualityReading.timestamp.desc())\
        .all()

    return success({
        "city":     city_name,
        "hours":    hours,
        "count":    len(readings),
        "readings": [r.to_dict() for r in readings],
    })


# ── GET /api/aqi/current ─────────────────────────────────────────────────────
@api_bp.route("/aqi/current", methods=["GET"])
def get_current_aqi():
    """Latest AQI reading for every city."""
    cities  = City.query.filter_by(is_active=True).all()
    result  = []
    for city in cities:
        latest = AirQualityReading.query\
            .filter_by(city_id=city.id)\
            .order_by(AirQualityReading.timestamp.desc())\
            .first()
        if latest:
            result.append(latest.to_dict())

    return success(result)


# ── GET /api/ranking ─────────────────────────────────────────────────────────
@api_bp.route("/ranking", methods=["GET"])
def get_ranking():
    """City ranking by average AQI (last 24h)."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    readings = AirQualityReading.query\
        .filter(AirQualityReading.timestamp >= cutoff)\
        .all()

    data = []
    for r in readings:
        d = r.to_dict()
        data.append(d)

    ranking = get_city_ranking(data)
    return success(ranking)


# ── GET /api/trend?city=Delhi&hours=168 ──────────────────────────────────────
@api_bp.route("/trend", methods=["GET"])
def get_trend():
    """AQI trend for a city. hours=168 → 7 days."""
    city_name = request.args.get("city")
    hours     = int(request.args.get("hours", 168))

    if not city_name:
        return error("city parameter is required")

    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    readings = AirQualityReading.query\
        .filter(AirQualityReading.city_id == city.id,
                AirQualityReading.timestamp >= cutoff)\
        .all()

    data = [r.to_dict() for r in readings]
    trend = get_aqi_trend(data, city_name, hours)
    return success(trend)


# ── GET /api/compare?cities=Delhi,Mumbai,Pune ────────────────────────────────
@api_bp.route("/compare", methods=["GET"])
def compare_cities():
    """Side-by-side city comparison chart data."""
    cities_param = request.args.get("cities", "")
    cities = [c.strip() for c in cities_param.split(",") if c.strip()]

    if not cities:
        return error("cities parameter required (comma-separated)")

    cutoff = datetime.utcnow() - timedelta(hours=168)
    readings = AirQualityReading.query\
        .filter(AirQualityReading.timestamp >= cutoff)\
        .all()

    data = [r.to_dict() for r in readings]
    comparison = get_city_comparison(data, cities)
    return success(comparison)


# ── GET /api/predictions?city=Delhi ──────────────────────────────────────────
@api_bp.route("/predictions", methods=["GET"])
def get_predictions():
    """7-day AQI forecast for a city."""
    city_name = request.args.get("city")
    if not city_name:
        return error("city parameter is required")

    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    # Try pre-computed predictions first
    predictions = AQIPrediction.query\
        .filter_by(city_id=city.id)\
        .order_by(AQIPrediction.predicted_date)\
        .all()

    if predictions:
        preds_data = [p.to_dict() for p in predictions]
    else:
        # Compute on-the-fly
        readings = AirQualityReading.query\
            .filter_by(city_id=city.id)\
            .order_by(AirQualityReading.timestamp.desc())\
            .limit(720).all()

        readings_data = [{"timestamp": r.timestamp, "aqi": r.aqi} for r in readings if r.aqi]
        raw_preds = forecast_city_aqi(readings_data, city_name, days=7)
        preds_data = [{"predicted_date": p["predicted_date"].isoformat(),
                       "predicted_aqi": p["predicted_aqi"],
                       "model_used": p["model_used"]} for p in raw_preds]

    summary = get_city_forecast_summary(
        [{"predicted_date": p["predicted_date"] if hasattr(p["predicted_date"], "strftime")
          else __import__("datetime").date.fromisoformat(p["predicted_date"]),
          "predicted_aqi": p["predicted_aqi"]} for p in preds_data]
    )

    return success({"city": city_name, "predictions": preds_data, "summary": summary})


# ── GET /api/pollutants?city=Delhi ───────────────────────────────────────────
@api_bp.route("/pollutants", methods=["GET"])
def get_pollutants():
    """Pollutant breakdown for a city (last 24h averages)."""
    city_name = request.args.get("city")
    if not city_name:
        return error("city parameter is required")

    cutoff = datetime.utcnow() - timedelta(hours=24)
    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    readings = AirQualityReading.query\
        .filter(AirQualityReading.city_id == city.id,
                AirQualityReading.timestamp >= cutoff)\
        .all()

    data = [r.to_dict() for r in readings]
    breakdown = get_pollutant_breakdown(data, city_name)
    return success({"city": city_name, "pollutants": breakdown})


# ── GET /api/alerts ───────────────────────────────────────────────────────────
@api_bp.route("/alerts", methods=["GET"])
def get_alerts():
    """Cities currently exceeding AQI threshold (default 200)."""
    threshold = int(request.args.get("threshold", 200))
    cutoff    = datetime.utcnow() - timedelta(hours=2)
    readings  = AirQualityReading.query\
        .filter(AirQualityReading.timestamp >= cutoff)\
        .all()

    data   = [r.to_dict() for r in readings]
    alerts = check_alerts(data, threshold)
    return success({"threshold": threshold, "alerts": alerts, "count": len(alerts)})


# ── GET /api/heatmap ─────────────────────────────────────────────────────────
@api_bp.route("/heatmap", methods=["GET"])
def get_heatmap():
    """GeoJSON-ready data for Leaflet.js heatmap."""
    cities  = City.query.filter_by(is_active=True).all()
    features = []

    for city in cities:
        latest = AirQualityReading.query\
            .filter_by(city_id=city.id)\
            .order_by(AirQualityReading.timestamp.desc())\
            .first()

        aqi = latest.aqi if latest else 100

        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [city.longitude, city.latitude],
            },
            "properties": {
                "city_name": city.city_name,
                "state":     city.state,
                "aqi":       aqi,
                "category":  latest.category if latest else "Unknown",
                "pm25":      latest.pm25 if latest else None,
                "pm10":      latest.pm10 if latest else None,
                "timestamp": latest.timestamp.isoformat() if latest else None,
            },
        })

    return success({"type": "FeatureCollection", "features": features})


# ── GET /api/stats ────────────────────────────────────────────────────────────
@api_bp.route("/stats", methods=["GET"])
def get_platform_stats():
    """Platform-wide statistics for the hero section."""
    total_readings = AirQualityReading.query.count()
    total_cities   = City.query.filter_by(is_active=True).count()
    latest         = AirQualityReading.query\
        .order_by(AirQualityReading.timestamp.desc()).first()

    return success({
        "total_readings": total_readings,
        "total_cities":   total_cities,
        "last_updated":   latest.timestamp.isoformat() if latest else None,
        "data_points_processed": f"{total_readings:,}",
    })
