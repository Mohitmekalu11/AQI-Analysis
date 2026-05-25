

from flask import Blueprint, jsonify, request
from app.models.city import City
from app.models.air_quality import AirQualityReading, AQIPrediction
from app.services.genai_service import (
    get_health_advice,
    chat_with_aqi_assistant,
    generate_health_report,
    interpret_satellite_no2,
)
from app.services.satellite_downscale import (
    run_sih1734_pipeline,
    fetch_satellite_no2,
    fetch_cpcb_ground_data,
)
from app.services.analytics import get_aqi_trend, get_pollutant_breakdown
from app.services.ml_forecast import get_city_forecast_summary
from datetime import datetime, timedelta
import numpy as np

genai_bp = Blueprint("genai", __name__)


def success(data, status=200):
    return jsonify({"status": "success", "data": data}), status


def error(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# ── POST /api/ai/health-advice ─────────────────────────────────────────────────
@genai_bp.route("/ai/health-advice", methods=["POST"])
def health_advice():
    """
    Get personalized AI health advice based on AQI + user profile.

    Request body (JSON):
    {
      "city": "Delhi",
      "user_profile": {
        "age_group": "adult",         // child | adult | elderly
        "conditions": ["asthma"],      // list of health conditions
        "activity": "outdoor_work"     // outdoor_work | exercise | indoor | commuting
      }
    }
    """
    body = request.get_json() or {}
    city_name = body.get("city")

    if not city_name:
        return error("city is required")

    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    latest = AirQualityReading.query \
        .filter_by(city_id=city.id) \
        .order_by(AirQualityReading.timestamp.desc()) \
        .first()

    if not latest or not latest.aqi:
        return error("No recent AQI data for this city", 404)

    advice = get_health_advice(
        city=city_name,
        aqi=latest.aqi,
        pm25=latest.pm25,
        pm10=latest.pm10,
        no2=latest.no2,
        user_profile=body.get("user_profile"),
    )

    return success(advice)


# ── POST /api/ai/chat ─────────────────────────────────────────────────────────
@genai_bp.route("/ai/chat", methods=["POST"])
def aqi_chat():
    """
    Conversational AQI assistant.

    Request body (JSON):
    {
      "message": "Is it safe to exercise in Delhi today?",
      "city": "Delhi",                   // optional — for context
      "history": [                       // optional — conversation history
        {"role": "user",  "text": "..."},
        {"role": "model", "text": "..."}
      ]
    }
    """
    body = request.get_json() or {}
    message = body.get("message", "").strip()

    if not message:
        return error("message is required")

    context = {}
    city_name = body.get("city")
    if city_name:
        city = City.query.filter_by(city_name=city_name).first()
        if city:
            latest = AirQualityReading.query \
                .filter_by(city_id=city.id) \
                .order_by(AirQualityReading.timestamp.desc()) \
                .first()
            if latest:
                context["city"] = city_name
                context["current_aqi"] = latest.aqi

                # Add 3-day forecast for context
                preds = AQIPrediction.query \
                    .filter_by(city_id=city.id) \
                    .order_by(AQIPrediction.predicted_date) \
                    .limit(3).all()
                if preds:
                    context["forecast"] = [
                        {"date": str(p.predicted_date), "aqi": p.predicted_aqi}
                        for p in preds
                    ]

    result = chat_with_aqi_assistant(
        user_message=message,
        context=context,
        history=body.get("history", []),
    )

    return success(result)


# ── GET /api/ai/health-report/<city> ──────────────────────────────────────────
@genai_bp.route("/ai/health-report/<city_name>", methods=["GET"])
def health_report(city_name):
    """
    Generate a comprehensive AI health report for a city.
    Combines 7-day historical AQI, pollutant averages, forecast, and satellite NO₂.
    """
    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    cutoff = datetime.utcnow() - timedelta(days=7)
    readings = AirQualityReading.query \
        .filter(AirQualityReading.city_id == city.id,
                AirQualityReading.timestamp >= cutoff) \
        .all()

    if not readings:
        return error("Insufficient data for report generation", 404)

    aqis = [r.aqi for r in readings if r.aqi]
    aqi_stats = {
        "min":   round(min(aqis), 1) if aqis else None,
        "max":   round(max(aqis), 1) if aqis else None,
        "avg":   round(float(np.mean(aqis)), 1) if aqis else None,
        "trend": "improving" if len(aqis) > 10 and aqis[-1] < aqis[0] else "worsening"
    }

    def safe_mean(vals):
        clean = [v for v in vals if v is not None and v > 0]
        return round(float(np.mean(clean)), 1) if clean else None

    pollutant_averages = {
        "pm25": safe_mean([r.pm25 for r in readings]),
        "pm10": safe_mean([r.pm10 for r in readings]),
        "no2":  safe_mean([r.no2 for r in readings]),
        "so2":  safe_mean([r.so2 for r in readings]),
        "co":   safe_mean([r.co for r in readings]),
        "o3":   safe_mean([r.o3 for r in readings]),
    }

    preds = AQIPrediction.query \
        .filter_by(city_id=city.id) \
        .order_by(AQIPrediction.predicted_date).all()

    forecast_summary = {}
    if preds:
        pred_aqis = [p.predicted_aqi for p in preds]
        forecast_summary = {
            "avg_aqi":   round(float(np.mean(pred_aqis)), 1),
            "worst_day": str(preds[int(np.argmax(pred_aqis))].predicted_date),
            "best_day":  str(preds[int(np.argmin(pred_aqis))].predicted_date),
        }

    # Get satellite NO₂ for SIH1734 context
    satellite_data = fetch_satellite_no2(city_name)
    satellite_no2 = satellite_data.get("no2_column_density")

    report = generate_health_report(
        city=city_name,
        aqi_stats=aqi_stats,
        pollutant_averages=pollutant_averages,
        forecast_summary=forecast_summary,
        satellite_no2=satellite_no2,
    )

    return success(report)


# ── GET /api/satellite/<city> ──────────────────────────────────────────────────
@genai_bp.route("/satellite/<city_name>", methods=["GET"])
def satellite_data(city_name):
    """
    Fetch raw satellite NO₂ data (Sentinel-5P / VEDAS ISRO) for a city.
    SIH1734 core dataset.
    """
    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    satellite = fetch_satellite_no2(city_name)
    ground = fetch_cpcb_ground_data(city_name)

    return success({
        "city":            city_name,
        "satellite":       satellite,
        "cpcb_stations":   ground,
        "methodology":     "SIH1734 — ISRO SAC",
        "data_note":       (
            "Satellite data is from Sentinel-5P TROPOMI (7km resolution). "
            "Use /api/satellite/{city}/downscale for 1km AI/ML downscaled maps."
        ),
    })


# ── GET /api/satellite/<city>/downscale ────────────────────────────────────────
@genai_bp.route("/satellite/<city_name>/downscale", methods=["GET"])
def downscale_data(city_name):
    """
    Run the full SIH1734 AI/ML downscaling pipeline for a city.
    Produces 1km resolution NO₂ map from 7km satellite data.

    Query params:
      method: xgboost (default) | random_forest | ann
    """
    city = City.query.filter_by(city_name=city_name).first()
    if not city:
        return error(f"City '{city_name}' not found", 404)

    method = request.args.get("method", "xgboost").lower()
    if method not in ("xgboost", "random_forest", "ann"):
        return error("method must be one of: xgboost, random_forest, ann")

    result = run_sih1734_pipeline(city_name, ml_method=method)
    return success(result)


# ── POST /api/satellite/<city>/interpret ──────────────────────────────────────
@genai_bp.route("/satellite/<city_name>/interpret", methods=["POST"])
def interpret_satellite(city_name):
    """
    Use Gemini AI to explain satellite NO₂ data in plain language.

    Request body (JSON):
    {
      "no2_column_density": 0.0023,   // µmol/m² (from satellite endpoint)
      "method": "XGBoost"             // optional — model used
    }
    """
    body = request.get_json() or {}

    # Auto-fetch if not provided
    no2_column = body.get("no2_column_density")
    if no2_column is None:
        satellite = fetch_satellite_no2(city_name)
        no2_column = satellite["no2_column_density"]

    # Get ground NO₂ for comparison
    ground = fetch_cpcb_ground_data(city_name)
    avg_ground_no2 = None
    if ground:
        avg_ground_no2 = round(
            float(np.mean([s["no2_ugm3"] for s in ground])), 1
        )

    interpretation = interpret_satellite_no2(
        city=city_name,
        no2_column_density=no2_column,
        ground_no2=avg_ground_no2,
        downscale_method=body.get("method", "XGBoost"),
    )

    return success({
        "city":              city_name,
        "no2_column_density": no2_column,
        "avg_ground_no2_ugm3": avg_ground_no2,
        "interpretation":    interpretation,
        "methodology":       "SIH1734 — ISRO SAC downscaling + Gemini AI explanation",
    })
