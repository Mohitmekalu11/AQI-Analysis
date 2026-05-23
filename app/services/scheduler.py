

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone="UTC")


def fetch_all_cities(app):
 
    with app.app_context():
        from app import db
        from app.models.city import City
        from app.models.air_quality import AirQualityReading
        from app.services.fetcher import fetch_city_aqi

        cities = City.query.filter_by(is_active=True).all()
        success, failed = 0, 0

        for city in cities:
            try:
                data = fetch_city_aqi(city.city_name)
                if data:
                    reading = AirQualityReading(
                        city_id=city.id,
                        aqi=data.get("aqi"),
                        pm25=data.get("pm25"),
                        pm10=data.get("pm10"),
                        co=data.get("co"),
                        no2=data.get("no2"),
                        so2=data.get("so2"),
                        o3=data.get("o3"),
                        source=data.get("source", "OpenAQ"),
                        timestamp=data.get("timestamp", datetime.utcnow()),
                    )
                    reading.category = reading.compute_category()
                    db.session.add(reading)
                    success += 1
            except Exception as e:
                logger.error(f"Failed to fetch/store {city.city_name}: {e}")
                failed += 1

        db.session.commit()
        logger.info(f"Fetch complete: {success} success, {failed} failed at {datetime.utcnow()}")


def retrain_models(app):
   
    with app.app_context():
        from app import db
        from app.models.city import City
        from app.models.air_quality import AirQualityReading, AQIPrediction
        from app.services.ml_forecast import forecast_city_aqi

        cities = City.query.filter_by(is_active=True).all()

        for city in cities:
            try:
                readings = AirQualityReading.query\
                    .filter_by(city_id=city.id)\
                    .order_by(AirQualityReading.timestamp.desc())\
                    .limit(720).all()  # Last 30 days hourly

                if len(readings) < 20:
                    continue

                readings_data = [{"timestamp": r.timestamp, "aqi": r.aqi} for r in readings if r.aqi]
                predictions = forecast_city_aqi(readings_data, city.city_name, days=7)

                # Clear old predictions for this city
                AQIPrediction.query.filter_by(city_id=city.id).delete()

                for pred in predictions:
                    p = AQIPrediction(
                        city_id=city.id,
                        predicted_date=pred["predicted_date"],
                        predicted_aqi=pred["predicted_aqi"],
                        model_used=pred["model_used"],
                    )
                    db.session.add(p)

                db.session.commit()
                logger.info(f"Predictions saved for {city.city_name}")

            except Exception as e:
                logger.error(f"Forecasting failed for {city.city_name}: {e}")
                db.session.rollback()


def start_scheduler(app):

    import os
    interval = int(os.getenv("FETCH_INTERVAL_MINUTES", 30))

    # Fetch AQI every 30 minutes
    scheduler.add_job(
        func=fetch_all_cities,
        args=[app],
        trigger=IntervalTrigger(minutes=interval),
        id="fetch_aqi",
        name="Fetch AQI Data",
        replace_existing=True,
    )

    # Retrain ML models daily at midnight UTC
    scheduler.add_job(
        func=retrain_models,
        args=[app],
        trigger=CronTrigger(hour=0, minute=0),
        id="retrain_ml",
        name="Retrain ML Models",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(f"Scheduler started. Fetching every {interval} minutes.")
    return scheduler
