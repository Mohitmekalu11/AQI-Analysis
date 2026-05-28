import logging
from app import create_app, db
from app.models.city import City, INDIAN_CITIES
from app.models.air_quality import AirQualityReading, AQIPrediction
from dotenv import load_dotenv
import os

load_dotenv()

print("GROQ_API_KEY =", repr(os.getenv("GROQ_API_KEY")))

for k, v in os.environ.items():
    if "GROQ" in k.upper() or "XAI" in k.upper():
        print(k, "=", repr(v))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = create_app()


# =========================
# DATABASE SEED FUNCTION
# =========================
def perform_seed():
    from app.services.fetcher import generate_historical_data

    db.create_all()

    # Seed cities
    for city_data in INDIAN_CITIES:
        if not City.query.filter_by(city_name=city_data["city_name"]).first():
            city = City(**city_data)
            db.session.add(city)
            print(f"  + Added city: {city_data['city_name']}")

    db.session.commit()

    # Seed historical AQI data
    cities = City.query.all()

    for city in cities:
        existing = AirQualityReading.query.filter_by(city_id=city.id).count()

        if existing < 10:
            print(f"Generating historical data for {city.city_name}...")

            readings = generate_historical_data(city.city_name, days=30)

            for r_data in readings[::2]:
                reading = AirQualityReading(
                    city_id=city.id,
                    aqi=r_data["aqi"],
                    pm25=r_data["pm25"],
                    pm10=r_data["pm10"],
                    co=r_data["co"],
                    no2=r_data["no2"],
                    so2=r_data["so2"],
                    o3=r_data["o3"],
                    source=r_data["source"],
                    timestamp=r_data["timestamp"],
                )

                reading.category = reading.compute_category()
                db.session.add(reading)

            db.session.commit()

            count = AirQualityReading.query.filter_by(city_id=city.id).count()

            print(f"→ {count} readings stored for {city.city_name}")

    total = AirQualityReading.query.count()

    print(f"\n✓ Database seeded successfully.")
    print(f"✓ Total readings: {total:,}")


# =========================
# CLI COMMANDS
# =========================
@app.cli.command("seed-db")
def seed_db():
    """
    Run:
        flask seed-db
    """
    with app.app_context():
        perform_seed()


@app.cli.command("fetch-now")
def fetch_now():
    """
    Run:
        flask fetch-now
    """
    from app.services.fetcher import fetch_city_aqi

    with app.app_context():
        cities = City.query.filter_by(is_active=True).all()

        for city in cities:
            data = fetch_city_aqi(city.city_name)

            if data:
                reading = AirQualityReading(
                    city_id=city.id,
                    **{
                        k: data.get(k)
                        for k in [
                            "aqi",
                            "pm25",
                            "pm10",
                            "co",
                            "no2",
                            "so2",
                            "o3",
                            "source",
                            "timestamp",
                        ]
                    }
                )

                reading.category = reading.compute_category()

                db.session.add(reading)

                print(
                    f"{city.city_name}: AQI {data.get('aqi', '—')} [{data.get('source')}]"
                )

        db.session.commit()


@app.cli.command("train-models")
def train_models():
    """
    Run:
        flask train-models
    """
    from app.services.ml_forecast import forecast_city_aqi

    with app.app_context():
        cities = City.query.filter_by(is_active=True).all()

        for city in cities:
            readings = (
                AirQualityReading.query
                .filter_by(city_id=city.id)
                .order_by(AirQualityReading.timestamp.desc())
                .limit(720)
                .all()
            )

            if len(readings) < 50:
                print(f"{city.city_name}: insufficient data ({len(readings)} readings)")
                continue

            data = [
                {"timestamp": r.timestamp, "aqi": r.aqi}
                for r in readings
                if r.aqi
            ]

            preds = forecast_city_aqi(data, city.city_name)

            print(f"{city.city_name}: forecast for next 7 days:")

            for p in preds:
                print(
                    f"  {p['predicted_date']} → AQI {p['predicted_aqi']} [{p['model_used']}]"
                )



# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":

    # Start background scheduler
    from app.services.scheduler import start_scheduler

    start_scheduler(app)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False
    )