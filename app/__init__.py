"""
AQI Platform — Application Factory
===================================
Flask app using the App Factory pattern for scalability.
Updated: SIH1734 GenAI features + Satellite downscaling routes registered.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_name="development"):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # ── Config ────────────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL", "sqlite:///aqi_dev.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Register Blueprints ───────────────────────────────────────────────────
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.genai_routes import genai_bp        # ← NEW: GenAI + Satellite

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(genai_bp, url_prefix="/api") # ← NEW
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")

    # ── Shell context ─────────────────────────────────────────────────────────
    @app.shell_context_processor
    def make_shell_context():
        from app.models.city import City
        from app.models.air_quality import AirQualityReading
        return {"db": db, "City": City, "AQR": AirQualityReading}

    return app
