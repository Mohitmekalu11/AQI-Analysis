
from flask import Blueprint, render_template
from app.models.city import City
from app.models.air_quality import AirQualityReading

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def home():
    cities = City.query.filter_by(is_active=True).all()
    city_cards = []
    for city in cities:
        latest = AirQualityReading.query\
            .filter_by(city_id=city.id)\
            .order_by(AirQualityReading.timestamp.desc())\
            .first()
        city_cards.append({
            "city": city,
            "latest": latest,
        })
    return render_template("dashboard/home.html", city_cards=city_cards)


@dashboard_bp.route("/city/<city_name>")
def city_detail(city_name):
    city = City.query.filter_by(city_name=city_name).first_or_404()
    latest = AirQualityReading.query\
        .filter_by(city_id=city.id)\
        .order_by(AirQualityReading.timestamp.desc())\
        .first()
    return render_template("dashboard/city.html", city=city, latest=latest)


@dashboard_bp.route("/compare")
def compare():
    cities = City.query.filter_by(is_active=True).all()
    return render_template("dashboard/compare.html", cities=cities)


@dashboard_bp.route("/heatmap")
def heatmap():
    return render_template("dashboard/heatmap.html")


@dashboard_bp.route("/forecast")
def forecast():
    cities = City.query.filter_by(is_active=True).all()
    return render_template("dashboard/forecast.html", cities=cities)
