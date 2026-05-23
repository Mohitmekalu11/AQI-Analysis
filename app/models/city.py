"""City model — stores Indian city metadata."""

from app import db


class City(db.Model):
    __tablename__ = "cities"

    id         = db.Column(db.Integer, primary_key=True)
    city_name  = db.Column(db.String(100), nullable=False, unique=True)
    state      = db.Column(db.String(100), nullable=False)
    latitude   = db.Column(db.Float, nullable=False)
    longitude  = db.Column(db.Float, nullable=False)
    is_active  = db.Column(db.Boolean, default=True)

    # Relationship — one city has many readings
    readings   = db.relationship("AirQualityReading", backref="city", lazy="dynamic")

    def to_dict(self):
        return {
            "id":        self.id,
            "city_name": self.city_name,
            "state":     self.state,
            "latitude":  self.latitude,
            "longitude": self.longitude,
        }

    def __repr__(self):
        return f"<City {self.city_name}>"


# ── Seed data ─────────────────────────────────────────────────────────────────
INDIAN_CITIES = [
    {"city_name": "Delhi",     "state": "Delhi",             "latitude": 28.6139, "longitude": 77.2090},
    {"city_name": "Mumbai",    "state": "Maharashtra",       "latitude": 19.0760, "longitude": 72.8777},
    {"city_name": "Nagpur",    "state": "Maharashtra",       "latitude": 21.1458, "longitude": 79.0882},
    {"city_name": "Pune",      "state": "Maharashtra",       "latitude": 18.5204, "longitude": 73.8567},
    {"city_name": "Bengaluru", "state": "Karnataka",         "latitude": 12.9716, "longitude": 77.5946},
    {"city_name": "Chennai",   "state": "Tamil Nadu",        "latitude": 13.0827, "longitude": 80.2707},
    {"city_name": "Hyderabad", "state": "Telangana",         "latitude": 17.3850, "longitude": 78.4867},
    {"city_name": "Kolkata",   "state": "West Bengal",       "latitude": 22.5726, "longitude": 88.3639},
    {"city_name": "Ahmedabad", "state": "Gujarat",           "latitude": 23.0225, "longitude": 72.5714},
    {"city_name": "Jaipur",    "state": "Rajasthan",         "latitude": 26.9124, "longitude": 75.7873},
]
