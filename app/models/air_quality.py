"""AirQualityReading model — one row per city per fetch."""

from datetime import datetime
from app import db


class AirQualityReading(db.Model):
    __tablename__ = "air_quality"

    id          = db.Column(db.Integer, primary_key=True)
    city_id     = db.Column(db.Integer, db.ForeignKey("cities.id"), nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Pollutant values (µg/m³ unless noted)
    aqi         = db.Column(db.Float)
    pm25        = db.Column(db.Float)   # Fine particulate matter
    pm10        = db.Column(db.Float)   # Coarse particulate matter
    co          = db.Column(db.Float)   # Carbon monoxide (µg/m³)
    no2         = db.Column(db.Float)   # Nitrogen dioxide
    so2         = db.Column(db.Float)   # Sulphur dioxide
    o3          = db.Column(db.Float)   # Ozone
    source      = db.Column(db.String(50), default="OpenAQ")

    # Derived
    category    = db.Column(db.String(50))  # Good / Moderate / etc.

    def compute_category(self):
        """Auto-classify AQI into WHO/CPCB categories."""
        if self.aqi is None:
            return "Unknown"
        aqi = self.aqi
        if aqi <= 50:   return "Good"
        if aqi <= 100:  return "Moderate"
        if aqi <= 150:  return "Unhealthy for Sensitive Groups"
        if aqi <= 200:  return "Unhealthy"
        if aqi <= 300:  return "Very Unhealthy"
        return "Hazardous"

    def save(self):
        self.category = self.compute_category()
        db.session.add(self)
        db.session.commit()

    def to_dict(self):
        return {
            "id":        self.id,
            "city_id":   self.city_id,
            "city_name": self.city.city_name if self.city else None,
            "timestamp": self.timestamp.isoformat(),
            "aqi":       self.aqi,
            "pm25":      self.pm25,
            "pm10":      self.pm10,
            "co":        self.co,
            "no2":       self.no2,
            "so2":       self.so2,
            "o3":        self.o3,
            "category":  self.category,
        }

    def __repr__(self):
        return f"<AQR city={self.city_id} aqi={self.aqi} ts={self.timestamp}>"


class AQIPrediction(db.Model):
    """Stores ML forecasts per city per day."""
    __tablename__ = "aqi_predictions"

    id              = db.Column(db.Integer, primary_key=True)
    city_id         = db.Column(db.Integer, db.ForeignKey("cities.id"), nullable=False, index=True)
    predicted_date  = db.Column(db.Date, nullable=False, index=True)
    predicted_aqi   = db.Column(db.Float, nullable=False)
    model_used      = db.Column(db.String(50))
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "city_id":        self.city_id,
            "predicted_date": self.predicted_date.isoformat(),
            "predicted_aqi":  round(self.predicted_aqi, 1),
            "model_used":     self.model_used,
        }
