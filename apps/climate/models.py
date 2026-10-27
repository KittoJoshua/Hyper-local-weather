"""
Meteorological Station and Chronological Climate Records Models.

Engineered for high-throughput time-series telemetry and TimescaleDB hypertable partitioning:
- MeteorologicalStation: Regional ground observation nodes across Kenyan agro-climatic zones.
- ClimateRecord: Hourly meteorological records partitioned on `recorded_at` with composite indices
  for high-performance historical range queries and ML training set extraction.
"""
from django.db import models
from apps.farms.models import FarmPlot


class MeteorologicalStation(models.Model):
    """
    Physical or virtual meteorological observation node in Kenya.
    Represents regional anchor points for micro-climate calibration and downscaling.
    """
    station_name = models.CharField(max_length=150, unique=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    elevation_m = models.DecimalField(max_digits=7, decimal_places=2, default=1800.0)
    county = models.CharField(max_length=100, default="Nakuru")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.station_name} ({self.county}) - {self.elevation_m}m"


class ClimateRecord(models.Model):
    """
    Time-series observation entity.
    Stores continuous atmospheric telemetry partitioned into TimescaleDB hypertables.
    """
    farmplot = models.ForeignKey(FarmPlot, on_delete=models.CASCADE, null=True, blank=True, related_name="climate_records")
    station = models.ForeignKey(MeteorologicalStation, on_delete=models.SET_NULL, null=True, blank=True, related_name="records")
    recorded_at = models.DateTimeField(db_index=True)
    
    # Continuous Atmospheric Parameters
    temperature_c = models.DecimalField(max_digits=5, decimal_places=2, help_text="Ambient temperature in Celsius")
    humidity_percent = models.DecimalField(max_digits=5, decimal_places=2, help_text="Relative humidity percentage")
    precipitation_mm = models.DecimalField(max_digits=6, decimal_places=2, default=0.00, help_text="Rainfall volume in mm")
    pressure_hpa = models.DecimalField(max_digits=6, decimal_places=2, default=1013.25, help_text="Barometric pressure in hPa")
    wind_speed_ms = models.DecimalField(max_digits=5, decimal_places=2, default=2.00, help_text="Wind speed in meters per second")

    class Meta:
        ordering = ["-recorded_at"]
        indexes = [
            models.Index(fields=["farmplot", "recorded_at"]),
            models.Index(fields=["station", "recorded_at"]),
        ]

    def __str__(self):
        source = self.farmplot.name if self.farmplot else (self.station.station_name if self.station else "Station")
        return f"[{self.recorded_at.strftime('%Y-%m-%d %H:%M')}] {source}: {self.temperature_c}C, {self.precipitation_mm}mm"
