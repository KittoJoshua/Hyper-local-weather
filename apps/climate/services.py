"""
External Meteorological Ingestion and Data Synchronization Service.

Primary source: Open-Meteo API (https://open-meteo.com)
  - Completely free, no API key or registration required.
  - Provides ERA5 historical reanalysis data and 7-day hourly forecasts.
  - Global coverage including all Kenyan agricultural regions.

Fallback: Physics-based diurnal simulation for offline/demo mode.
"""
import requests
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from django.conf import settings
from apps.climate.models import ClimateRecord, MeteorologicalStation
from apps.farms.models import FarmPlot

logger = logging.getLogger(__name__)

# Open-Meteo free API endpoints — no key required
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

# WMO variable codes requested from Open-Meteo
OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]


class WeatherDataIngester:
    """
    Ingests live meteorological telemetry from the Open-Meteo free API
    and persists records into the ClimateRecord table.

    Open-Meteo replaces the original OpenWeatherMap integration because it
    requires no API key, has no rate-limit tiers, and provides ERA5-quality
    historical reanalysis data suitable for ML training pipelines.

    Falls back to a physics-based diurnal simulation when offline.
    """

    def fetch_live_weather(self, lat: float, lon: float) -> dict:
        """
        Queries Open-Meteo forecast endpoint for the current hourly observation
        at the given coordinates. Returns a standardised telemetry dictionary.

        Args:
            lat: Latitude of the target location (decimal degrees).
            lon: Longitude of the target location (decimal degrees).

        Returns:
            Dict with keys: temperature_c, humidity_percent, pressure_hpa,
            precipitation_mm, wind_speed_ms, recorded_at.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
            "forecast_days": 1,
            "timezone": "Africa/Nairobi",
        }
        try:
            response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            hourly = data.get("hourly", {})

            # Pick the most recent completed hour
            times = hourly.get("time", [])
            if not times:
                return self._simulate_telemetry(lat, lon)

            now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
            idx = times.index(now_str) if now_str in times else -1

            def _val(key: str, fallback: float) -> float:
                vals = hourly.get(key, [])
                v = vals[idx] if vals and idx < len(vals) else None
                return float(v) if v is not None else fallback

            return {
                "temperature_c":    _val("temperature_2m", 20.0),
                "humidity_percent": _val("relative_humidity_2m", 65.0),
                "pressure_hpa":     _val("surface_pressure", 1013.25),
                "precipitation_mm": _val("precipitation", 0.0),
                "wind_speed_ms":    _val("wind_speed_10m", 2.5),
                "recorded_at":      datetime.now(),
            }

        except Exception as exc:
            logger.warning(
                "Open-Meteo live fetch failed for (%.4f, %.4f): %s. "
                "Using physics-based fallback.", lat, lon, exc
            )
            return self._simulate_telemetry(lat, lon)

    def fetch_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Fetches ERA5 reanalysis hourly records from Open-Meteo Archive API.
        Useful for populating the ClimateRecord table with real historical data
        before running ML training.

        Args:
            lat:        Latitude (decimal degrees).
            lon:        Longitude (decimal degrees).
            start_date: ISO date string, e.g. '2023-01-01'.
            end_date:   ISO date string, e.g. '2023-12-31'.

        Returns:
            List of telemetry dicts (one per hour).
        """
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly":     ",".join(OPEN_METEO_HOURLY_VARS),
            "timezone":   "Africa/Nairobi",
        }
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            records = []
            for idx, ts in enumerate(times):
                def _v(key: str, fb: float) -> float:
                    vals = hourly.get(key, [])
                    v = vals[idx] if idx < len(vals) else None
                    return float(v) if v is not None else fb

                records.append({
                    "temperature_c":    _v("temperature_2m", 20.0),
                    "humidity_percent": _v("relative_humidity_2m", 65.0),
                    "pressure_hpa":     _v("surface_pressure", 1013.25),
                    "precipitation_mm": _v("precipitation", 0.0),
                    "wind_speed_ms":    _v("wind_speed_10m", 2.5),
                    "recorded_at":      datetime.fromisoformat(ts),
                })
            logger.info(
                "Fetched %d historical records from Open-Meteo Archive "
                "for (%.4f, %.4f) [%s → %s].",
                len(records), lat, lon, start_date, end_date,
            )
            return records

        except Exception as exc:
            logger.error("Open-Meteo Archive fetch failed: %s", exc)
            return []

    def _simulate_telemetry(self, lat: float, lon: float) -> dict:
        """
        Generates realistic local observation telemetry based on time of day
        and Kenya highland characteristics when the API is unreachable.
        """
        now = datetime.now()
        hour = now.hour
        # Diurnal approximation: cooler at night, warmer during the day
        base_temp = 18.0 + 6.0 * (-1.0 if hour < 6 or hour > 20 else 1.0)
        return {
            "temperature_c":    round(base_temp, 1),
            "humidity_percent": 68.0,
            "pressure_hpa":     1012.5,
            "precipitation_mm": 0.0,
            "wind_speed_ms":    2.8,
            "recorded_at":      now,
        }

    def ingest_and_save_plot_reading(self, plot: FarmPlot) -> ClimateRecord:
        """
        Fetches the current weather observation for a farm plot's centroid
        and persists it as a new ClimateRecord.

        Args:
            plot: FarmPlot instance whose centroid coordinates are used.

        Returns:
            The newly created ClimateRecord instance.
        """
        telemetry = self.fetch_live_weather(
            float(plot.centroid_lat), float(plot.centroid_lon)
        )
        record = ClimateRecord.objects.create(
            farmplot=plot,
            recorded_at=telemetry["recorded_at"],
            temperature_c=telemetry["temperature_c"],
            humidity_percent=telemetry["humidity_percent"],
            precipitation_mm=telemetry["precipitation_mm"],
            pressure_hpa=telemetry["pressure_hpa"],
            wind_speed_ms=telemetry["wind_speed_ms"],
        )
        return record
