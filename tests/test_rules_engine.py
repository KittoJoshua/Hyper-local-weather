"""
Unit Tests for Agronomic Risk Rules Engine.
Validates fertilizer washout, seed desiccation, and fungal blight detection.
"""
from django.test import TestCase
from apps.farms.models import CropProfile
from apps.advisories.rules_engine import AgronomicRulesEngine
from apps.advisories.models import AdvisoryAlert


class RulesEngineTestCase(TestCase):
    def setUp(self):
        self.engine = AgronomicRulesEngine()
        self.maize = CropProfile.objects.create(
            crop_name="Maize",
            optimal_temp_min=15.0,
            optimal_temp_max=28.0,
            drought_threshold_mm=15.0,
            growth_cycle_days=150
        )

    def test_fertilizer_washout_trigger(self):
        # Forecast with heavy rain (> 25mm in 48 hours)
        hourly_heavy_rain = [
            {"hour_label": f"H{i}", "temperature_c": 18.0, "humidity_percent": 85.0, "precipitation_mm": 5.0 if i in (10, 11, 12, 13, 14, 15) else 0.0, "wind_speed_ms": 2.0}
            for i in range(72)
        ]
        forecast_data = {
            "hourly_forecast": hourly_heavy_rain,
            "daily_forecast": []
        }
        alerts = self.engine.evaluate_risks(forecast_data, self.maize)
        alert_types = [a["alert_type"] for a in alerts]
        self.assertIn(AdvisoryAlert.AlertType.FERTILIZER_WASHOUT, alert_types)
        # Should be critical
        washout_alert = [a for a in alerts if a["alert_type"] == AdvisoryAlert.AlertType.FERTILIZER_WASHOUT][0]
        self.assertTrue(washout_alert["is_critical"])

    def test_seed_desiccation_trigger(self):
        # Forecast with high heat (> 30C) and zero rain
        hourly_dry_heat = [
            {"hour_label": f"H{i}", "temperature_c": 32.0, "humidity_percent": 30.0, "precipitation_mm": 0.0, "wind_speed_ms": 3.0}
            for i in range(72)
        ]
        forecast_data = {
            "hourly_forecast": hourly_dry_heat,
            "daily_forecast": []
        }
        alerts = self.engine.evaluate_risks(forecast_data, self.maize)
        alert_types = [a["alert_type"] for a in alerts]
        self.assertIn(AdvisoryAlert.AlertType.SEED_DESICCATION, alert_types)
