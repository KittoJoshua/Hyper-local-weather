"""
Agronomic Rules Engine for Micro-Climate Risk Assessment.
Evaluates forecasted meteorological trajectories against crop-specific physiological thresholds
to generate timely agricultural advisories.
"""
from typing import Dict, List, Any, Optional
from apps.farms.models import CropProfile
from apps.advisories.models import AdvisoryAlert


class AgronomicRulesEngine:
    """
    Translates continuous numerical forecasts into discrete, farmer-centric alerts.
    Addresses key agricultural failure modes identified in literature:
    1. Seed Desiccation Trap
    2. Fertilizer Washout
    3. Harvest Humidity & Aflatoxin Spoilage
    4. Pest/Disease Infection Windows
    """

    def evaluate_risks(
        self,
        forecast_data: Dict[str, Any],
        crop: Optional[CropProfile] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes hourly (72h) and daily (7d) forecast arrays to identify risk triggers.
        Returns a list of structured alert dictionaries.
        """
        alerts = []
        hourly = forecast_data.get("hourly_forecast", [])
        daily = forecast_data.get("daily_forecast", [])

        if not hourly:
            return alerts

        crop_name = crop.crop_name if crop else "General Crop"

        # 1. Check Fertilizer Washout Risk (High rainfall within next 48 hours)
        rain_next_48h = sum(h.get("precipitation_mm", 0.0) for h in hourly[:48])
        max_hourly_rain = max(h.get("precipitation_mm", 0.0) for h in hourly[:48])

        if rain_next_48h >= 25.0 or max_hourly_rain >= 15.0:
            alerts.append({
                "alert_type": AdvisoryAlert.AlertType.FERTILIZER_WASHOUT,
                "severity": AdvisoryAlert.Severity.CRITICAL,
                "headline": f"Heavy Rainfall Alert: {rain_next_48h:.1f}mm expected in next 48 hours",
                "agronomic_advice": (
                    f"Postpone top-dressing fertilizers (CAN/Urea) on your {crop_name}. "
                    f"Predicted rainfall intensity will cause severe surface runoff and leaching into subsoil. "
                    "Wait until rainfall subsides and topsoil drains."
                ),
                "is_critical": True
            })
        elif rain_next_48h >= 12.0:
            alerts.append({
                "alert_type": AdvisoryAlert.AlertType.FERTILIZER_WASHOUT,
                "severity": AdvisoryAlert.Severity.MODERATE,
                "headline": f"Moderate Rain Approaching: {rain_next_48h:.1f}mm over 48 hours",
                "agronomic_advice": (
                    f"Exercise caution with fertilizer or foliar spray on {crop_name}. "
                    "Ensure adequate soil cover to maximize fertilizer absorption."
                ),
                "is_critical": False
            })

        # 2. Check Seed Desiccation Risk (High temperature with zero or negligible rainfall)
        max_temp_next_72h = max(h.get("temperature_c", 0.0) for h in hourly[:72])
        rain_next_72h = sum(h.get("precipitation_mm", 0.0) for h in hourly[:72])

        if max_temp_next_72h >= 29.5 and rain_next_72h < 2.0:
            alerts.append({
                "alert_type": AdvisoryAlert.AlertType.SEED_DESICCATION,
                "severity": AdvisoryAlert.Severity.HIGH,
                "headline": f"Thermal Spike Warning: Peak {max_temp_next_72h:.1f}°C with dry topsoil",
                "agronomic_advice": (
                    f"High risk of seed desiccation for recién-sown {crop_name}. "
                    "Topsoil temperatures will trigger seedling mortality without supplementary watering. "
                    "Consider mulching planting basins or deferring shallow-seeded crops."
                ),
                "is_critical": True
            })

        # 3. Check Fungal Blight / Stripe Rust Risk (Sustained high humidity > 80% with mild warmth)
        high_humidity_hours = sum(1 for h in hourly[:48] if h.get("humidity_percent", 0.0) >= 80.0)
        avg_temp_48h = sum(h.get("temperature_c", 0.0) for h in hourly[:48]) / 48.0

        if high_humidity_hours >= 20 and (16.0 <= avg_temp_48h <= 25.0):
            alerts.append({
                "alert_type": AdvisoryAlert.AlertType.FUNGAL_BLIGHT,
                "severity": AdvisoryAlert.Severity.MODERATE,
                "headline": "Elevated Fungal Spore & Blight Conditions",
                "agronomic_advice": (
                    f"Prolonged humidity ({high_humidity_hours} hours > 80%) creates favorable conditions "
                    f"for fungal pathogens such as Maize Stripe Rust, Leaf Blight, and Potato Mildew. "
                    "Scout field margins for early lesion symptoms. Prepare preventive fungicide if crops are in vulnerable vegetative stage."
                ),
                "is_critical": False
            })

        # 4. Check Optimal Spraying Window (Low wind < 3.5 m/s, zero rain, daylight hours)
        def _extract_hour(h_entry):
            ts = h_entry.get("timestamp")
            if not ts or not isinstance(ts, str):
                return 12
            if "T" in ts:
                try:
                    return int(ts.split("T")[1][:2])
                except (ValueError, IndexError):
                    pass
            elif " " in ts:
                try:
                    return int(ts.split(" ")[1][:2])
                except (ValueError, IndexError):
                    pass
            return 12

        calm_windows = [
            h for h in hourly[:24]
            if h.get("precipitation_mm", 0.0) == 0.0
            and h.get("wind_speed_ms", 0.0) <= 3.5
            and 6 <= _extract_hour(h) <= 18
        ]
        if calm_windows and not any(a["alert_type"] == AdvisoryAlert.AlertType.FERTILIZER_WASHOUT for a in alerts):
            first_window = calm_windows[0]
            alerts.append({
                "alert_type": AdvisoryAlert.AlertType.OPTIMAL_WINDOW,
                "severity": AdvisoryAlert.Severity.LOW,
                "headline": f"Favorable Application Window around {first_window.get('hour_label')}",
                "agronomic_advice": (
                    f"Calm wind conditions ({first_window.get('wind_speed_ms')} m/s) and zero rainfall "
                    f"provide an ideal window for pesticide, herbicide, or foliar nutrient application on your {crop_name}."
                ),
                "is_critical": False
            })

        return alerts
