"""
Agronomic Advisory Alert Database Models.

Stores machine-generated actionable intelligence triggered by downscaled micro-climate thresholds:
- Fertilizer Washout: Detects heavy convective precipitation (> 20mm/48h) to protect top-dressing inputs.
- Seed Desiccation: Warns against high heat (> 29.5C) with dry topsoil during sowing windows.
- Fungal Pathogens: Monitors prolonged relative humidity (> 80%) for rust, blight, and aflatoxin hazards.
- Optimal Spraying: Identifies low-wind (< 3.5 m/s), rain-free windows for chemical efficacy.
"""
from django.db import models
from apps.farms.models import FarmPlot
from apps.forecasting.models import ForecastLog


class AdvisoryAlert(models.Model):
    """
    Actionable agricultural advisory triggered by forecasted micro-climatic thresholds.
    Links directly to smallholder farm plots and underlying forecast timelines.
    """
    class AlertType(models.TextChoices):
        FERTILIZER_WASHOUT = "FERTILIZER_WASHOUT", "Fertilizer Leaching / Washout Risk"
        SEED_DESICCATION = "SEED_DESICCATION", "Seed Desiccation / Topsoil Drought"
        FUNGAL_BLIGHT = "FUNGAL_BLIGHT", "Fungal Blight / Aflatoxin Hazard"
        DRY_SPELL = "DRY_SPELL", "Prolonged Dry Spell"
        HEAT_STRESS = "HEAT_STRESS", "Thermal Crop Stress"
        OPTIMAL_WINDOW = "OPTIMAL_WINDOW", "Optimal Input / Spraying Window"

    class Severity(models.TextChoices):
        LOW = "LOW", "Advisory Notice"
        MODERATE = "MODERATE", "Moderate Precaution"
        HIGH = "HIGH", "High Risk"
        CRITICAL = "CRITICAL", "Immediate Action Required"

    farmplot = models.ForeignKey(FarmPlot, on_delete=models.CASCADE, related_name="advisories")
    forecast_log = models.ForeignKey(ForecastLog, on_delete=models.SET_NULL, null=True, blank=True, related_name="alerts")
    alert_type = models.CharField(max_length=50, choices=AlertType.choices)
    severity = models.CharField(max_length=20, choices=Severity.choices, default=Severity.MODERATE)
    headline = models.CharField(max_length=200)
    agronomic_advice = models.TextField(help_text="Plain-language actionable guidance for the farmer")
    is_critical = models.BooleanField(default=False)
    is_acknowledged = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.severity}] {self.headline} ({self.farmplot.name})"
