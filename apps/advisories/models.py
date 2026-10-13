"""
Advisory Alert Database Models.
Stores machine-generated actionable agronomic intelligence and input risk warnings.
"""
from django.db import models
from apps.farms.models import FarmPlot
from apps.forecasting.models import ForecastLog


class AdvisoryAlert(models.Model):
    """
    Actionable agricultural advisory triggered by forecasted micro-climatic thresholds.
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
