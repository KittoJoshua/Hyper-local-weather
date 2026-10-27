"""
Model Registry and Forecast Log Entities for MLOps Tracking.

Maintains end-to-end MLOps lifecycle governance:
- ModelRegistry: Versioned model artifacts, algorithm types (LSTM, XGBoost, Hybrid),
                 hyperparameter payloads, and walk-forward validation benchmark metrics (RMSE, MAE, R²).
- ForecastLog: Historical audit trail of downscaled predictions produced per farm plot
               across 72-hour and 7-day forecast horizons.
"""
from django.db import models
from apps.farms.models import FarmPlot


class ModelRegistry(models.Model):
    """
    Registry of trained Machine Learning models with cross-validation scores.
    Used by the Admin ML Control Panel and inference engine for active model routing.
    """
    model_name = models.CharField(max_length=100)
    version_tag = models.CharField(max_length=50, unique=True)
    algorithm = models.CharField(max_length=100, default="Hybrid (LSTM + XGBoost)")
    hyperparameters = models.JSONField(default=dict)
    rmse_score = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    mae_score = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    r2_score = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    is_active = models.BooleanField(default=False)
    weights_path = models.CharField(max_length=255, blank=True, null=True)
    deployed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-deployed_at"]

    def __str__(self):
        return f"{self.model_name} [{self.version_tag}] (R2: {self.r2_score})"


class ForecastLog(models.Model):
    """
    Stores persisted downscaled micro-climate predictions anchored to a specific farm plot.
    """
    farmplot = models.ForeignKey(FarmPlot, on_delete=models.CASCADE, related_name="forecast_logs")
    model = models.ForeignKey(ModelRegistry, on_delete=models.SET_NULL, null=True, blank=True, related_name="logs")
    target_timestamp = models.DateTimeField(db_index=True)
    
    # Continuous Predicted Target Variables
    predicted_temp = models.DecimalField(max_digits=5, decimal_places=2)
    predicted_precip = models.DecimalField(max_digits=6, decimal_places=2)
    predicted_humidity = models.DecimalField(max_digits=5, decimal_places=2)
    confidence_interval = models.DecimalField(max_digits=5, decimal_places=2, default=90.00)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["target_timestamp"]
        indexes = [
            models.Index(fields=["farmplot", "target_timestamp"]),
        ]

    def __str__(self):
        return f"Forecast for {self.farmplot.name} at {self.target_timestamp.strftime('%Y-%m-%d %H:%M')}: {self.predicted_temp}C, {self.predicted_precip}mm"
