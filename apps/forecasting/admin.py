"""
Admin configuration for ModelRegistry and ForecastLog models.
"""
from django.contrib import admin
from apps.forecasting.models import ModelRegistry, ForecastLog


@admin.register(ModelRegistry)
class ModelRegistryAdmin(admin.ModelAdmin):
    list_display = ("model_name", "version_tag", "algorithm", "rmse_score", "mae_score", "r2_score", "is_active", "deployed_at")
    list_filter = ("algorithm", "is_active")


@admin.register(ForecastLog)
class ForecastLogAdmin(admin.ModelAdmin):
    list_display = ("farmplot", "target_timestamp", "predicted_temp", "predicted_precip", "predicted_humidity", "confidence_interval")
    list_filter = ("farmplot__county", "target_timestamp")
    date_hierarchy = "target_timestamp"
