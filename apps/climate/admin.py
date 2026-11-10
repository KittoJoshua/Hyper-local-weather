"""
Admin configuration for climate and meteorological models.
"""
from django.contrib import admin
from apps.climate.models import MeteorologicalStation, ClimateRecord


@admin.register(MeteorologicalStation)
class MeteorologicalStationAdmin(admin.ModelAdmin):
    list_display = ("station_name", "county", "elevation_m", "latitude", "longitude", "is_active")
    list_filter = ("county", "is_active")


@admin.register(ClimateRecord)
class ClimateRecordAdmin(admin.ModelAdmin):
    list_display = ("recorded_at", "station", "farmplot", "temperature_c", "humidity_percent", "precipitation_mm")
    list_filter = ("station", "farmplot__county", "recorded_at")
    date_hierarchy = "recorded_at"
    list_per_page = 50
