"""
Admin configuration for FarmPlot and CropProfile models.
"""
from django.contrib import admin
from apps.farms.models import FarmPlot, CropProfile


@admin.register(CropProfile)
class CropProfileAdmin(admin.ModelAdmin):
    list_display = ("crop_name", "optimal_temp_min", "optimal_temp_max", "drought_threshold_mm", "growth_cycle_days")
    search_fields = ("crop_name",)


@admin.register(FarmPlot)
class FarmPlotAdmin(admin.ModelAdmin):
    list_display = ("name", "farmer", "current_crop", "county", "size_acres", "centroid_lat", "centroid_lon", "elevation_m")
    list_filter = ("county", "current_crop")
    search_fields = ("name", "farmer__username", "county")
