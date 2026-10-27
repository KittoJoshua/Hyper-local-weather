"""
Farm Plot and Crop Profile Database Models with Spatial Coordinates and Elevation.

Provides core spatial and agronomic primitives:
- CropProfile: Physiological temperature/rainfall thresholds for Kenyan crops.
- FarmPlot: Smallholder parcel spatial centroids, GeoJSON polygons, and terrain elevation
            used for topographical lapse rate downscaling in the ML engine.
"""
from django.db import models
from django.conf import settings
import json


class CropProfile(models.Model):
    """
    Agronomic metadata defining physiological tolerances for staple Kenyan crops.
    Used by the AgronomicRulesEngine to generate targeted advisory alerts (e.g. washout, desiccation).
    """
    crop_name = models.CharField(max_length=100, unique=True)
    optimal_temp_min = models.DecimalField(max_digits=5, decimal_places=2, help_text="Minimum optimal temperature (C)")
    optimal_temp_max = models.DecimalField(max_digits=5, decimal_places=2, help_text="Maximum optimal temperature (C)")
    drought_threshold_mm = models.DecimalField(max_digits=5, decimal_places=2, help_text="Critical minimum rainfall (mm/week)")
    growth_cycle_days = models.PositiveIntegerField(default=120, help_text="Typical lifecycle in days")
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.crop_name


class FarmPlot(models.Model):
    """
    Physical agricultural plot operated by a smallholder farmer.
    Anchors micro-climate predictions to a precise spatial polygon or coordinate centroid.
    """
    farmer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="plots")
    current_crop = models.ForeignKey(CropProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name="plots")
    name = models.CharField(max_length=150, help_text="Farmer's label for the parcel (e.g., North Hill Plot)")
    county = models.CharField(max_length=100, default="Nakuru")
    sub_county = models.CharField(max_length=100, blank=True, null=True)
    ward = models.CharField(max_length=100, blank=True, null=True)
    size_acres = models.DecimalField(max_digits=6, decimal_places=2, default=1.00)
    
    # Geographic Centroid & Elevation
    centroid_lat = models.DecimalField(max_digits=9, decimal_places=6, help_text="Latitude centroid (-90 to +90)")
    centroid_lon = models.DecimalField(max_digits=9, decimal_places=6, help_text="Longitude centroid (-180 to +180)")
    elevation_m = models.DecimalField(max_digits=7, decimal_places=2, default=1950.00, help_text="Terrain altitude in meters")
    
    # Boundary vertices stored as GeoJSON polygon / point array
    boundary_geojson = models.JSONField(blank=True, null=True, help_text="List of [lat, lon] coordinates defining polygon")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} - {self.farmer.username} ({self.county})"

    def calculate_spatial_centroid(self):
        """
        Computes the geometric centroid from boundary_geojson vertices if provided.
        """
        if self.boundary_geojson and isinstance(self.boundary_geojson, list) and len(self.boundary_geojson) > 0:
            lats = [p[0] for p in self.boundary_geojson]
            lons = [p[1] for p in self.boundary_geojson]
            self.centroid_lat = round(sum(lats) / len(lats), 6)
            self.centroid_lon = round(sum(lons) / len(lons), 6)
            return self.centroid_lat, self.centroid_lon
        return self.centroid_lat, self.centroid_lon

    def save(self, *args, **kwargs):
        if self.boundary_geojson and (not self.centroid_lat or not self.centroid_lon):
            self.calculate_spatial_centroid()
        super().save(*args, **kwargs)
