"""
Unit and Integration Tests for FarmPlot Spatial Centroid and Registration.
"""
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from apps.farms.models import FarmPlot, CropProfile

User = get_user_model()


class FarmPlotTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="nakuru_farmer",
            phone_number="0722001122",
            password="FarmerPassword123!"
        )
        self.crop = CropProfile.objects.create(
            crop_name="Maize",
            optimal_temp_min=15.0,
            optimal_temp_max=28.0,
            drought_threshold_mm=15.0,
            growth_cycle_days=150
        )
        self.client.force_authenticate(user=self.user)

    def test_centroid_auto_calculation_from_polygon(self):
        boundary = [
            [-0.350, 35.940],
            [-0.350, 35.950],
            [-0.360, 35.950],
            [-0.360, 35.940]
        ]
        plot = FarmPlot.objects.create(
            farmer=self.user,
            name="Polygon Test Shamba",
            current_crop=self.crop,
            boundary_geojson=boundary,
            size_acres=2.0
        )
        # Expected centroid: lat = -0.355, lon = 35.945
        self.assertAlmostEqual(float(plot.centroid_lat), -0.355, places=3)
        self.assertAlmostEqual(float(plot.centroid_lon), 35.945, places=3)

    def test_farmplot_api_creation(self):
        payload = {
            "name": "API Registered Plot",
            "current_crop": self.crop.id,
            "county": "Nakuru",
            "size_acres": 3.5,
            "centroid_lat": -0.354000,
            "centroid_lon": 35.942000,
            "elevation_m": 2180.0
        }
        response = self.client.post("/api/v1/farms/plots/", payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["name"], "API Registered Plot")
        self.assertEqual(response.data["farmer_name"], "nakuru_farmer")
