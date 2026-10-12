"""
Integration Tests for Forecast and Advisory REST API Endpoints.
"""
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from apps.farms.models import FarmPlot, CropProfile

User = get_user_model()


class ForecastAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.farmer = User.objects.create_user(
            username="eldoret_farmer",
            phone_number="0733445566",
            password="SecureFarmer123!"
        )
        self.crop = CropProfile.objects.create(
            crop_name="Wheat",
            optimal_temp_min=14.0,
            optimal_temp_max=24.0,
            drought_threshold_mm=12.0,
            growth_cycle_days=120
        )
        self.plot = FarmPlot.objects.create(
            farmer=self.farmer,
            name="Eldoret North Plot",
            current_crop=self.crop,
            county="Uasin Gishu",
            centroid_lat=0.514,
            centroid_lon=35.269,
            elevation_m=2095.0,
            size_acres=5.0
        )
        self.client.force_authenticate(user=self.farmer)

    def test_point_forecast_public_api(self):
        # Public point forecast endpoint
        anon_client = APIClient()
        response = anon_client.get("/api/v1/forecasts/point/?lat=0.514&lon=35.269&elevation_m=2095")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("hourly_forecast", response.data)
        self.assertIn("daily_forecast", response.data)
        self.assertEqual(len(response.data["hourly_forecast"]), 72)
        self.assertEqual(len(response.data["daily_forecast"]), 7)

    def test_plot_forecast_authenticated_api(self):
        response = self.client.get(f"/api/v1/forecasts/plot/{self.plot.id}/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["plot_id"], self.plot.id)
        self.assertEqual(response.data["crop"], "Wheat")
        self.assertIn("hourly_forecast", response.data)

    def test_unauthorized_plot_access(self):
        other_farmer = User.objects.create_user(
            username="other_farmer",
            phone_number="0744556677",
            password="OtherPassword123!"
        )
        self.client.force_authenticate(user=other_farmer)
        response = self.client.get(f"/api/v1/forecasts/plot/{self.plot.id}/")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
