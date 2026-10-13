"""
Unit and Integration Tests for User Authentication and Role Permissions.
"""
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status

User = get_user_model()


class AuthenticationTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.farmer_data = {
            "username": "test_farmer",
            "phone_number": "0799112233",
            "email": "farmer@test.com",
            "password": "SecurePass123!",
            "role": "FARMER",
            "county": "Nakuru"
        }

    def test_farmer_registration(self):
        response = self.client.post("/api/v1/auth/register/", self.farmer_data, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", response.data)
        self.assertEqual(response.data["user"]["username"], "test_farmer")

        user = User.objects.get(username="test_farmer")
        self.assertTrue(user.is_farmer)
        self.assertEqual(user.phone_number, "0799112233")

    def test_farmer_login_with_phone_number(self):
        User.objects.create_user(**self.farmer_data)
        login_data = {
            "username": "0799112233",
            "password": "SecurePass123!"
        }
        response = self.client.post("/api/v1/auth/login/", login_data, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("token", response.data)

    def test_invalid_login(self):
        response = self.client.post("/api/v1/auth/login/", {"username": "ghost", "password": "bad"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
