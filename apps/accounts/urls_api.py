"""
API route definitions for accounts.
"""
from django.urls import path
from apps.accounts.views_api import RegisterAPIView, LoginAPIView, ProfileAPIView

urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='api_register'),
    path('login/', LoginAPIView.as_view(), name='api_login'),
    path('profile/', ProfileAPIView.as_view(), name='api_profile'),
]
