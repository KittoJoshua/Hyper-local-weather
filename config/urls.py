"""
Main URL Configuration for Hyper-Local Micro-Climate Platform.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Farmer Dashboard and Admin Panel
    path('', include('apps.forecasting.urls_ui')),
    
    # User Authentication & Accounts
    path('accounts/', include('apps.accounts.urls_ui')),
    path('login/', RedirectView.as_view(pattern_name='login', permanent=False)),
    path('register/', RedirectView.as_view(pattern_name='register', permanent=False)),
    
    # Farm Plot GIS Management
    path('farms/', include('apps.farms.urls_ui')),
    
    # REST API v1
    path('api/v1/auth/', include('apps.accounts.urls_api')),
    path('api/v1/farms/', include('apps.farms.urls_api')),
    path('api/v1/climate/', include('apps.climate.urls_api')),
    path('api/v1/forecasts/', include('apps.forecasting.urls_api')),
    path('api/v1/advisories/', include('apps.advisories.urls_api')),
]
