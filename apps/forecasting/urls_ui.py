"""
UI route definitions for dashboard and admin panels.
"""
from django.urls import path
from apps.forecasting.views_ui import dashboard_view, admin_ml_panel

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('admin-panel/ml/', admin_ml_panel, name='admin_ml'),
]
