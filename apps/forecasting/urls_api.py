"""
API route definitions for forecasting.
"""
from django.urls import path
from rest_framework.routers import DefaultRouter
from apps.forecasting.views_api import (
    PlotForecastAPIView, PointForecastAPIView,
    ModelRegistryViewSet, TriggerRetrainAPIView
)

router = DefaultRouter()
router.register(r'models', ModelRegistryViewSet, basename='modelregistry')

urlpatterns = [
    path('plot/<int:plot_id>/', PlotForecastAPIView.as_view(), name='api_plot_forecast'),
    path('point/', PointForecastAPIView.as_view(), name='api_point_forecast'),
    path('retrain/', TriggerRetrainAPIView.as_view(), name='api_trigger_retrain'),
] + router.urls
