"""
API route definitions for climate data.
"""
from rest_framework.routers import DefaultRouter
from apps.climate.views_api import ClimateRecordViewSet, MeteorologicalStationViewSet

router = DefaultRouter()
router.register(r'records', ClimateRecordViewSet, basename='climaterecord')
router.register(r'stations', MeteorologicalStationViewSet, basename='meteorologicalstation')

urlpatterns = router.urls
