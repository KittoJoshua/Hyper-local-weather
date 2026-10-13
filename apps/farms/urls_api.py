"""
API route definitions for farms.
"""
from rest_framework.routers import DefaultRouter
from apps.farms.views_api import FarmPlotViewSet, CropProfileViewSet

router = DefaultRouter()
router.register(r'plots', FarmPlotViewSet, basename='farmplot')
router.register(r'crops', CropProfileViewSet, basename='cropprofile')

urlpatterns = router.urls
