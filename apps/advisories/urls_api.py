"""
API route definitions for advisories.
"""
from rest_framework.routers import DefaultRouter
from apps.advisories.views_api import AdvisoryAlertViewSet

router = DefaultRouter()
router.register(r'alerts', AdvisoryAlertViewSet, basename='advisoryalert')

urlpatterns = router.urls
