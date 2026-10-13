"""
REST API Views for Plot Registration, GIS queries, and Crops.
"""
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from apps.farms.models import FarmPlot, CropProfile
from apps.farms.serializers import FarmPlotSerializer, CropProfileSerializer


class CropProfileViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET: List available Kenyan staple crop profiles and physiological thresholds.
    """
    queryset = CropProfile.objects.all()
    serializer_class = CropProfileSerializer
    permission_classes = [permissions.AllowAny]


class FarmPlotViewSet(viewsets.ModelViewSet):
    """
    CRUD ViewSet for smallholder farm plots.
    Restricts access to the authenticated owner unless user is an Administrator.
    """
    serializer_class = FarmPlotSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_platform_admin:
            return FarmPlot.objects.all()
        return FarmPlot.objects.filter(farmer=user)

    def perform_create(self, serializer):
        serializer.save(farmer=self.request.user)
