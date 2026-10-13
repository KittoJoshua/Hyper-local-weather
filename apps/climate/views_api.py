"""
REST API Views for querying chronological climate records and stations.
"""
from rest_framework import viewsets, permissions, filters
from apps.climate.models import ClimateRecord, MeteorologicalStation
from apps.climate.serializers import ClimateRecordSerializer, MeteorologicalStationSerializer


class MeteorologicalStationViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = MeteorologicalStation.objects.filter(is_active=True)
    serializer_class = MeteorologicalStationSerializer
    permission_classes = [permissions.AllowAny]


class ClimateRecordViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ClimateRecord.objects.all()
    serializer_class = ClimateRecordSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['recorded_at', 'temperature_c', 'precipitation_mm']
    ordering = ['-recorded_at']

    def get_queryset(self):
        qs = super().get_queryset()
        plot_id = self.request.query_params.get('plot_id')
        station_id = self.request.query_params.get('station_id')
        if plot_id:
            qs = qs.filter(farmplot_id=plot_id)
        if station_id:
            qs = qs.filter(station_id=station_id)
        return qs
