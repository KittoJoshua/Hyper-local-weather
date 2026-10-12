"""
REST API Views for querying active agronomic alerts.
"""
from rest_framework import viewsets, permissions, filters
from apps.advisories.models import AdvisoryAlert
from apps.advisories.serializers import AdvisoryAlertSerializer


class AdvisoryAlertViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET: List active agronomic risk alerts for authenticated farmer's plots.
    """
    serializer_class = AdvisoryAlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering = ['-created_at']

    def get_queryset(self):
        user = self.request.user
        if user.is_platform_admin:
            qs = AdvisoryAlert.objects.all()
        else:
            qs = AdvisoryAlert.objects.filter(farmplot__farmer=user)

        plot_id = self.request.query_params.get('plot_id')
        if plot_id:
            qs = qs.filter(farmplot_id=plot_id)

        is_critical = self.request.query_params.get('critical')
        if is_critical is not None:
            qs = qs.filter(is_critical=is_critical.lower() in ['true', '1'])

        return qs
