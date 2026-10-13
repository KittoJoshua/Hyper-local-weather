"""
Serializers for AdvisoryAlert entity.
"""
from rest_framework import serializers
from apps.advisories.models import AdvisoryAlert


class AdvisoryAlertSerializer(serializers.ModelSerializer):
    plot_name = serializers.ReadOnlyField(source='farmplot.name')

    class Meta:
        model = AdvisoryAlert
        fields = [
            'id', 'farmplot', 'plot_name', 'forecast_log',
            'alert_type', 'severity', 'headline',
            'agronomic_advice', 'is_critical', 'is_acknowledged',
            'created_at'
        ]
