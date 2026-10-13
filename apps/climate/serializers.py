"""
Serializers for ClimateRecord and MeteorologicalStation entities.
"""
from rest_framework import serializers
from apps.climate.models import ClimateRecord, MeteorologicalStation


class MeteorologicalStationSerializer(serializers.ModelSerializer):
    class Meta:
        model = MeteorologicalStation
        fields = '__all__'


class ClimateRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClimateRecord
        fields = '__all__'
