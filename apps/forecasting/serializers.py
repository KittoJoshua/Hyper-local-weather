"""
Serializers for ModelRegistry and ForecastLog entities.
"""
from rest_framework import serializers
from apps.forecasting.models import ModelRegistry, ForecastLog


class ModelRegistrySerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelRegistry
        fields = '__all__'


class ForecastLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastLog
        fields = '__all__'
