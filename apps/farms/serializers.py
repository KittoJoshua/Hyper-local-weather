"""
Serializers for FarmPlot and CropProfile entities.
"""
from rest_framework import serializers
from apps.farms.models import FarmPlot, CropProfile


class CropProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CropProfile
        fields = '__all__'


class FarmPlotSerializer(serializers.ModelSerializer):
    crop_name = serializers.ReadOnlyField(source='current_crop.crop_name')
    farmer_name = serializers.ReadOnlyField(source='farmer.username')

    class Meta:
        model = FarmPlot
        fields = [
            'id', 'farmer', 'farmer_name', 'name', 'current_crop', 'crop_name',
            'county', 'sub_county', 'ward', 'size_acres',
            'centroid_lat', 'centroid_lon', 'elevation_m',
            'boundary_geojson', 'created_at'
        ]
        read_only_fields = ['id', 'farmer', 'created_at']

    def create(self, validated_data):
        validated_data['farmer'] = self.context['request'].user
        return super().create(validated_data)
