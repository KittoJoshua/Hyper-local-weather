"""
REST API Views for Real-Time Forecast Inference and Model Management.
"""
from rest_framework import views, viewsets, status, permissions
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from apps.farms.models import FarmPlot
from apps.forecasting.models import ModelRegistry, ForecastLog
from apps.forecasting.serializers import ModelRegistrySerializer, ForecastLogSerializer
from ml_engine.inference import MicroClimateInferenceEngine


class PlotForecastAPIView(views.APIView):
    """
    GET: /api/v1/forecasts/plot/<id>/?horizon=72h (or ?horizon=7d)
    Returns downscaled micro-climate forecast tailored to the farm plot's
    exact coordinates, elevation, and historical micro-climate characteristics.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, plot_id):
        user = request.user
        if user.is_platform_admin:
            plot = get_object_or_404(FarmPlot, id=plot_id)
        else:
            plot = get_object_or_404(FarmPlot, id=plot_id, farmer=user)

        engine = MicroClimateInferenceEngine()
        forecast_data = engine.generate_forecast(
            lat=float(plot.centroid_lat),
            lon=float(plot.centroid_lon),
            elevation_m=float(plot.elevation_m)
        )

        # Include plot metadata
        response_data = {
            "plot_id": plot.id,
            "plot_name": plot.name,
            "crop": plot.current_crop.crop_name if plot.current_crop else "Unspecified",
            "county": plot.county,
            "sub_county": plot.sub_county,
            **forecast_data
        }
        return Response(response_data, status=status.HTTP_200_OK)


class PointForecastAPIView(views.APIView):
    """
    GET: /api/v1/forecasts/point/?lat=-0.354&lon=35.942&elevation_m=2200
    Generates micro-climate forecast for an arbitrary coordinate point in Kenya.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        lat = request.query_params.get("lat")
        lon = request.query_params.get("lon")
        elevation_m = request.query_params.get("elevation_m", 1900.0)

        if not lat or not lon:
            return Response(
                {"error": "Both 'lat' and 'lon' query parameters are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            lat = float(lat)
            lon = float(lon)
            elevation_m = float(elevation_m)
        except ValueError:
            return Response(
                {"error": "Coordinates and elevation must be valid decimal numbers."},
                status=status.HTTP_400_BAD_REQUEST
            )

        engine = MicroClimateInferenceEngine()
        forecast_data = engine.generate_forecast(lat=lat, lon=lon, elevation_m=elevation_m)
        return Response(forecast_data, status=status.HTTP_200_OK)


class ModelRegistryViewSet(viewsets.ModelViewSet):
    """
    CRUD ViewSet for Machine Learning models in the registry.
    """
    queryset = ModelRegistry.objects.all()
    serializer_class = ModelRegistrySerializer
    permission_classes = [permissions.IsAdminUser]


class TriggerRetrainAPIView(views.APIView):
    """
    POST: Trigger walk-forward validation and model retraining loop.
    """
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        region = request.data.get("region", "Nakuru_Njoro")
        max_records = request.data.get("max_records")
        if max_records:
            try:
                max_records = int(max_records)
            except ValueError:
                max_records = 8760
        else:
            max_records = 8760

        from ml_engine.training_pipeline import run_training_and_benchmarking
        
        try:
            results = run_training_and_benchmarking(region_key=region, max_records=max_records)
            return Response({
                "message": f"Retraining pipeline completed successfully for {region} with {max_records} records.",
                "summary": results["benchmark_summary"]
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
