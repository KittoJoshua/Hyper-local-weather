"""
Web UI Views for Farmer Forecast Dashboard and Admin ML Control Panel.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.farms.models import FarmPlot
from apps.forecasting.models import ModelRegistry
from apps.advisories.rules_engine import AgronomicRulesEngine
from ml_engine.inference import MicroClimateInferenceEngine
import json


@login_required
def dashboard_view(request):
    """
    Primary Farmer Dashboard corresponding to Wireframe 1 (Proposal page 41).
    Displays current conditions, alerts, 72-hour forecast chart, 7-day outlook, and Leaflet preview.
    """
    user = request.user
    plots = FarmPlot.objects.filter(farmer=user) if not user.is_platform_admin else FarmPlot.objects.all()

    selected_plot_id = request.GET.get("plot")
    selected_plot = None
    if selected_plot_id:
        selected_plot = plots.filter(id=selected_plot_id).first()

    if not selected_plot and plots.exists():
        selected_plot = plots.first()

    forecast_data = None
    alerts = []
    hourly_json = "[]"

    if selected_plot:
        engine = MicroClimateInferenceEngine()
        forecast_data = engine.generate_forecast(
            lat=float(selected_plot.centroid_lat),
            lon=float(selected_plot.centroid_lon),
            elevation_m=float(selected_plot.elevation_m)
        )

        # Generate agronomic advisories
        rules = AgronomicRulesEngine()
        alerts = rules.evaluate_risks(
            forecast_data=forecast_data,
            crop=selected_plot.current_crop
        )

        hourly_json = json.dumps(forecast_data.get("hourly_forecast", [])[:72]) # Full 72 hours for trajectory chart

    context = {
        "plots": plots,
        "selected_plot": selected_plot,
        "forecast": forecast_data,
        "alerts": alerts,
        "hourly_json": hourly_json,
    }
    return render(request, "dashboard/farmer_dashboard.html", context)


@login_required
def admin_ml_panel(request):
    """
    System Administration & ML Management corresponding to Wireframe 4 (Proposal page 42).
    """
    if not request.user.is_platform_admin:
        messages.error(request, "Access restricted to System Administrators.")
        return redirect("dashboard")

    # Load benchmark report from weights if available
    benchmark_report = None
    import os
    from ml_engine.config import WEIGHTS_DIR
    json_path = WEIGHTS_DIR / "model_benchmark_report.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                benchmark_report = json.load(f)
        except Exception:
            pass

    # Build dynamic rows for the Walk-Forward Validation Benchmark Table
    benchmark_rows = []
    if benchmark_report:
        labels_map = {
            "temperature_c": "Temperature (°C)",
            "humidity_percent": "Relative Humidity (%)",
            "precipitation_mm": "Precipitation (mm)"
        }
        badge_map = {
            "Persistence Baseline": ("Naive Baseline", "bg-secondary"),
            "Historical Rolling Average (24h)": ("Statistical Baseline", "bg-secondary"),
            "Standalone XGBoost": ("Strong Tabular", "bg-primary"),
            "Standalone LSTM": ("Sequential Deep Learning", "bg-info text-dark"),
            "Hybrid (LSTM + XGBoost)": ("Optimal Architecture", "bg-success")
        }
        for arch_name, targets in benchmark_report.items():
            badge_text, badge_class = badge_map.get(arch_name, ("Benchmarked Model", "bg-secondary"))
            is_hybrid = "Hybrid" in arch_name
            target_items = []
            for target_key, metrics in targets.items():
                target_label = labels_map.get(target_key, target_key)
                target_items.append({
                    "target_label": target_label,
                    "rmse": f"{float(metrics.get('RMSE', 0.0)):.4f}",
                    "mae": f"{float(metrics.get('MAE', 0.0)):.4f}",
                    "r2": f"{float(metrics.get('R2', 0.0)):.4f}",
                })
            benchmark_rows.append({
                "architecture": arch_name,
                "badge_text": badge_text,
                "badge_class": badge_class,
                "is_hybrid": is_hybrid,
                "targets": target_items
            })

        # Synchronize ModelRegistry database entries with newly trained scores
        hybrid_temp = benchmark_report.get("Hybrid (LSTM + XGBoost)", {}).get("temperature_c", {})
        if hybrid_temp:
            ModelRegistry.objects.filter(algorithm="Hybrid Stacking").update(
                rmse_score=hybrid_temp.get("RMSE"),
                mae_score=hybrid_temp.get("MAE"),
                r2_score=hybrid_temp.get("R2")
            )
        xgb_temp = benchmark_report.get("Standalone XGBoost", {}).get("temperature_c", {})
        if xgb_temp:
            ModelRegistry.objects.filter(algorithm="Extreme Gradient Boosting").update(
                rmse_score=xgb_temp.get("RMSE"),
                mae_score=xgb_temp.get("MAE"),
                r2_score=xgb_temp.get("R2")
            )
        lstm_temp = benchmark_report.get("Standalone LSTM", {}).get("temperature_c", {})
        if lstm_temp:
            ModelRegistry.objects.filter(algorithm="Recurrent Deep Learning").update(
                rmse_score=lstm_temp.get("RMSE"),
                mae_score=lstm_temp.get("MAE"),
                r2_score=lstm_temp.get("R2")
            )

    models_list = ModelRegistry.objects.all()
    active_model = models_list.filter(is_active=True).first()

    context = {
        "models": models_list,
        "active_model": active_model,
        "benchmark_report": benchmark_report,
        "benchmark_rows": benchmark_rows,
    }
    return render(request, "admin_panel/model_management.html", context)
