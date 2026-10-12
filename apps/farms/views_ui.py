"""
Web UI Views for Farm Plot Management and Leaflet Map Registration.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.farms.models import FarmPlot, CropProfile
import json


@login_required
def register_plot_page(request):
    crops = CropProfile.objects.all()

    if request.method == "POST":
        name = request.POST.get("name")
        crop_id = request.POST.get("current_crop")
        county = request.POST.get("county", "Nakuru")
        sub_county = request.POST.get("sub_county", "")
        ward = request.POST.get("ward", "")
        size_acres = request.POST.get("size_acres", 1.0)
        centroid_lat = request.POST.get("centroid_lat")
        centroid_lon = request.POST.get("centroid_lon")
        elevation_m = request.POST.get("elevation_m", 1950.0)
        boundary_json_str = request.POST.get("boundary_geojson", "")

        crop_obj = CropProfile.objects.filter(id=crop_id).first() if crop_id else None

        boundary_coords = None
        if boundary_json_str:
            try:
                boundary_coords = json.loads(boundary_json_str)
            except Exception:
                pass

        plot = FarmPlot.objects.create(
            farmer=request.user,
            name=name,
            current_crop=crop_obj,
            county=county,
            sub_county=sub_county,
            ward=ward,
            size_acres=float(size_acres),
            centroid_lat=float(centroid_lat or -0.354),
            centroid_lon=float(centroid_lon or 35.942),
            elevation_m=float(elevation_m or 1950.0),
            boundary_geojson=boundary_coords
        )
        messages.success(request, f"Farm Plot '{plot.name}' registered successfully!")
        return redirect("dashboard")

    return render(request, "farms/plot_register.html", {"crops": crops})


@login_required
def plot_list_page(request):
    plots = FarmPlot.objects.filter(farmer=request.user) if not request.user.is_platform_admin else FarmPlot.objects.all()
    return render(request, "farms/plot_list.html", {"plots": plots})
