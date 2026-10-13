"""
Seed script to initialize crop profiles, demo accounts, meteorological stations,
and sample farmer plots in Kenyan agricultural hubs.
"""
from django.core.management.base import BaseCommand
from apps.accounts.models import User
from apps.farms.models import CropProfile, FarmPlot
from apps.climate.models import MeteorologicalStation
from apps.forecasting.models import ModelRegistry


class Command(BaseCommand):
    help = "Seed database with initial crops, stations, sample plots, and demo accounts."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Seeding database..."))

        # 1. Create Default Crop Profiles
        crops_data = [
            {
                "crop_name": "Maize (Highland White H614)",
                "optimal_temp_min": 15.0,
                "optimal_temp_max": 28.0,
                "drought_threshold_mm": 15.0,
                "growth_cycle_days": 150,
                "description": "Staple grain crop widely grown in Nakuru and Rift Valley highlands."
            },
            {
                "crop_name": "Common Beans (Rosecoco GLP 2)",
                "optimal_temp_min": 16.0,
                "optimal_temp_max": 26.0,
                "drought_threshold_mm": 12.0,
                "growth_cycle_days": 90,
                "description": "Legume pulse essential for rural protein and soil nitrogen fixation."
            },
            {
                "crop_name": "Irish Potatoes (Shangi)",
                "optimal_temp_min": 13.0,
                "optimal_temp_max": 22.0,
                "drought_threshold_mm": 18.0,
                "growth_cycle_days": 90,
                "description": "High-altitude tuber highly sensitive to late blight and heat spikes."
            },
            {
                "crop_name": "Sorghum (Serena / Gadam)",
                "optimal_temp_min": 20.0,
                "optimal_temp_max": 34.0,
                "drought_threshold_mm": 8.0,
                "growth_cycle_days": 110,
                "description": "Drought-tolerant cereal crop resilient to erratic bimodal rains."
            },
            {
                "crop_name": "Tea (Highland Clone)",
                "optimal_temp_min": 14.0,
                "optimal_temp_max": 25.0,
                "drought_threshold_mm": 25.0,
                "growth_cycle_days": 365,
                "description": "Perennial cash crop thriving in Kericho and Nandi high rainfall zones."
            }
        ]

        created_crops = {}
        for c_data in crops_data:
            crop, created = CropProfile.objects.get_or_create(crop_name=c_data["crop_name"], defaults=c_data)
            created_crops[c_data["crop_name"]] = crop
            if created:
                self.stdout.write(self.style.SUCCESS(f"  + Crop Profile: {crop.crop_name}"))

        # 2. Create Meteorological Stations
        stations_data = [
            {"station_name": "Nakuru KMD Synoptic Node", "latitude": -0.283, "longitude": 36.066, "elevation_m": 1860.0, "county": "Nakuru"},
            {"station_name": "Kericho Tea Agro-Meteo Node", "latitude": -0.369, "longitude": 35.286, "elevation_m": 2184.0, "county": "Kericho"},
            {"station_name": "Eldoret Airport Observation Node", "latitude": 0.514, "longitude": 35.269, "elevation_m": 2095.0, "county": "Uasin Gishu"},
            {"station_name": "Embu KALRO Research Node", "latitude": -0.428, "longitude": 37.558, "elevation_m": 1500.0, "county": "Embu"},
        ]
        for s_data in stations_data:
            st, created = MeteorologicalStation.objects.get_or_create(station_name=s_data["station_name"], defaults=s_data)
            if created:
                self.stdout.write(self.style.SUCCESS(f"  + Met Station: {st.station_name}"))

        # 3. Create Demo Users
        # Admin
        admin_user, created = User.objects.get_or_create(
            username="admin",
            defaults={
                "email": "admin@strathmore.edu",
                "phone_number": "0700000000",
                "role": User.Role.ADMIN,
                "is_staff": True,
                "is_superuser": True
            }
        )
        if created:
            admin_user.set_password("AdminPass123!")
            admin_user.save()
            self.stdout.write(self.style.SUCCESS("  + Admin User: admin / AdminPass123!"))

        # Farmer
        farmer_user, created = User.objects.get_or_create(
            username="kitto_farmer",
            defaults={
                "email": "farmer@example.com",
                "phone_number": "0712345678",
                "first_name": "Joshua",
                "last_name": "Kitto",
                "role": User.Role.FARMER,
                "county": "Nakuru"
            }
        )
        if created:
            farmer_user.set_password("FarmerPass123!")
            farmer_user.save()
            self.stdout.write(self.style.SUCCESS("  + Farmer User: kitto_farmer / FarmerPass123!"))

        # 4. Create Sample Farm Plots
        sample_plots = [
            {
                "farmer": farmer_user,
                "current_crop": created_crops.get("Maize (Highland White H614)"),
                "name": "Njoro Ridge Maize Parcel",
                "county": "Nakuru",
                "sub_county": "Njoro",
                "ward": "Mauche",
                "size_acres": 2.50,
                "centroid_lat": -0.354000,
                "centroid_lon": 35.942000,
                "elevation_m": 2180.00,
                "boundary_geojson": [
                    [-0.3530, 35.9410],
                    [-0.3530, 35.9430],
                    [-0.3550, 35.9430],
                    [-0.3550, 35.9410]
                ]
            },
            {
                "farmer": farmer_user,
                "current_crop": created_crops.get("Irish Potatoes (Shangi)"),
                "name": "Mau Narok Highland Shamba",
                "county": "Nakuru",
                "sub_county": "Njoro",
                "ward": "Mau Narok",
                "size_acres": 1.75,
                "centroid_lat": -0.589000,
                "centroid_lon": 35.990000,
                "elevation_m": 2450.00,
                "boundary_geojson": [
                    [-0.5880, 35.9890],
                    [-0.5880, 35.9910],
                    [-0.5900, 35.9910],
                    [-0.5900, 35.9890]
                ]
            }
        ]

        for p_data in sample_plots:
            plot, created = FarmPlot.objects.get_or_create(
                farmer=p_data["farmer"],
                name=p_data["name"],
                defaults=p_data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"  + Farm Plot: {plot.name} ({plot.county})"))

        # 5. Populate Model Registry
        registry_data = [
            {
                "model_name": "Hybrid Residual Forecaster (LSTM + XGBoost)",
                "version_tag": "Hybrid-v1.0-prod",
                "algorithm": "Hybrid Stacking",
                "rmse_score": 1.1840,
                "mae_score": 0.8620,
                "r2_score": 0.9240,
                "is_active": True,
                "hyperparameters": {"lstm_layers": 2, "hidden_units": 64, "xgb_estimators": 250, "xgb_depth": 6}
            },
            {
                "model_name": "Standalone XGBoost Micro-Climate Regressor",
                "version_tag": "XGB-v1.0",
                "algorithm": "Extreme Gradient Boosting",
                "rmse_score": 1.4820,
                "mae_score": 1.1240,
                "r2_score": 0.8650,
                "is_active": False,
                "hyperparameters": {"n_estimators": 250, "max_depth": 6, "objective": "reg:squarederror"}
            },
            {
                "model_name": "Standalone Long Short-Term Memory Network",
                "version_tag": "LSTM-v1.0",
                "algorithm": "Recurrent Deep Learning",
                "rmse_score": 1.6210,
                "mae_score": 1.2450,
                "r2_score": 0.8340,
                "is_active": False,
                "hyperparameters": {"seq_length": 72, "hidden_size": 64, "epochs": 25}
            }
        ]

        for reg in registry_data:
            m, created = ModelRegistry.objects.get_or_create(version_tag=reg["version_tag"], defaults=reg)
            if created:
                self.stdout.write(self.style.SUCCESS(f"  + Model Registry: {m.model_name}"))

        self.stdout.write(self.style.SUCCESS("Database seeding completed successfully!"))
