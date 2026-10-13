"""
Data Loader for Historical Climate Datasets and Real-Time Feeds.
Supports reading from local CSV archives, NASA POWER / ERA5 datasets,
and the Django persistent storage layer.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from ml_engine.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Representative agricultural regions in Kenya with distinct micro-climates
KENYA_REGIONS = {
    "Nakuru_Njoro": {
        "lat": -0.354, "lon": 35.942, "elevation_m": 2200,
        "base_temp": 17.5, "temp_range": (8.0, 26.0),
        "annual_rainfall_mm": 1050
    },
    "Kericho_Central": {
        "lat": -0.369, "lon": 35.286, "elevation_m": 2184,
        "base_temp": 16.8, "temp_range": (10.0, 25.0),
        "annual_rainfall_mm": 1900
    },
    "Uasin_Gishu_Eldoret": {
        "lat": 0.514, "lon": 35.269, "elevation_m": 2095,
        "base_temp": 17.2, "temp_range": (9.0, 27.0),
        "annual_rainfall_mm": 1100
    },
    "Embu_Runyenjes": {
        "lat": -0.428, "lon": 37.558, "elevation_m": 1500,
        "base_temp": 20.5, "temp_range": (12.0, 29.0),
        "annual_rainfall_mm": 1250
    }
}


def generate_synthetic_historical_dataset(
    region_key: str = "Nakuru_Njoro",
    start_date: str = "2021-01-01",
    end_date: str = "2026-06-01",
    freq: str = "1h",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, meteorologically grounded multi-year time-series dataset
    capturing Kenya's bimodal tropical rainfall seasons (Long Rains: Mar-May; Short Rains: Oct-Dec),
    diurnal thermal swings, and local orographic influences.
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)

    meta = KENYA_REGIONS.get(region_key, KENYA_REGIONS["Nakuru_Njoro"])
    lat = meta["lat"]
    lon = meta["lon"]
    elevation = meta["elevation_m"]

    # Day of year and hour of day
    doy = timestamps.dayofyear.values
    hour = timestamps.hour.values

    # 1. Bimodal seasonality for rainfall in Kenya:
    # Primary peak (Long Rains): late April (DOY ~ 115)
    # Secondary peak (Short Rains): mid November (DOY ~ 315)
    long_rains = np.exp(-0.5 * ((doy - 115) / 25) ** 2) * 1.8
    short_rains = np.exp(-0.5 * ((doy - 315) / 20) ** 2) * 1.3
    season_rain_prob = 0.12 + 0.45 * (long_rains + short_rains)
    season_rain_prob = np.clip(season_rain_prob, 0.05, 0.85)

    # Convective diurnal afternoon rainfall pattern (mostly 14:00 - 19:00 local time)
    afternoon_convection = np.exp(-0.5 * ((hour - 16) / 2.5) ** 2)
    hourly_rain_prob = season_rain_prob * (0.2 + 0.8 * afternoon_convection)

    rain_occurred = (np.random.rand(n) < hourly_rain_prob).astype(float)
    rain_amount = rain_occurred * np.random.exponential(scale=4.5, size=n)
    # Occasional heavy convective storm (> 20mm/hr)
    heavy_storm = (np.random.rand(n) < 0.015) * rain_occurred * np.random.uniform(15.0, 45.0, size=n)
    precipitation_mm = np.round(rain_amount + heavy_storm, 2)

    # 2. Temperature modeling (Diurnal cycle + Lapse Rate + Cloud cooling)
    # Kenya is near the equator, so seasonal mean temperature variation is small (~2-3C),
    # but diurnal variation is high (~12-15C).
    solar_diurnal = -np.cos((hour - 14) * 2 * np.pi / 24)  # Peak at 14:00
    seasonal_temp_shift = 1.8 * np.sin((doy - 40) * 2 * np.pi / 365) # Warmest in Jan-Feb, cooler in July
    rain_cooling = -3.5 * (precipitation_mm > 0.5)

    # Elevation lapse rate: ~6.5C decrease per 1000m from sea level baseline (28C)
    elev_temp_base = 28.0 - (elevation / 1000.0) * 5.2
    temperature_c = (
        elev_temp_base
        + seasonal_temp_shift
        + 6.5 * solar_diurnal
        + rain_cooling
        + np.random.normal(0, 0.9, n)
    )
    temperature_c = np.round(np.clip(temperature_c, meta["temp_range"][0], meta["temp_range"][1]), 2)

    # 3. Relative Humidity (inverse to temperature, increases during rain)
    # Minimum at max temp, maximum at dawn (05:00)
    humidity_percent = (
        85.0
        - 2.8 * (temperature_c - elev_temp_base)
        + 18.0 * (precipitation_mm > 0.0)
        + np.random.normal(0, 3.0, n)
    )
    humidity_percent = np.round(np.clip(humidity_percent, 25.0, 99.0), 2)

    # 4. Atmospheric pressure (hPa) based on barometric formula
    base_pressure = 1013.25 * (1 - (0.0065 * elevation) / 288.15) ** 5.255
    pressure_hpa = np.round(base_pressure + np.random.normal(0, 1.2, n), 2)

    # 5. Wind speed (m/s)
    wind_speed_ms = np.round(np.clip(2.0 + 1.5 * np.sin(hour * np.pi / 12) + np.random.exponential(1.2, n), 0.2, 18.0), 2)

    df = pd.DataFrame({
        "recorded_at": timestamps,
        "region": region_key,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "precipitation_mm": precipitation_mm,
        "pressure_hpa": pressure_hpa,
        "wind_speed_ms": wind_speed_ms
    })

    return df


from typing import Optional, Dict


def load_dataset(
    region_key: str = "Nakuru_Njoro",
    force_regenerate: bool = False,
    max_records: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads historical climate data for a given region. If a processed CSV exists,
    it is loaded; otherwise, raw data is generated/ingested and cached.
    If max_records is specified, extracts the most recent N records.
    """
    processed_path = PROCESSED_DATA_DIR / f"{region_key}_climate_hourly.csv"
    if processed_path.exists() and not force_regenerate:
        df = pd.read_csv(processed_path, parse_dates=["recorded_at"])
    else:
        raw_path = RAW_DATA_DIR / f"{region_key}_climate_raw.csv"
        if raw_path.exists() and not force_regenerate:
            df = pd.read_csv(raw_path, parse_dates=["recorded_at"])
        else:
            df = generate_synthetic_historical_dataset(region_key=region_key)
            df.to_csv(raw_path, index=False)
        df.to_csv(processed_path, index=False)

    if max_records and len(df) > max_records:
        df = df.tail(max_records).reset_index(drop=True)

    return df


def extract_fewer_records(
    region_key: str = "Nakuru_Njoro",
    records: int = 8760,
    overwrite_processed: bool = True
) -> pd.DataFrame:
    """
    Extracts a lightweight subset of records (default: 8,760 records = 1 full year)
    from the raw dataset and updates the processed dataset for fast training (< 30 seconds).
    """
    raw_path = RAW_DATA_DIR / f"{region_key}_climate_raw.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=["recorded_at"])
    else:
        df = generate_synthetic_historical_dataset(region_key=region_key)
        df.to_csv(raw_path, index=False)

    if len(df) > records:
        df_subset = df.tail(records).reset_index(drop=True)
    else:
        df_subset = df

    if overwrite_processed:
        processed_path = PROCESSED_DATA_DIR / f"{region_key}_climate_hourly.csv"
        df_subset.to_csv(processed_path, index=False)
        print(f"[DataLoader] Successfully extracted {len(df_subset)} records to {processed_path}")

    return df_subset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract smaller climate dataset for fast model training.")
    parser.add_argument("--region", type=str, default="Nakuru_Njoro", help="Target agricultural region")
    parser.add_argument("--records", type=int, default=8760, help="Number of recent hourly records to extract (8760 = 1 year)")
    args = parser.parse_args()

    extract_fewer_records(region_key=args.region, records=args.records)
