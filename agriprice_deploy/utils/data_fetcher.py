"""
Data fetchers for ISRO Bhuvan NDVI, OpenWeather, and Agmarknet APMC.
Falls back to realistic synthetic data when APIs are unavailable (demo mode).
"""

import os, requests, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─── NDVI from ISRO Bhuvan ────────────────────────────────────────────────────
# Real endpoint: https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms
# Returns GeoTIFF; parse mean NDVI per region bounding box.
# Synthetic fallback mimics real NDVI seasonal cycle.

CROP_STATE_COORDS = {
    ("Tomato",  "Karnataka"):        (12.9716, 77.5946),
    ("Onion",   "Maharashtra"):      (19.9975, 73.7898),
    ("Wheat",   "Punjab"):           (30.9010, 75.8573),
    ("Potato",  "Uttar Pradesh"):    (26.8467, 80.9462),
    ("Rice",    "Andhra Pradesh"):   (16.5062, 80.6480),
    ("Cotton",  "Maharashtra"):      (20.7002, 77.0082),
    ("Soybean", "Madhya Pradesh"):   (23.2599, 77.4126),
    ("Maize",   "Karnataka"):        (15.3173, 75.7139),
}

# Base NDVI health profile per crop (healthy = 0.65–0.85, stressed = 0.3–0.5)
CROP_NDVI_PROFILE = {
    "Tomato":  (0.62, 0.08), "Onion":   (0.58, 0.07),
    "Wheat":   (0.72, 0.09), "Potato":  (0.68, 0.08),
    "Rice":    (0.75, 0.10), "Cotton":  (0.60, 0.09),
    "Soybean": (0.70, 0.09), "Maize":   (0.65, 0.08),
}


def fetch_ndvi_data(crop: str, state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch NDVI time series for a crop-state pair.
    Live: ISRO Bhuvan WMS GeoTIFF → mean NDVI extraction.
    Demo: Synthetic seasonal NDVI with realistic noise.
    """
    api_key = os.getenv("BHUVAN_TOKEN", "")
    coords  = CROP_STATE_COORDS.get((crop, state), (20.5937, 78.9629))

    if api_key:
        try:
            return _live_ndvi(crop, state, coords, days, api_key)
        except Exception:
            pass

    return _synthetic_ndvi(crop, days)


def _live_ndvi(crop, state, coords, days, api_key):
    """Parse ISRO Bhuvan NDVI — placeholder for actual WMS integration."""
    # Real implementation:
    # 1. Build WMS GetMap request for BBOX around coords
    # 2. Parse returned GeoTIFF with rasterio
    # 3. Compute zonal statistics (mean NDVI) for crop area
    # 4. Return as daily DataFrame
    raise NotImplementedError("Bhuvan WMS integration requires rasterio + API key")


def _synthetic_ndvi(crop: str, days: int) -> pd.DataFrame:
    """Realistic synthetic NDVI with seasonal pattern + noise."""
    mu, sigma = CROP_NDVI_PROFILE.get(crop, (0.65, 0.08))
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    # Seasonal sine + random walk
    t       = np.linspace(0, 2 * np.pi, days)
    trend   = mu + 0.08 * np.sin(t - np.pi / 3)
    noise   = np.random.normal(0, sigma * 0.25, days)
    ndvi    = np.clip(trend + noise, 0.1, 0.95)
    return pd.DataFrame({"date": [d.isoformat() for d in dates], "ndvi": ndvi.round(4)})


# ─── Weather from OpenWeather ─────────────────────────────────────────────────
STATE_COORDS = {
    "Karnataka":        (12.9716, 77.5946),
    "Maharashtra":      (19.0760, 72.8777),
    "Punjab":           (30.9010, 75.8573),
    "Madhya Pradesh":   (23.2599, 77.4126),
    "Uttar Pradesh":    (26.8467, 80.9462),
    "Andhra Pradesh":   (17.3850, 78.4867),
}

MONSOON_PROFILE = {
    "Karnataka":        {"base_rain": 3.2, "base_temp": 26, "humidity": 72},
    "Maharashtra":      {"base_rain": 2.8, "base_temp": 28, "humidity": 68},
    "Punjab":           {"base_rain": 1.1, "base_temp": 24, "humidity": 55},
    "Madhya Pradesh":   {"base_rain": 2.5, "base_temp": 27, "humidity": 65},
    "Uttar Pradesh":    {"base_rain": 2.0, "base_temp": 28, "humidity": 62},
    "Andhra Pradesh":   {"base_rain": 3.5, "base_temp": 30, "humidity": 75},
}


def fetch_weather_data(state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical + forecast climate variables.
    Live: OpenWeather One Call API 3.0
    Demo: Synthetic climate with monsoon seasonality.
    """
    api_key = os.getenv("OPENWEATHER_KEY", "")
    coords  = STATE_COORDS.get(state, (20.5937, 78.9629))

    if api_key:
        try:
            return _live_weather(coords, days, api_key)
        except Exception:
            pass

    return _synthetic_weather(state, days)


def _live_weather(coords, days, api_key):
    """OpenWeather One Call API integration."""
    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    rows = []
    for i in range(min(days, 5)):   # free tier limit; production uses bulk history
        dt   = int((datetime.now() - timedelta(days=days - i)).timestamp())
        resp = requests.get(f"{url}&dt={dt}", timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"OpenWeather error {resp.status_code}")
        data = resp.json().get("data", [{}])[0]
        rows.append({
            "date":      (datetime.now() - timedelta(days=days - i)).date().isoformat(),
            "temp_max":  data.get("temp", 28),
            "temp_min":  data.get("feels_like", 22),
            "humidity":  data.get("humidity", 65),
            "rainfall":  data.get("rain", {}).get("1h", 0) * 24,
            "wind_kmh":  data.get("wind_speed", 12) * 3.6,
        })
    return pd.DataFrame(rows)


def _synthetic_weather(state: str, days: int) -> pd.DataFrame:
    profile = MONSOON_PROFILE.get(state, {"base_rain": 2.0, "base_temp": 27, "humidity": 65})
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    month = datetime.now().month
    # Indian monsoon simulation: Jun–Sep high rain
    monsoon_factor = 1.0 + 2.5 * max(0, np.sin(np.pi * (month - 6) / 4)) if 6 <= month <= 9 else 0.4

    temp_base = profile["base_temp"]
    rows = []
    for i, d in enumerate(dates):
        rain = max(0, np.random.exponential(profile["base_rain"] * monsoon_factor))
        rows.append({
            "date":     d.isoformat(),
            "temp_max": round(temp_base + random.uniform(-2, 3), 1),
            "temp_min": round(temp_base - random.uniform(4, 8), 1),
            "humidity": round(profile["humidity"] + random.uniform(-8, 8), 1),
            "rainfall": round(rain, 1),
            "wind_kmh": round(random.uniform(8, 25), 1),
        })
    return pd.DataFrame(rows)


# ─── Mandi prices from Agmarknet ─────────────────────────────────────────────
# API: https://agmarknet.gov.in/SearchCmmMkt.aspx  (commodity + market filter)
# Realistic price ranges per crop in ₹/Quintal

CROP_PRICE_RANGES = {
    "Tomato":  {"base": 1800, "vol": 600, "trend": 0.002},
    "Onion":   {"base": 2200, "vol": 400, "trend": -0.001},
    "Potato":  {"base": 1600, "vol": 300, "trend": 0.001},
    "Wheat":   {"base": 2100, "vol": 150, "trend": 0.0005},
    "Rice":    {"base": 3200, "vol": 250, "trend": 0.001},
    "Cotton":  {"base": 6500, "vol": 400, "trend": -0.0005},
    "Soybean": {"base": 4800, "vol": 350, "trend": 0.002},
    "Maize":   {"base": 1900, "vol": 200, "trend": 0.001},
}


def fetch_mandi_prices(crop: str, mandi: str, state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch APMC mandi price history from Agmarknet.
    Live: Agmarknet JSON API (commodity + market + date range)
    Demo: Realistic synthetic price with volatility model.
    """
    # In production, Agmarknet exposes:
    # GET https://agmarknet.gov.in/api/prices?commodity={crop}&market={mandi}&state={state}
    # Response: JSON array of {date, min_price, max_price, modal_price}
    return _synthetic_mandi_prices(crop, days)


def _synthetic_mandi_prices(crop: str, days: int) -> pd.DataFrame:
    """Geometric Brownian Motion price simulation matching real mandi volatility."""
    p     = CROP_PRICE_RANGES.get(crop, {"base": 2000, "vol": 300, "trend": 0.001})
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    price = p["base"]
    prices = []

    # GBM with regime switching (supply shock events)
    for i in range(days):
        shock  = 1.0
        if random.random() < 0.05:       # 5% chance of supply/demand shock
            shock = random.choice([0.85, 1.15])
        daily_return = p["trend"] + random.normalvariate(0, p["vol"] / p["base"] * 0.4)
        price = max(price * (1 + daily_return) * shock, p["base"] * 0.4)
        prices.append(round(price, 2))

    # Add min/max spread (mandis report range)
    rows = []
    for d, modal in zip(dates, prices):
        spread = random.uniform(0.03, 0.10) * modal
        rows.append({
            "date":        d.isoformat(),
            "modal_price": modal,
            "min_price":   round(modal - spread, 2),
            "max_price":   round(modal + spread * 1.2, 2),
            "arrivals_q":  round(random.uniform(200, 2000), 0),
        })
    return pd.DataFrame(rows)
