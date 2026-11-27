"""Weather Bricks — National Weather Service variant.

Pulls current observations from api.weather.gov.
No API key required; NWS only asks for a User-Agent identifying the app so
they can contact you about API issues.

Weather data is fetched in three hops, each cached:
  1. /points/{lat},{lon} → URL of the observation-stations list for that grid.
  2. observation-stations list → identifier of the nearest station.
  3. /stations/{id}/observations/latest → current observation.

NWS observations are reported in metric units (°C, km/h, Pa). We convert to
imperial here. Individual fields can be null when a station hasn't reported that
quantity in its most recent cycle.
"""
import os
from typing import Any, Optional

import geonamescache
import requests
from cachetools import TTLCache
from flask import Flask, jsonify, render_template, request


APP_NAME = os.environ.get("DATABRICKS_APP_NAME", "weather-app")
NWS_API_BASE = "https://api.weather.gov"
# NWS requires a User-Agent identifying the application. Format guidance from
# weather.gov: "AppName (contact-url-or-email)". Override via env var in app.yaml
# so each deployment identifies itself.
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT",
    f"{APP_NAME} (https://www.databricks.com)",
)
REQUEST_TIMEOUT = 5
OBSERVATION_TTL_SECONDS = 300        # observations refresh ~hourly; 5 min is conservative
STATION_TTL_SECONDS = 24 * 60 * 60   # station mapping for a grid is effectively static
CACHE_MAXSIZE = 256
MAX_STATIONS_TO_TRY = 5              # some stations (RAWS, fire weather) report nulls — skip and retry

# Build name-indexed lookups of US cities for free-text search.
# geonamescache.get_cities() returns ~28k world cities (population >= 15k).
# We collapse to US, then index by both primary name and GeoNames' bundled
# alternate names (which include common forms like "New York", "NYC",
# "Big Apple" for NYC since its primary name is "New York City"). Primary-
# name matches win over alternate-name matches so "Manhattan" resolves to
# Manhattan, KS — not NYC, which lists "Manhattan" as one of its altnames.
_gc = geonamescache.GeonamesCache()
_US_CITIES_BY_NAME: dict[str, list[dict[str, Any]]] = {}
_US_CITIES_BY_ALTNAME: dict[str, list[dict[str, Any]]] = {}
for _city in _gc.get_cities().values():
    if _city["countrycode"] != "US":
        continue
    _US_CITIES_BY_NAME.setdefault(_city["name"].lower(), []).append(_city)
    for _alt in _city.get("alternatenames", []):
        _US_CITIES_BY_ALTNAME.setdefault(_alt.lower(), []).append(_city)

stations_cache:    TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=STATION_TTL_SECONDS)
observation_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=OBSERVATION_TTL_SECONDS)

# Single shared session so the User-Agent header is sent on every call.
session = requests.Session()
session.headers.update({
    "User-Agent": NWS_USER_AGENT,
    "Accept": "application/geo+json",
})

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Unit conversions and helpers
# ---------------------------------------------------------------------------

def c_to_f(value: Optional[float]) -> Optional[int]:
    return round(value * 9 / 5 + 32) if value is not None else None


def kmh_to_mph(value: Optional[float]) -> Optional[int]:
    return round(value * 0.621371) if value is not None else None


_COMPASS_POINTS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def degrees_to_compass(degrees: Optional[float]) -> Optional[str]:
    if degrees is None:
        return None
    return _COMPASS_POINTS[int((degrees + 11.25) % 360 / 22.5)]


# ---------------------------------------------------------------------------
# Geocoding (offline, GeoNames data)
# ---------------------------------------------------------------------------

def geocode_us_city(query: str) -> dict[str, Any]:
    """Resolve a free-text city name to lat/lon for a US city.

    Disambiguates duplicate names (Springfield, Portland, etc.) by picking the
    highest-population match. Raises LookupError if no US city matches.
    """
    key = query.strip().lower()
    # Primary name first (so "Manhattan" → Manhattan, KS rather than NYC, which
    # carries "Manhattan" as one of its alternate names). Fall back to altnames
    # when no primary match exists ("New York", "NYC", "Big Apple" → NYC).
    matches = _US_CITIES_BY_NAME.get(key) or _US_CITIES_BY_ALTNAME.get(key) or []
    if not matches:
        raise LookupError(
            f"No US city found matching '{query}'. NWS only covers US locations."
        )
    best = max(matches, key=lambda c: c["population"])
    return {
        "name":  best["name"],
        "state": best.get("admin1code", ""),
        "lat":   best["latitude"],
        "lon":   best["longitude"],
    }


# ---------------------------------------------------------------------------
# NWS lookups
# ---------------------------------------------------------------------------

def get_stations(lat: float, lon: float) -> list[str]:
    """Resolve lat/lon to an ordered list of nearby NWS station identifiers.

    Two-step NWS lookup: /points/{lat},{lon} → observationStations URL → list
    of stations sorted by distance from the grid centroid. We return the full
    ordered list so callers can fall through if the closest station is silent.
    """
    key = (round(lat, 4), round(lon, 4))
    if key in stations_cache:
        return stations_cache[key]

    response = session.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    stations_url = response.json()["properties"]["observationStations"]

    response = session.get(stations_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise ValueError("No observation stations found near this location.")
    station_ids = [f["properties"]["stationIdentifier"] for f in features]

    stations_cache[key] = station_ids
    return station_ids


def fetch_observation(query: str) -> dict[str, Any]:
    cache_key = query.strip().lower()
    if cache_key in observation_cache:
        return observation_cache[cache_key]

    city = geocode_us_city(query)
    station_ids = get_stations(city["lat"], city["lon"])

    # Walk the station list until we find one reporting a real temperature.
    # The closest station is sometimes a RAWS / fire-weather / specialty
    # station that doesn't push standard METAR-style observations — those
    # come back with every value set to null. Bail when we find one with
    # actual data, or after MAX_STATIONS_TO_TRY attempts.
    for station_id in station_ids[:MAX_STATIONS_TO_TRY]:
        try:
            response = session.get(
                f"{NWS_API_BASE}/stations/{station_id}/observations/latest",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.HTTPError:
            continue

        obs = response.json().get("properties", {})
        if (obs.get("temperature") or {}).get("value") is None:
            continue

        shaped = shape_observation_response(city, obs)
        observation_cache[cache_key] = shaped
        return shaped

    raise ValueError(
        f"No NWS station near '{query}' is currently reporting observations."
    )


def shape_observation_response(city: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    # Each measurement is a dict like {"value": X, "unitCode": ..., "qualityControl": ...}
    # Use `or {}` so a null/missing field still dereferences cleanly.
    temp_c        = (obs.get("temperature")       or {}).get("value")
    heat_index_c  = (obs.get("heatIndex")         or {}).get("value")
    wind_chill_c  = (obs.get("windChill")         or {}).get("value")
    wind_kmh      = (obs.get("windSpeed")         or {}).get("value")
    wind_deg      = (obs.get("windDirection")     or {}).get("value")
    humidity      = (obs.get("relativeHumidity")  or {}).get("value")

    # NWS sets heatIndex when hot and windChill when cold (mutually exclusive);
    # when neither applies, "feels like" is just the actual temperature.
    feels_like_c = heat_index_c if heat_index_c is not None else wind_chill_c
    if feels_like_c is None:
        feels_like_c = temp_c

    state = city.get("state", "")
    return {
        "location":         f"{city['name']}, {state}" if state else city["name"],
        "city":             city["name"],
        "state":            state,
        "condition":        obs.get("textDescription"),
        "temperature":      c_to_f(temp_c),
        "feels_like":       c_to_f(feels_like_c),
        "humidity":         round(humidity) if humidity is not None else None,
        "wind_speed":       kmh_to_mph(wind_kmh),
        "wind_direction":   degrees_to_compass(wind_deg),
        "units":            "imperial",
        "temperature_unit": "°F",
        "wind_unit":        "mph",
        "icon_url":         obs.get("icon"),
        "observed_at":      obs.get("timestamp"),
        "source":           "National Weather Service",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/weather/conditions")
def weather_conditions():
    city = request.args.get("city", "").strip()

    if not city:
        return jsonify({"error": "Query parameter 'city' is required."}), 400

    try:
        result = fetch_observation(city)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return jsonify({"error": f"NWS returned HTTP {status} for '{city}'."}), 502
    except requests.RequestException:
        return jsonify({"error": "Weather service is unreachable. Please try again."}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")),
        debug=False,
    )
