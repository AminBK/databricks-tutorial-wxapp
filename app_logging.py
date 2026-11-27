"""Weather Bricks — observability lab solution.

Instrumented version of app.py. Same code, with logging statements added at
the decision points where production signal matters: request boundaries,
cache hits/misses, the geocoder, the multi-station retry, and every error
path. Levels follow the usual convention: INFO for normal operation,
WARNING for client errors or recoverable upstream issues, ERROR for
service-level failures.
"""
import logging
import os
import sys
from typing import Any, Optional

import geonamescache
import requests
from cachetools import TTLCache
from flask import Flask, jsonify, render_template, request


APP_NAME = os.environ.get("DATABRICKS_APP_NAME", "weather-app")
NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT",
    f"{APP_NAME} (https://www.databricks.com)",
)
REQUEST_TIMEOUT = 5
OBSERVATION_TTL_SECONDS = 300
STATION_TTL_SECONDS = 24 * 60 * 60
CACHE_MAXSIZE = 256
MAX_STATIONS_TO_TRY = 5

# ---------------------------------------------------------------------------
# Logger setup — single stdout handler with timestamp / level / message format.
# propagate=False keeps our records out of the root logger to avoid duplicates
# when something else (gunicorn, Flask's default handler) also writes to stdout.
# ---------------------------------------------------------------------------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)

# Build name and altname indexes for offline US-city geocoding.
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

session = requests.Session()
session.headers.update({
    "User-Agent": NWS_USER_AGENT,
    "Accept": "application/geo+json",
})

app = Flask(__name__)

logger.info(
    "weather app initialized us_cities=%d observation_ttl_seconds=%s station_ttl_seconds=%s max_stations=%d",
    len(_US_CITIES_BY_NAME),
    OBSERVATION_TTL_SECONDS,
    STATION_TTL_SECONDS,
    MAX_STATIONS_TO_TRY,
)


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


def geocode_us_city(query: str) -> dict[str, Any]:
    key = query.strip().lower()
    matches = _US_CITIES_BY_NAME.get(key) or _US_CITIES_BY_ALTNAME.get(key) or []
    if not matches:
        raise LookupError(
            f"No US city found matching '{query}'. NWS only covers US locations."
        )
    best = max(matches, key=lambda c: c["population"])
    resolved = {
        "name":  best["name"],
        "state": best.get("admin1code", ""),
        "lat":   best["latitude"],
        "lon":   best["longitude"],
    }
    logger.info(
        "geocode resolved query=%r -> name=%r state=%s population=%s",
        query,
        resolved["name"],
        resolved["state"],
        best["population"],
    )
    return resolved


def get_stations(lat: float, lon: float) -> list[str]:
    key = (round(lat, 4), round(lon, 4))
    if key in stations_cache:
        logger.info("stations cache hit lat=%s lon=%s", key[0], key[1])
        return stations_cache[key]

    logger.info("stations cache miss lat=%s lon=%s", key[0], key[1])

    response = session.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    stations_url = response.json()["properties"]["observationStations"]

    response = session.get(stations_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise ValueError("No observation stations found near this location.")
    station_ids = [f["properties"]["stationIdentifier"] for f in features]

    logger.info(
        "stations resolved lat=%s lon=%s count=%d first=%s",
        key[0],
        key[1],
        len(station_ids),
        station_ids[0],
    )

    stations_cache[key] = station_ids
    return station_ids


def fetch_observation(query: str) -> dict[str, Any]:
    cache_key = query.strip().lower()
    if cache_key in observation_cache:
        logger.info("observation cache hit query=%r", query)
        return observation_cache[cache_key]

    logger.info("observation cache miss query=%r", query)

    city = geocode_us_city(query)
    station_ids = get_stations(city["lat"], city["lon"])

    for station_id in station_ids[:MAX_STATIONS_TO_TRY]:
        try:
            response = session.get(
                f"{NWS_API_BASE}/stations/{station_id}/observations/latest",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning(
                "station fetch error station_id=%s status=%s; trying next",
                station_id,
                status,
            )
            continue

        obs = response.json().get("properties", {})
        if (obs.get("temperature") or {}).get("value") is None:
            logger.info("station silent station_id=%s; trying next", station_id)
            continue

        logger.info("station reporting station_id=%s", station_id)
        shaped = shape_observation_response(city, obs)
        observation_cache[cache_key] = shaped
        return shaped

    raise ValueError(
        f"No NWS station near '{query}' is currently reporting observations."
    )


def shape_observation_response(city: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    temp_c        = (obs.get("temperature")       or {}).get("value")
    heat_index_c  = (obs.get("heatIndex")         or {}).get("value")
    wind_chill_c  = (obs.get("windChill")         or {}).get("value")
    wind_kmh      = (obs.get("windSpeed")         or {}).get("value")
    wind_deg      = (obs.get("windDirection")     or {}).get("value")
    humidity      = (obs.get("relativeHumidity")  or {}).get("value")

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


@app.route("/")
def index():
    logger.info("index requested")
    return render_template("index.html")


@app.route("/api/weather/conditions")
def weather_conditions():
    city = request.args.get("city", "").strip()

    if not city:
        logger.warning("weather lookup rejected: missing city parameter")
        return jsonify({"error": "Query parameter 'city' is required."}), 400

    logger.info("weather lookup requested city=%r", city)

    try:
        result = fetch_observation(city)
    except LookupError as exc:
        logger.warning("geocode failed city=%r error=%r", city, str(exc))
        return jsonify({"error": str(exc)}), 404
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        logger.warning("nws api error city=%r status_code=%s", city, status_code)
        return jsonify({"error": f"NWS returned HTTP {status_code} for '{city}'."}), 502
    except requests.RequestException as exc:
        logger.error("weather service unreachable city=%r error=%r", city, exc)
        return jsonify({"error": "Weather service is unreachable. Please try again."}), 503
    except ValueError as exc:
        logger.warning("weather lookup failed city=%r error=%r", city, str(exc))
        return jsonify({"error": str(exc)}), 502

    logger.info(
        "weather lookup succeeded city=%r location=%r temp=%s%s condition=%r humidity=%s wind=%s%s",
        city,
        result.get("location"),
        result.get("temperature"),
        result.get("temperature_unit"),
        result.get("condition"),
        result.get("humidity"),
        result.get("wind_speed"),
        result.get("wind_unit"),
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")),
        debug=False,
    )
