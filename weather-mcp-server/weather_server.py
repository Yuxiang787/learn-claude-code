#!/usr/bin/env python3
"""
Weather Forecast MCP Server

A Model Context Protocol server that provides weather forecast tools.
Uses Open-Meteo API (free, no API key required).
"""

import asyncio
from datetime import datetime
from typing import Any

import httpx
from mcp.server import FastMCP

# Create FastMCP server instance
mcp = FastMCP("weather-forecast-server")

# Open-Meteo API base URLs
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


async def get_coordinates(location: str) -> dict[str, Any]:
    """Get coordinates for a location name."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("results"):
            raise ValueError(f"Location '{location}' not found")
        
        result = data["results"][0]
        return {
            "name": result.get("name", location),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "country": result.get("country", ""),
            "admin1": result.get("admin1", ""),  # State/province
        }


async def fetch_weather_data(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    hourly: bool = False
) -> dict[str, Any]:
    """Fetch weather data from Open-Meteo API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max",
        "timezone": "auto",
        "forecast_days": forecast_days,
    }
    
    if hourly:
        params["hourly"] = "temperature_2m,relative_humidity_2m,weather_code,precipitation_probability,precipitation"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(WEATHER_URL, params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()


def weather_code_description(code: int) -> str:
    """Convert WMO weather code to description."""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return weather_codes.get(code, f"Unknown ({code})")


def wind_direction(degrees: int) -> str:
    """Convert wind direction degrees to compass direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


def format_temperature(celsius: float) -> str:
    """Format temperature with both Celsius and Fahrenheit."""
    fahrenheit = celsius * 9/5 + 32
    return f"{celsius:.1f}°C ({fahrenheit:.1f}°F)"


# Define MCP tools

@mcp.tool()
async def get_current_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: City name or location (e.g., "London", "New York", "Tokyo")
    """
    try:
        # Get coordinates
        geo_data = await get_coordinates(location)
        lat, lon = geo_data["latitude"], geo_data["longitude"]
        
        # Fetch weather
        weather_data = await fetch_weather_data(lat, lon)
        
        current = weather_data.get("current", {})
        
        # Build response
        location_name = f"{geo_data['name']}"
        if geo_data.get("admin1"):
            location_name += f", {geo_data['admin1']}"
        if geo_data.get("country"):
            location_name += f", {geo_data['country']}"
        
        weather_code = current.get("weather_code", 0)
        temp = current.get("temperature_2m", 0)
        feels_like = current.get("apparent_temperature", temp)
        humidity = current.get("relative_humidity_2m", 0)
        wind_speed = current.get("wind_speed_10m", 0)
        wind_dir = current.get("wind_direction_10m", 0)
        pressure = current.get("pressure_msl", 0)
        cloud_cover = current.get("cloud_cover", 0)
        precipitation = current.get("precipitation", 0)
        
        result = f"""🌤️ Current Weather for {location_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Coordinates: {lat:.4f}°, {lon:.4f}°

🌡️ Temperature: {format_temperature(temp)}
🤔 Feels Like: {format_temperature(feels_like)}

☁️ Conditions: {weather_code_description(weather_code)}
🌧️ Precipitation: {precipitation:.1f} mm
☁️ Cloud Cover: {cloud_cover}%

💨 Wind: {wind_speed:.1f} km/h ({wind_speed * 0.621371:.1f} mph) from {wind_direction(wind_dir)} ({wind_dir}°)
💧 Humidity: {humidity}%
🌡️ Pressure: {pressure:.1f} hPa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return result
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
async def get_weather_forecast(location: str, days: int = 7) -> str:
    """Get weather forecast for a location for up to 7 days.

    Args:
        location: City name or location (e.g., "London", "New York", "Tokyo")
        days: Number of days to forecast (1-7, default 7)
    """
    try:
        days = min(max(1, days), 7)  # Clamp between 1 and 7
        
        # Get coordinates
        geo_data = await get_coordinates(location)
        lat, lon = geo_data["latitude"], geo_data["longitude"]
        
        # Fetch weather
        weather_data = await fetch_weather_data(lat, lon, forecast_days=days)
        
        daily = weather_data.get("daily", {})
        
        # Build response
        location_name = f"{geo_data['name']}"
        if geo_data.get("admin1"):
            location_name += f", {geo_data['admin1']}"
        if geo_data.get("country"):
            location_name += f", {geo_data['country']}"
        
        result = f"📅 {days}-Day Weather Forecast for {location_name}\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        weather_codes = daily.get("weather_code", [])
        precipitation = daily.get("precipitation_sum", [])
        wind_max = daily.get("wind_speed_10m_max", [])
        uv_index = daily.get("uv_index_max", [])
        
        for i, date in enumerate(dates):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            today_marker = " (Today)" if i == 0 else ""
            
            result += f"📆 {date} ({day_name}){today_marker}\n"
            result += f"   ┌─ 🌡️ {format_temperature(max_temps[i])} / {format_temperature(min_temps[i])}\n"
            result += f"   ├─ ☁️ {weather_code_description(weather_codes[i])}\n"
            result += f"   ├─ 🌧️ Precipitation: {precipitation[i]:.1f} mm\n"
            result += f"   ├─ 💨 Wind: {wind_max[i]:.1f} km/h\n"
            result += f"   └─ ☀️ UV Index: {uv_index[i]:.1f}\n\n"
        
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        return result
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
async def get_hourly_forecast(location: str, hours: int = 24) -> str:
    """Get hourly weather forecast for a location.

    Args:
        location: City name or location (e.g., "London", "New York", "Tokyo")
        hours: Number of hours to forecast (1-48, default 24)
    """
    try:
        hours = min(max(1, hours), 48)  # Clamp between 1 and 48
        
        # Get coordinates
        geo_data = await get_coordinates(location)
        lat, lon = geo_data["latitude"], geo_data["longitude"]
        
        # Fetch weather with hourly data
        weather_data = await fetch_weather_data(lat, lon, forecast_days=2, hourly=True)
        
        hourly = weather_data.get("hourly", {})
        
        # Build response
        location_name = f"{geo_data['name']}"
        if geo_data.get("admin1"):
            location_name += f", {geo_data['admin1']}"
        if geo_data.get("country"):
            location_name += f", {geo_data['country']}"
        
        result = f"🕐 {hours}-Hour Forecast for {location_name}\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        times = hourly.get("time", [])[:hours]
        temps = hourly.get("temperature_2m", [])[:hours]
        weather_codes = hourly.get("weather_code", [])[:hours]
        precip_prob = hourly.get("precipitation_probability", [])[:hours]
        humidity = hourly.get("relative_humidity_2m", [])[:hours]
        
        for i, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            hour_str = dt.strftime("%Y-%m-%d %H:%M")
            
            # Weather emoji based on conditions
            code = weather_codes[i]
            if code == 0:
                emoji = "☀️"
            elif code in [1, 2]:
                emoji = "🌤️"
            elif code == 3:
                emoji = "☁️"
            elif code in [45, 48]:
                emoji = "🌫️"
            elif 51 <= code <= 67:
                emoji = "🌧️"
            elif 71 <= code <= 77:
                emoji = "🌨️"
            elif 80 <= code <= 82:
                emoji = "🌦️"
            elif 85 <= code <= 86:
                emoji = "🌨️"
            else:
                emoji = "⛈️"
            
            result += f"{hour_str} | {emoji} {format_temperature(temps[i])} | {weather_code_description(code)[:15]:<15} | 💧 {humidity[i]:.0f}% | 🌧️ {precip_prob[i]:.0f}%\n"
        
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        return result
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
async def get_detailed_weather(location: str) -> str:
    """Get detailed weather analysis with comfort index and advisories.

    Args:
        location: City name or location (e.g., "London", "New York", "Tokyo")
    """
    try:
        # Get coordinates
        geo_data = await get_coordinates(location)
        lat, lon = geo_data["latitude"], geo_data["longitude"]
        
        # Fetch weather
        weather_data = await fetch_weather_data(lat, lon)
        
        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})
        
        location_name = f"{geo_data['name']}"
        if geo_data.get("admin1"):
            location_name += f", {geo_data['admin1']}"
        if geo_data.get("country"):
            location_name += f", {geo_data['country']}"
        
        # Calculate comfort index
        temp = current.get("temperature_2m", 20)
        humidity = current.get("relative_humidity_2m", 50)
        wind = current.get("wind_speed_10m", 0)
        
        # Simple comfort assessment
        if 18 <= temp <= 24 and humidity < 60:
            comfort = "😊 Comfortable"
        elif temp > 30:
            comfort = "🥵 Hot"
        elif temp < 10:
            comfort = "🥶 Cold"
        elif humidity > 80:
            comfort = "😰 Humid"
        else:
            comfort = "🙂 Moderate"
        
        # UV Index from daily
        uv_index = daily.get("uv_index_max", [0])[0] if daily.get("uv_index_max") else 0
        if uv_index < 3:
            uv_level = "🟢 Low"
        elif uv_index < 6:
            uv_level = "🟡 Moderate"
        elif uv_index < 8:
            uv_level = "🟠 High"
        elif uv_index < 11:
            uv_level = "🔴 Very High"
        else:
            uv_level = "🟣 Extreme"
        
        result = f"""🔬 Detailed Weather Analysis for {location_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Location: {geo_data['name']} ({lat:.4f}°, {lon:.4f}°)
🌡️ Current Temperature: {format_temperature(current.get('temperature_2m', 0))}
🤔 Feels Like: {format_temperature(current.get('apparent_temperature', 0))}

📊 Conditions Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☁️ Cloud Cover: {current.get('cloud_cover', 0)}%
💧 Humidity: {humidity}%
💨 Wind Speed: {wind:.1f} km/h
🧭 Wind Direction: {wind_direction(current.get('wind_direction_10m', 0))} ({current.get('wind_direction_10m', 0)}°)
🌡️ Pressure: {current.get('pressure_msl', 1013):.1f} hPa
🌧️ Precipitation: {current.get('precipitation', 0):.1f} mm

☀️ UV Index: {uv_index:.1f} ({uv_level})
😊 Comfort Level: {comfort}

⚠️ Weather Advisory:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Add weather advisory based on conditions
        advisories = []
        if uv_index >= 6:
            advisories.append("• Wear sunscreen and protective clothing")
        if wind > 40:
            advisories.append("• Strong winds expected - secure loose items")
        if temp > 35:
            advisories.append("• Extreme heat - stay hydrated and avoid sun exposure")
        elif temp < 0:
            advisories.append("• Freezing temperatures - bundle up!")
        if humidity > 90:
            advisories.append("• Very high humidity - stay cool and hydrated")
        if current.get('precipitation', 0) > 0:
            advisories.append("• Precipitation occurring - bring an umbrella")
        
        if advisories:
            result += "\n".join(advisories) + "\n"
        else:
            result += "No special advisories - enjoy your day! ✨\n"
        
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        return result
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
async def search_location(query: str) -> str:
    """Search for a location and get its coordinates.

    Args:
        query: Location name to search for
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GEOCODING_URL,
                params={"name": query, "count": 5, "language": "en", "format": "json"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                return f"No locations found for '{query}'"
            
            result = f"🔍 Search Results for '{query}'\n"
            result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for i, loc in enumerate(data["results"], 1):
                name = loc.get("name", "Unknown")
                admin = loc.get("admin1", "")
                country = loc.get("country", "")
                lat = loc.get("latitude", 0)
                lon = loc.get("longitude", 0)
                pop = loc.get("population", 0)
                
                location_line = f"{i}. {name}"
                if admin:
                    location_line += f", {admin}"
                if country:
                    location_line += f", {country}"
                
                result += f"{location_line}\n"
                result += f"   📍 Coordinates: {lat:.4f}°, {lon:.4f}°\n"
                if pop:
                    result += f"   👥 Population: {pop:,}\n"
                result += "\n"
            
            result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            return result
            
    except httpx.HTTPError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    mcp.run()