# Weather Forecast MCP Server 🌤️

A Model Context Protocol (MCP) server that provides weather forecast tools using the Open-Meteo API (free, no API key required).

## Features

- **Current Weather**: Get real-time weather conditions for any location
- **7-Day Forecast**: Detailed daily forecasts with temperature, precipitation, wind, and UV index
- **Hourly Forecast**: Up to 48 hours of detailed hourly predictions
- **Detailed Analysis**: Comfort index, UV levels, and weather advisories
- **Location Search**: Find coordinates for any city or place

## Installation

### 1. Create and activate virtual environment

```bash
cd weather-mcp-server
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Add the server to your Claude MCP configuration file at `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "weather": {
      "command": "/Users/slumber/Downloads/learn-claude-code/weather-mcp-server/venv/bin/python",
      "args": ["/Users/slumber/Downloads/learn-claude-code/weather-mcp-server/weather_server.py"]
    }
  }
}
```

Or if using system Python:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python3",
      "args": ["/path/to/weather-mcp-server/weather_server.py"]
    }
  }
}
```

## Available Tools

### 1. `get_current_weather`
Get current weather conditions for a location.

**Parameters:**
- `location` (string): City name or location (e.g., "London", "New York", "Tokyo")

**Example:**
```
What's the current weather in Tokyo?
```

### 2. `get_weather_forecast`
Get weather forecast for up to 7 days.

**Parameters:**
- `location` (string): City name or location
- `days` (integer, optional): Number of days to forecast (1-7, default 7)

**Example:**
```
Give me a 5-day forecast for Paris
```

### 3. `get_hourly_forecast`
Get hourly weather forecast for up to 48 hours.

**Parameters:**
- `location` (string): City name or location
- `hours` (integer, optional): Number of hours to forecast (1-48, default 24)

**Example:**
```
Show me the hourly forecast for the next 12 hours in Sydney
```

### 4. `get_detailed_weather`
Get detailed weather analysis with comfort index and advisories.

**Parameters:**
- `location` (string): City name or location

**Example:**
```
Give me a detailed weather analysis for Denver
```

### 5. `search_location`
Search for a location and get its coordinates.

**Parameters:**
- `query` (string): Location name to search for

**Example:**
```
Search for locations named Springfield
```

## Testing

Test the server using the MCP Inspector:

```bash
npx @anthropics/mcp-inspector python3 weather_server.py
```

Or test directly:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 weather_server.py
```

## API Used

This server uses the [Open-Meteo API](https://open-meteo.com/), which is:
- Free to use
- No API key required
- No rate limiting for reasonable use

## Example Output

### Current Weather
```
🌤️ Current Weather for Tokyo, Japan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Coordinates: 35.6895°, 139.6917°

🌡️ Temperature: 22.5°C (72.5°F)
🤔 Feels Like: 23.0°C (73.4°F)

☁️ Conditions: Partly cloudy
🌧️ Precipitation: 0.0 mm
☁️ Cloud Cover: 35%

💨 Wind: 12.5 km/h (7.8 mph) from SE (135°)
💧 Humidity: 65%
🌡️ Pressure: 1013.2 hPa
```

### Forecast
```
📅 7-Day Weather Forecast for Tokyo, Japan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📆 2024-03-27 (Wednesday)
   ┌─ 🌡️ 25.0°C (77.0°F) / 18.0°C (64.4°F)
   ├─ ☁️ Partly cloudy
   ├─ 🌧️ Precipitation: 0.0 mm
   ├─ 💨 Wind: 15.2 km/h
   └─ ☀️ UV Index: 6.0
```

## License

MIT License