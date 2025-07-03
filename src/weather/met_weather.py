import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
import statistics

# Get weather XML from Met Éireann
def get_met_weather(lat: float, lon: float) -> str:

    url = f"http://openaccess.pf.api.met.ie/metno-wdb2ts/locationforecast?lat={lat};long={lon}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Parse XML forecast and merge entries
def parse_forecast_xml(xml_data: str):
    root = ET.fromstring(xml_data)
    forecasts = defaultdict(dict)

    for time_elem in root.findall(".//time[@datatype='forecast']"):
        time_to = time_elem.attrib.get("to")
        loc = time_elem.find("location")
        if loc is None:
            continue

        if (temp := loc.find("temperature")) is not None:
            forecasts[time_to]["temperature"] = temp.attrib.get("value")
        if (hum := loc.find("humidity")) is not None:
            forecasts[time_to]["humidity"] = hum.attrib.get("value")
        if (wd := loc.find("windDirection")) is not None:
            forecasts[time_to]["wind_direction"] = wd.attrib.get("name")
        if (ws := loc.find("windSpeed")) is not None:
            forecasts[time_to]["wind_speed"] = ws.attrib.get("mps")
        if (cloud := loc.find("cloudiness")) is not None:
            forecasts[time_to]["cloudiness"] = cloud.attrib.get("percent")
        if (precip := loc.find("precipitation")) is not None:
            forecasts[time_to]["precipitation"] = precip.attrib.get("value")
        if (sym := loc.find("symbol")) is not None:
            forecasts[time_to]["symbol"] = sym.attrib.get("id")

        forecasts[time_to]["from"] = time_elem.attrib.get("from")
        forecasts[time_to]["to"] = time_to

    return [dict(time=to, **data) for to, data in sorted(forecasts.items())]

# Format summary for LLM
def summarize_forecast_data(forecast_data: list):
    summary = "Weather forecast summary:\n"
    for entry in forecast_data:
        try:
            dt = datetime.fromisoformat(entry["from"].replace("Z", "+00:00"))
            time_str = dt.strftime('%a %H:%M')
        except Exception:
            time_str = entry.get("from", "Unknown time")

        temperature = entry.get("temperature", "N/A")
        cloudiness = entry.get("cloudiness", "N/A")
        precipitation = entry.get("precipitation", "N/A")

        summary += f"{time_str}: {temperature}°C, {cloudiness}% cloud, {precipitation}mm rain\n"

    return summary

def group_by_day(forecast_data):
    grouped = defaultdict(list)
    for entry in forecast_data:
        try:
            dt = datetime.fromisoformat(entry["from"].replace("Z", "+00:00"))
            day = dt.strftime("%A")
            grouped[day].append((dt, entry))
        except Exception:
            continue
    return grouped

def summarize_forecast_data_rich(forecast_data: list) -> str:
    grouped = defaultdict(list)

    # Group entries by day
    for entry in forecast_data:
        try:
            dt = datetime.fromisoformat(entry["from"].replace("Z", "+00:00"))
            day = dt.strftime("%A")
        except Exception:
            continue  # Skip malformed entries

        grouped[day].append(entry)

    # Build summary
    summary = "Structured weather forecast by day:\n"

    for day, entries in grouped.items():
        try:
            temps = [float(e.get("temperature", 0)) for e in entries]
            rain = [float(e.get("precipitation", 0)) for e in entries]
            clouds = [float(e.get("cloudiness", 0)) for e in entries]
            wind_speeds = [float(e.get("wind_speed", 0)) for e in entries]
            wind_dirs = [e.get("wind_direction", "N/A") for e in entries if e.get("wind_direction")]

            summary += f"\n{day}:\n"
            summary += f"Temp: {min(temps):.1f}°C to {max(temps):.1f}°C\n"
        except Exception:
            continue
