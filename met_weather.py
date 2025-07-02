import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

# Get weather XML from Met Éireann
def get_met_weather(location: str):
    coords = {
        "claremorris": (53.7236, -9.0045),
        "dublin": (53.3498, -6.2603),
        "galway": (53.2707, -9.0568),
    }

    loc_key = location.lower().split(",")[0].strip()
    lat, lon = coords.get(loc_key, (53.3498, -6.2603))  # Default to Dublin

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

# Optional test/debug block
if __name__ == "__main__":
    xml = get_met_weather("Claremorris, Ireland")
    parsed = parse_forecast_xml(xml)
    print(summarize_forecast_data(parsed))
