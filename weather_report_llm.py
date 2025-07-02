from dotenv import load_dotenv
from langchain_together import Together
from met_weather import get_met_weather, parse_forecast_xml, summarize_forecast_data_rich

load_dotenv()

llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=1024)
#llm = Together(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_tokens=1024)

# Query the LLM with a prompt
def query_llm(prompt: str) -> str:
    return llm.invoke(prompt)

# Main app loop
def main():
    coords = {
        "claremorris": (53.7236, -9.0045),
        "dublin": (53.3498, -6.2603),
        "galway": (53.2707, -9.0568),
    }
    location = "claremorris"
    loc_key = location.lower().split(",")[0].strip()
    lat, lon = coords.get(loc_key, (53.3498, -6.2603))  # Default to Dublin

    while True:
        user_input = input("Ask your farm assistant (or 'quit'): ").strip().lower()

        if user_input in ['quit', 'exit', 'q']:
            break
        elif user_input not in ['quit', 'exit', 'q', '', 'weather', 'forecast', 'current', 'report', 'general']:
            # Specific weather query
            print("SPECIFIC WEATHER QUERY")
            xml = get_met_weather(lat, lon)
            parsed = parse_forecast_xml(xml)
            weather_data = summarize_forecast_data_rich(parsed)
            prompt = f"""
            You are a helpful weather assistant with detailed forecast data below.

            Weather data:
            {weather_data}

            User question: "{user_input}"

            Answer the question clearly and concisely based ONLY on the data above.
            If the data does not contain the answer, say: "Sorry, I don't have that information."
            """
            response = query_llm(prompt)
            print("Assistant:", response)
        else:
            print("GENERAL WEATHER QUERY")
            # You could pass weather as context if needed
            xml = get_met_weather(lat, lon)
            parsed = parse_forecast_xml(xml)
            weather_data = summarize_forecast_data_rich(parsed)
            prompt = f"""
            Can you give a general weather report in the style of a tv presenter given the following weather data?
            
            Data:
            {weather_data}

            """
            #print("Prompt:", prompt)
            response = query_llm(prompt)
            print("Assistant:", response)
            #engine.say(response)
            #engine.runAndWait()

if __name__ == "__main__":
    main()