from dotenv import load_dotenv
from langchain_together import Together
import requests
import pyttsx3

# Load environment variables
load_dotenv()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Optional: adjust speaking speed

# Initialize Together LLM
llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=150)

# Query the LLM with a prompt
def query_llm(prompt: str) -> str:
    return llm.invoke(prompt)

# Get weather info from wttr.in
def get_weather(location):
    url = f"https://wttr.in/{location}?format=j1"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers).json()

    current = response["current_condition"][0]
    temp = current["temp_C"]
    condition = current["weatherDesc"][0]["value"]
    result = f"The current temperature in {location} is {temp}°C with {condition}."
    print("Weather Info:", result)
    return result


# Report weather using LLM + TTS
def report_weather(location):
    raw_weather = get_weather(location)
    prompt = f"Summarize this weather info for a farmer: {raw_weather}"
    summary = query_llm(prompt)

    print("Assistant:", summary)
    engine.say(summary)
    engine.runAndWait()

# Main app loop
def main():
    location = "Claremorris, Ireland"

    while True:
        user_input = input("Ask your farm assistant (or type 'weather' / 'quit'): ").strip().lower()

        if user_input in ['quit', 'exit']:
            break
        elif user_input == 'weather':
            report_weather(location)
        else:
            # You could pass weather as context if needed
            response = query_llm(user_input)
            print("Assistant:", response)
            engine.say(response)
            engine.runAndWait()

if __name__ == "__main__":
    main()
