import requests
from config import HF_API_KEY

MODEL_URL = "https://router.huggingface.co/hf-inference/models/google/pegasus-xsum"


def summarize(text):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 250,
            "min_length": 100,
        }
    }
    
    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]['summary_text']
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    print("Generating summary...")
    summary = summarize(text)
    print(summary)