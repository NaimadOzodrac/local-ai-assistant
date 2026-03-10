import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"


def ask_llm(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 80
            }
        }
    )

    data = response.json()

    return data["response"]

def stream_llm(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": 80
            }
        },
        stream=True
    )

    for line in response.iter_lines():

        if line:

            data = json.loads(line)

            token = data.get("response", "")

            print(token, end="", flush=True)

            if data.get("done", False):
                break