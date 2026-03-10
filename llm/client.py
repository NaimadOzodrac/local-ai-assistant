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
                "temperature": 0.7
            }
        }
    )

    data = response.json()

    return data["response"]