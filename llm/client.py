import json
from typing import Generator

import requests

from config import MODEL, OLLAMA_URL


def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the full response."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 120
            }
        }
    )

    data = response.json()

    return data["response"]


def stream_llm(prompt: str) -> None:
    """Send a prompt to the LLM and stream tokens to stdout."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 120
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