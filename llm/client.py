import json
from typing import Generator

import requests

from config import MODEL, OLLAMA_URL, LLM_TIMEOUT
from rag.exceptions import LLMError


def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the full response."""
    try:
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
            },
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"Failed to connect to LLM: {e}") from e

    try:
        data = response.json()
        return data["response"]
    except (json.JSONDecodeError, KeyError) as e:
        raise LLMError(f"Invalid response from LLM: {e}") from e


def stream_llm(prompt: str) -> None:
    """Send a prompt to the LLM and stream tokens to stdout."""
    try:
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
            stream=True,
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"Failed to connect to LLM: {e}") from e

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                if data.get("done", False):
                    break
            except json.JSONDecodeError as e:
                raise LLMError(f"Invalid JSON in stream: {e}") from e