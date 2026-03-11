import json
import time
from typing import Generator

import httpx

from config import MODEL, OLLAMA_URL, LLM_TIMEOUT, MAX_RETRIES, RETRY_BASE_WAIT, RETRY_BACKOFF_FACTOR
from rag.exceptions import LLMError
from rag.logger import get_logger

logger = get_logger(__name__)


def _make_llm_request(prompt: str, stream: bool = False) -> httpx.Response:
    """Make a synchronous request to the LLM with retry logic."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=LLM_TIMEOUT) as client:
                response = client.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "prompt": prompt,
                        "stream": stream,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 120
                        }
                    }
                )
                response.raise_for_status()
                return response
        except httpx.RequestError as e:
            last_exception = e
            
            if attempt == MAX_RETRIES:
                logger.error(f"LLM request failed after {MAX_RETRIES} retries")
                raise LLMError(f"Failed to connect to LLM after {MAX_RETRIES} retries: {e}") from e
            
            wait_time = RETRY_BASE_WAIT * (RETRY_BACKOFF_FACTOR ** attempt)
            logger.warning(
                f"LLM request attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
    
    raise LLMError(f"Failed to connect to LLM: {last_exception}") from last_exception


def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the full response."""
    response = _make_llm_request(prompt, stream=False)

    try:
        data = response.json()
        return data["response"]
    except (json.JSONDecodeError, KeyError) as e:
        raise LLMError(f"Invalid response from LLM: {e}") from e


def stream_llm(prompt: str) -> None:
    """Send a prompt to the LLM and stream tokens to stdout."""
    response = _make_llm_request(prompt, stream=True)

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


async def _make_llm_request_async(prompt: str, stream: bool = False) -> httpx.Response:
    """Make an async request to the LLM with retry logic."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "prompt": prompt,
                        "stream": stream,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 120
                        }
                    }
                )
                response.raise_for_status()
                return response
        except httpx.RequestError as e:
            last_exception = e
            
            if attempt == MAX_RETRIES:
                logger.error(f"LLM request failed after {MAX_RETRIES} retries")
                raise LLMError(f"Failed to connect to LLM after {MAX_RETRIES} retries: {e}") from e
            
            wait_time = RETRY_BASE_WAIT * (RETRY_BACKOFF_FACTOR ** attempt)
            logger.warning(
                f"LLM request attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
    
    raise LLMError(f"Failed to connect to LLM: {last_exception}") from last_exception


async def ask_llm_async(prompt: str) -> str:
    """Send a prompt to the LLM asynchronously and return the full response."""
    response = await _make_llm_request_async(prompt, stream=False)

    try:
        data = response.json()
        return data["response"]
    except (json.JSONDecodeError, KeyError) as e:
        raise LLMError(f"Invalid response from LLM: {e}") from e


async def stream_llm_async(prompt: str) -> None:
    """Send a prompt to the LLM asynchronously and stream tokens to stdout."""
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 120
                }
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        print(token, end="", flush=True)
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        raise LLMError(f"Invalid JSON in stream: {e}") from e
