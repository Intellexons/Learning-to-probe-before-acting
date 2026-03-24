import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama API with structured output support."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        schema_path: str = "configs/schema.json",
    ):
        self.base_url = base_url.rstrip("/")
        with open(schema_path) as f:
            self.schema = json.load(f)

    def check_health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def generate_code(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 900,
    ) -> dict:
        """Call /api/chat with structured output schema.

        Returns dict with keys: code, success, error, raw_response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": self.schema,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            resp = r.json()
            content = resp.get("message", {}).get("content", "")

            parsed = json.loads(content)
            code = parsed.get("code", "")
            if not code:
                return {
                    "code": "",
                    "success": False,
                    "error": "Empty code field in response",
                    "raw_response": content,
                }
            return {
                "code": code,
                "success": True,
                "error": None,
                "raw_response": content,
            }

        except requests.RequestException as e:
            return {
                "code": "",
                "success": False,
                "error": f"Request failed: {e}",
                "raw_response": "",
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "code": "",
                "success": False,
                "error": f"Failed to parse response: {e}",
                "raw_response": content if "content" in dir() else "",
            }

    def generate_batch(
        self,
        model: str,
        prompt: str,
        n_samples: int,
        **kwargs,
    ) -> list[dict]:
        """Generate n_samples code candidates for the same prompt."""
        results = []
        for i in range(n_samples):
            logger.info(f"  Generating sample {i + 1}/{n_samples}")
            result = self.generate_code(model, prompt, **kwargs)
            results.append(result)
            if not result["success"]:
                logger.warning(f"  Sample {i + 1} failed: {result['error']}")
        return results
