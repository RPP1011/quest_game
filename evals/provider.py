"""Promptfoo provider that calls our local llama-server via httpx."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config", {})
    base_url = config.get("base_url", "http://127.0.0.1:8082")
    model = config.get("model", None)
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 4000)

    user_prompt = context.get("vars", {}).get("user_prompt", "")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        with httpx.Client(timeout=300) as client:
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")

        return {
            "output": content,
            "tokenUsage": {
                "prompt": data.get("usage", {}).get("prompt_tokens", 0),
                "completion": data.get("usage", {}).get("completion_tokens", 0),
                "total": data.get("usage", {}).get("total_tokens", 0),
            },
        }
    except Exception as e:
        return {"error": str(e)}
