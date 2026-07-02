"""Print OpenRouter API key usage/limit information."""
from __future__ import annotations
import json
import requests
from config.settings import settings


def main() -> None:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    response = requests.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    main()
