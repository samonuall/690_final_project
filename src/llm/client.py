import logging
import os

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Stateful OpenRouter chat client that maintains conversation history."""

    def __init__(self, model: str):
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set — add it to .env")
        self._api_key = api_key
        self.model = model
        self.messages: list[dict] = []
        # One entry per send() call: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        self.token_usage: list[dict] = []

    def add_system(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        resp = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "messages": self.messages},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": content})
        self.token_usage.append(data.get("usage", {}))
        logger.debug(
            "OpenRouter call: tokens_used=%s model=%s",
            data.get("usage", {}).get("total_tokens"), self.model
        )
        return content

    @property
    def cumulative_tokens(self) -> dict:
        return {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in self.token_usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in self.token_usage),
            "total_tokens": sum(u.get("total_tokens", 0) for u in self.token_usage),
        }
