from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from groq import AsyncGroq, Groq

load_dotenv()

_MODEL_LOCKS: dict[str, asyncio.Lock] = {}
_MODEL_LAST_REQUEST_AT: dict[str, float] = {}

class BaseAuditAgent:
    """Base class providing async Groq API interaction with retry and token accounting."""

    def __init__(
        self,
        name: str,
        model: str = "qwen/qwen3.8-27b",
        fallback_model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        max_retries: int = 3,
        request_timeout_seconds: float = 60.0,
        min_request_interval_seconds: float = 15.0,
        max_completion_tokens: int = 256,
        provider: str = "groq",
        api_key_env: str | None = None,
    ):
        self.name = name
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_completion_tokens = max_completion_tokens
        self.provider = provider.lower().strip()
        self.api_key_env = api_key_env or ("GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY")

        if self.provider == "groq":
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise ValueError(f"{self.api_key_env} not found in environment. Set it in .env.")
            # Disable SDK-level retries; the bounded, visible retry loop below owns them.
            self.client = AsyncGroq(api_key=api_key, timeout=request_timeout_seconds, max_retries=0)
        elif self.provider == "gemini":
            if not os.getenv(self.api_key_env):
                raise ValueError(f"{self.api_key_env} not found in environment. Set it in .env to use Gemini.")
            self.client = None
        else:
            raise ValueError(f"Unsupported LLM provider: {provider!r}; expected 'groq' or 'gemini'.")
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_by_model: dict[str, dict[str, int]] = {}

    async def _generate_json_response(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Calls the configured provider with JSON output and bounded retries."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        current_model = self.model
        while attempt < self.max_retries:
            try:
                slot_key = f"{self.provider}:{self.api_key_env}:{current_model}"
                lock = _MODEL_LOCKS.setdefault(slot_key, asyncio.Lock())
                async with lock:
                    elapsed = time.monotonic() - _MODEL_LAST_REQUEST_AT.get(slot_key, 0.0)
                    wait_for_slot = self.min_request_interval_seconds - elapsed
                    if wait_for_slot > 0:
                        await asyncio.sleep(wait_for_slot)
                    if self.provider == "groq":
                        response = await self.client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            temperature=self.temperature,
                            response_format={"type": "json_object"},
                            max_tokens=self.max_completion_tokens,
                        )
                        raw_text = response.choices[0].message.content.strip()
                        usage = getattr(response, "usage", None)
                        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    else:
                        response, raw_text, prompt_tokens, completion_tokens = await asyncio.to_thread(
                            self._gemini_generate, current_model, system_prompt, user_prompt
                        )
                    _MODEL_LAST_REQUEST_AT[slot_key] = time.monotonic()
                if prompt_tokens or completion_tokens:
                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
                    model_usage = self.usage_by_model.setdefault(current_model, {"prompt_tokens": 0, "completion_tokens": 0})
                    model_usage["prompt_tokens"] += prompt_tokens
                    model_usage["completion_tokens"] += completion_tokens
                parsed = json.loads(raw_text)
                return parsed

            except Exception as exc:
                attempt += 1
                error_str = str(exc)
                error_lower = error_str.lower()
                if any(marker in error_lower for marker in ("per day", "daily limit", "tokens per day", "requests per day")):
                    raise RuntimeError(f"[{self.name}] Provider daily quota reached; stop and resume after quota reset: {exc}") from exc
                if getattr(exc, "status_code", None) == 400 and "rate_limit" not in error_lower:
                    raise RuntimeError(f"[{self.name}] Provider rejected the request; not retrying a non-transient 400: {exc}") from exc
                if self.provider == "groq" and ("not found" in error_lower or "does not exist" in error_lower):
                    print(f"[{self.name}] Model {current_model} unavailable, switching to fallback {self.fallback_model}...")
                    current_model = self.fallback_model
                elif attempt >= self.max_retries:
                    raise RuntimeError(f"[{self.name}] Failed after {attempt} attempt(s): {exc}") from exc

                retry_after = None
                response = getattr(exc, "response", None)
                if response is not None:
                    retry_after = getattr(response, "headers", {}).get("retry-after")
                if not retry_after:
                    match = re.search(r"try again in ([0-9.]+)s", error_lower)
                    retry_after = match.group(1) if match else None
                wait_time = min(float(retry_after), 60.0) if retry_after else min(2 ** (attempt - 1) + random.uniform(0, 1), 10.0)
                print(f"[{self.name}] API request failed; retrying in {wait_time:.1f}s ({attempt}/{self.max_retries}): {exc}")
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"[{self.name}] Failed to generate valid JSON response.")

    def _gemini_generate(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> tuple[dict[str, Any], str, int, int]:
        """Calls Gemini's generateContent REST API using the standard library."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"{self.api_key_env} not found in environment.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_completion_tokens,
                "responseMimeType": "application/json",
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

        candidates = body.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        raw_text = "".join(str(part.get("text", "")) for part in parts).strip()
        if not raw_text:
            raise ValueError("Gemini returned no candidate text.")
        usage = body.get("usageMetadata") or {}
        return (
            body,
            raw_text,
            int(usage.get("promptTokenCount", 0) or 0),
            int(usage.get("candidatesTokenCount", 0) or 0),
        )
