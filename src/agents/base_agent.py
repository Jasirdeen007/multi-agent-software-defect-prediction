from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from dotenv import load_dotenv
from groq import AsyncGroq, Groq

load_dotenv()

class BaseAuditAgent:
    """Base class providing async Groq API interaction with exponential backoff and JSON validation."""

    def __init__(
        self,
        name: str,
        model: str = "qwen/qwen3.8-27b",
        fallback_model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.name = name
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

        self.client = AsyncGroq(api_key=api_key)

    async def _generate_json_response(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Calls Groq with strict JSON output mode and exponential backoff retry logic."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        current_model = self.model
        import random

        while attempt < self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
                raw_text = response.choices[0].message.content.strip()
                parsed = json.loads(raw_text)
                return parsed

            except Exception as exc:
                attempt += 1
                error_str = str(exc)
                jitter = random.uniform(0.5, 1.5)
                wait_time = (2.0 ** attempt) + jitter

                if "429" in error_str or "rate_limit" in error_str.lower():
                    print(f"[{self.name}] Rate limit (429). Retrying in {wait_time:.1f}s (attempt {attempt}/{self.max_retries})...")
                    await asyncio.sleep(wait_time)
                elif "not found" in error_str.lower() or "does not exist" in error_str.lower():
                    print(f"[{self.name}] Model {current_model} unavailable, switching to fallback {self.fallback_model}...")
                    current_model = self.fallback_model
                    await asyncio.sleep(1.0)
                else:
                    if attempt >= self.max_retries:
                        raise RuntimeError(f"[{self.name}] Failed after {self.max_retries} attempts: {exc}")
                    await asyncio.sleep(wait_time)

        raise RuntimeError(f"[{self.name}] Failed to generate valid JSON response.")
