"""Pluggable LLM backends.

Retrieval is the same whichever model writes the answer, so the model sits
behind one small interface: give it a system prompt and a user prompt, get
back a stream of text. Everything above this file is provider-agnostic.

Three backends ship:

  ollama     runs on your own machine, no API key, no internet
  gemini     Google's free tier — a key, but a free one, and it works on
             hosting where a local model cannot run
  anthropic  Claude, paid

`get_provider()` picks one automatically: whatever `LLM_BACKEND` names, else
a running Ollama, else whichever API key is present.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class ProviderError(RuntimeError):
    """Raised when a backend cannot be reached or rejects the request."""


# ── Ollama: local, no key ───────────────────────────────────────────────
class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str = OLLAMA_HOST):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = host.rstrip("/")

    @property
    def label(self) -> str:
        return f"Ollama · {self.model} (local)"

    @staticmethod
    def running(host: str = OLLAMA_HOST) -> bool:
        try:
            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def installed_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": True,
                    # Low temperature: this is grounded extraction from supplied
                    # text, not creative writing. Invention is the failure mode.
                    "options": {"temperature": 0.2},
                },
                stream=True,
                timeout=120,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"cannot reach Ollama at {self.host}: {exc}") from exc

        if response.status_code == 404:
            have = self.installed_models()
            raise ProviderError(
                f"Ollama has no model called '{self.model}'. "
                f"Installed: {', '.join(have) if have else 'none'}. "
                f"Run:  ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise ProviderError(f"Ollama returned HTTP {response.status_code}")

        # Ollama streams newline-delimited JSON, one object per token batch.
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = payload.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if payload.get("done"):
                break


# ── Gemini: free tier, deployable ───────────────────────────────────────
class GeminiProvider:
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def label(self) -> str:
        return f"Gemini · {self.model}"

    def stream(self, system: str, user: str) -> Iterator[str]:
        url = f"{self.ENDPOINT}/{self.model}:streamGenerateContent"
        try:
            response = requests.post(
                url,
                params={"alt": "sse", "key": self.api_key},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                stream=True,
                timeout=120,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"cannot reach Gemini: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:300]
            raise ProviderError(
                f"Gemini returned HTTP {response.status_code}. "
                f"If it names the model, set GEMINI_MODEL to one your key can "
                f"use. Detail: {detail}"
            )

        # Server-sent events: lines of "data: {json}".
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if not body or body == "[DONE]":
                continue
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            for candidate in payload.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text = part.get("text", "")
                    if text:
                        yield text


# ── Anthropic: paid ─────────────────────────────────────────────────────
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5"):
        self.model = model

    @property
    def label(self) -> str:
        return f"Claude · {self.model}"

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "the anthropic package is not installed — pip install anthropic"
            ) from exc

        client = anthropic.Anthropic()
        with client.beta.messages.stream(
            model=self.model,
            max_tokens=4096,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
            if stream.get_final_message().stop_reason == "refusal":
                yield "\n\n_The model declined to answer this one._"


# ── selection ───────────────────────────────────────────────────────────
def get_provider():
    """Pick a backend, preferring an explicit choice, then free, then paid."""
    choice = (os.environ.get("LLM_BACKEND") or "").strip().lower()

    if choice == "ollama":
        return OllamaProvider()
    if choice == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError("LLM_BACKEND=gemini but GEMINI_API_KEY is not set")
        return GeminiProvider(key)
    if choice == "anthropic":
        return AnthropicProvider()
    if choice:
        raise ProviderError(f"unknown LLM_BACKEND '{choice}'")

    # Auto: a local model costs nothing per question, so it wins when running.
    if OllamaProvider.running():
        return OllamaProvider()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider(os.environ["GEMINI_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()

    raise ProviderError(
        "No LLM backend available. Either start Ollama (ollama serve), or set "
        "GEMINI_API_KEY, or set ANTHROPIC_API_KEY."
    )
