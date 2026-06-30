"""
Traced Groq client.

Groq exposes an OpenAI-compatible chat-completions API on very fast LPU
hardware, which makes it ideal for real-time agent work and for LLM-as-judge
evals. This thin client calls Groq with the stdlib only (no `requests`/`openai`
dependency) and records each call as an AgentLens `llm` span with the real token
usage Groq returns.

    from agentlens import Tracer
    from agentlens.providers.groq import GroqClient

    tracer = Tracer()
    groq = GroqClient(tracer=tracer)          # reads GROQ_API_KEY

    with tracer.trace("chat"):
        reply = groq.complete("Say hello in 3 words.")
    print(reply.text, reply.total_tokens)

Get a free key at https://console.groq.com and `export GROQ_API_KEY=...`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..tracer import Tracer
from ..pricing import estimate_cost

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: Dict[str, Any]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GroqError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 tracer: Optional[Tracer] = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.tracer = tracer
        self.timeout = timeout

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise GroqError(
                "No Groq API key. Set GROQ_API_KEY or pass api_key=. "
                "Get a free key at https://console.groq.com"
            )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            GROQ_URL, data=data, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise GroqError(f"Groq HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise GroqError(f"Groq request failed: {exc.reason}") from exc

    def chat(self, messages: List[Dict[str, str]], *, model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 1024,
             span_name: str = "groq.chat", **extra: Any) -> Completion:
        """Call Groq chat completions, tracing the call if a tracer is active."""
        model = model or self.model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }

        def _run() -> Completion:
            body = self._post(payload)
            usage = body.get("usage", {}) or {}
            text = body["choices"][0]["message"]["content"]
            return Completion(
                text=text,
                model=model,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                raw=body,
            )

        # If there's an active trace, wrap the call in an llm span.
        if self.tracer is not None and self.tracer._current is not None:  # noqa: SLF001
            with self.tracer.llm(span_name, model=model) as span:
                span.set_input(messages=messages, temperature=temperature)
                comp = _run()
                span.record_tokens(
                    prompt=comp.prompt_tokens,
                    completion=comp.completion_tokens,
                    model=model,
                )
                span.set_output(text=comp.text)
                return comp
        return _run()

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 **kwargs: Any) -> Completion:
        """Convenience: single user prompt (+ optional system)."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
