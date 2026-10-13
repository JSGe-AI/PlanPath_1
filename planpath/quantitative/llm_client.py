"""Minimal OpenAI-compatible client for local Qwen and Patho-R1 services."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import requests


_CONTEXT_LIMIT_ERROR = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"requested\s+(?P<output>\d+)\s+output tokens.*?"
    r"prompt contains at least\s+(?P<input>\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)
_CONTEXT_SAFETY_MARGIN = 32


def _context_error_message(response: requests.Response) -> str:
    """Extract an OpenAI/vLLM error message without assuming one envelope."""

    try:
        payload = response.json()
    except (TypeError, ValueError):
        return str(getattr(response, "text", "") or "")
    if not isinstance(payload, dict):
        return str(payload)
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    return str(payload.get("message") or payload)


def _context_adjustment(
    response: requests.Response, requested_max_tokens: int
) -> dict[str, int] | None:
    """Return a safe output budget for a vLLM context-length rejection."""

    if response.status_code != 400:
        return None
    match = _CONTEXT_LIMIT_ERROR.search(_context_error_message(response))
    if match is None:
        return None
    context_window = int(match.group("context"))
    input_tokens = int(match.group("input"))
    effective = min(
        requested_max_tokens,
        context_window - input_tokens - _CONTEXT_SAFETY_MARGIN,
    )
    # A smaller allowance is unlikely to hold even the compact JSON schema
    # and would merely turn the context error into deterministic truncation.
    if effective < 128 or effective >= requested_max_tokens:
        return None
    return {
        "context_window": context_window,
        "input_tokens": input_tokens,
        "requested_output_tokens": requested_max_tokens,
        "effective_output_tokens": effective,
        "safety_margin_tokens": _CONTEXT_SAFETY_MARGIN,
    }


class OpenAICompatClient:
    def __init__(
        self, endpoint: str, model: str, timeout: float = 120.0, seed: int = 0
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.seed = seed
        # ``chat`` is called from the observer's worker pool.  A
        # requests.Session owns mutable connection-pool state and is not safe
        # to share concurrently across those threads.  In practice that left
        # completed local-vLLM responses in CLOSE-WAIT while the corresponding
        # future remained blocked.  Keep connection reuse within one worker,
        # but isolate sessions between workers.
        self._session_local = threading.local()

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session

    @property
    def session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = self._new_session()
            self._session_local.session = session
        return session

    @session.setter
    def session(self, value: requests.Session) -> None:
        # Preserve explicit injection for deterministic unit tests while
        # scoping the injected session to the current thread.
        self._session_local.session = value

    def _reset_session(self) -> None:
        """Discard a failed worker-local pool before retrying the request."""

        session = getattr(self._session_local, "session", None)
        if session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                close()
            del self._session_local.session

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.endpoint}/models", timeout=30)
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_error: Exception | None = None
        context_adjustments: list[dict[str, int]] = []
        for retry in range(3):
            try:
                response = self.session.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                adjustment = _context_adjustment(response, int(payload["max_tokens"]))
                if adjustment is not None:
                    context_adjustments.append(adjustment)
                    payload["max_tokens"] = adjustment["effective_output_tokens"]
                    continue
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                return {
                    "content": str(message.get("content") or "").strip(),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {}),
                    "retry_count": retry,
                    "context_adjustments": context_adjustments,
                }
            except Exception as error:  # pragma: no cover - network path
                last_error = error
                self._reset_session()
                if retry < 2:
                    time.sleep(2**retry)
        raise RuntimeError(f"request failed after three attempts: {last_error}")
