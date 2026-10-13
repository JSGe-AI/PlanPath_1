"""Task-local context shared by the isolated r25 component adapters."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any


_CURRENT_QUESTION: ContextVar[dict[str, Any] | None] = ContextVar(
    "planpath.evidence_fusion_current_question", default=None
)


def current_question() -> dict[str, Any] | None:
    return _CURRENT_QUESTION.get()


def register_roi_stratum(roi_id: str, stratum: str) -> None:
    """Attach a navigation-only stratum to the current task-local ROI."""

    question = _CURRENT_QUESTION.get()
    if question is None:
        return
    registry = question.setdefault("_r25_roi_strata", {})
    if isinstance(registry, dict):
        registry[str(roi_id)] = str(stratum)


def roi_stratum(roi_id: str) -> str | None:
    question = _CURRENT_QUESTION.get() or {}
    registry = question.get("_r25_roi_strata", {})
    if not isinstance(registry, dict):
        return None
    value = registry.get(str(roi_id))
    return str(value) if value is not None else None


def set_current_question(question: dict[str, Any]) -> Token[dict[str, Any] | None]:
    return _CURRENT_QUESTION.set(question)


def reset_current_question(token: Token[dict[str, Any] | None]) -> None:
    _CURRENT_QUESTION.reset(token)
