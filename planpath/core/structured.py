"""Schema-validated JSON generation with concise repair attempts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)
SemanticValidator = Callable[[BaseModel], dict[str, str]]
CandidateNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


class StructuredGenerationError(ValueError):
    """A structured response remained invalid after bounded repairs."""

    def __init__(
        self,
        phase: str,
        last_error: str,
        audit: list[dict[str, Any]],
        last_response: str,
    ) -> None:
        super().__init__(f"{phase} generation failed after repairs: {last_error}")
        self.phase = phase
        self.last_error = last_error
        self.audit = audit
        self.last_response = last_response


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]: ...


@dataclass
class GenerationResult(Generic[ModelT]):
    value: ModelT
    audit: list[dict[str, Any]]


def extract_first_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (TypeError, json.JSONDecodeError):
        pass
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def salvage_top_level_fields(text: str) -> dict[str, Any]:
    """Recover individually valid fields from a truncated/malformed object."""

    decoder = json.JSONDecoder()
    recovered: dict[str, Any] = {}
    for match in re.finditer(r'"([^"\\]+)"\s*:', text):
        try:
            value, _ = decoder.raw_decode(text[match.end() :].lstrip())
        except (json.JSONDecodeError, ValueError):
            continue
        recovered.setdefault(match.group(1), value)
    return recovered


def select_schema_candidate(
    value: dict[str, Any], model: type[BaseModel]
) -> dict[str, Any]:
    """Find and filter the object that most closely matches ``model``.

    Local models occasionally echo the request payload or wrap the requested
    object.  Unknown keys must not poison later repair attempts.
    """

    allowed = set(model.model_fields)
    candidates: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            candidates.append(item)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    best = max(
        candidates,
        key=lambda item: (len(allowed & set(item)), -len(set(item) - allowed)),
        default={},
    )
    return {key: best[key] for key in allowed if key in best}


class StructuredGenerator:
    def __init__(self, client: ChatClient, max_repairs: int = 2) -> None:
        self.client = client
        self.max_repairs = max_repairs

    def generate(
        self,
        model: type[ModelT],
        messages: list[dict[str, Any]],
        max_tokens: int,
        phase: str,
        semantic_validator: SemanticValidator | None = None,
        candidate_normalizer: CandidateNormalizer | None = None,
    ) -> GenerationResult[ModelT]:
        current = list(messages)
        audit: list[dict[str, Any]] = []
        last_error = "invalid JSON"
        last_raw = ""
        for attempt in range(self.max_repairs + 1):
            response = self.client.chat(current, max_tokens)
            raw = str(response.get("content") or "")
            last_raw = raw
            audit_item = {
                "phase": phase,
                "attempt": attempt,
                "finish_reason": response.get("finish_reason"),
                "usage": response.get("usage", {}),
                "retry_count": response.get("retry_count", 0),
                "context_adjustments": response.get("context_adjustments", []),
                "response_chars": len(raw),
                "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }
            audit.append(audit_item)
            parsed = extract_first_json(raw) or salvage_top_level_fields(raw)
            if parsed:
                if candidate_normalizer is not None:
                    parsed = candidate_normalizer(parsed)
                candidate = select_schema_candidate(parsed, model)
                try:
                    value = model.model_validate(candidate)
                    semantic_errors = semantic_validator(value) if semantic_validator else {}
                    if not semantic_errors:
                        audit_item["validation"] = "accepted"
                        return GenerationResult(value=value, audit=audit)
                    last_error = json.dumps(
                        semantic_errors, ensure_ascii=False, default=str
                    )
                except ValidationError as error:
                    last_error = json.dumps(
                        error.errors(include_url=False),
                        ensure_ascii=False,
                        default=str,
                    )
            else:
                last_error = "no JSON object could be parsed"
            audit_item["validation"] = "rejected"
            audit_item["validation_error"] = last_error[:2000]
            audit_item["rejected_response_excerpt"] = raw[-2000:]
            if attempt < self.max_repairs:
                current = list(messages) + [
                    {"role": "assistant", "content": raw[-5000:]},
                    {
                        "role": "user",
                        "content": (
                            "The JSON above is invalid. Fix it and return exactly one object matching "
                            f"the schema. Validation errors: {last_error}"
                        ),
                    },
                ]
        raise StructuredGenerationError(
            phase=phase,
            last_error=last_error,
            audit=audit,
            last_response=last_raw[-5000:],
        )
