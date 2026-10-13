"""Process-local r27 focus state used only during one zoom transition."""

from __future__ import annotations

from contextvars import ContextVar


_depth_priority_parent_ids: ContextVar[tuple[str, ...]] = ContextVar(
    "r27_depth_priority_parent_ids",
    default=(),
)


def set_depth_priority_parent_ids(values: list[str]) -> None:
    _depth_priority_parent_ids.set(tuple(dict.fromkeys(str(value) for value in values)))


def depth_priority_parent_ids() -> tuple[str, ...]:
    return _depth_priority_parent_ids.get()


__all__ = ["depth_priority_parent_ids", "set_depth_priority_parent_ids"]
