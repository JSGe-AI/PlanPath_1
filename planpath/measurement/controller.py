"""Core routing guard for the r24 quantitative-only extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .quantification import QuantificationProfile, quantitative_profile


R24_ROUTE_VERSION = "r24-explicit-quantitative-route-v1"


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["r23_5", "r24_quantitative"]
    profile: QuantificationProfile | None
    reason: str


class R24CoreController:
    """Choose r24 only for explicit measurements; preserve r23_5 otherwise."""

    def route(self, question: dict[str, Any]) -> RouteDecision:
        profile = quantitative_profile(question)
        if profile is None:
            return RouteDecision(
                route="r23_5",
                profile=None,
                reason="no explicit percentage, density, distance, or focus-count request",
            )
        return RouteDecision(
            route="r24_quantitative",
            profile=profile,
            reason=f"explicit quantitative task: {profile.kind}",
        )
