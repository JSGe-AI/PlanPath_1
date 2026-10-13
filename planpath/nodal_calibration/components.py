"""Route-local efficiency adapters for PlanPath r28."""

from __future__ import annotations

from typing import Any, Literal

from planpath.evidence_fusion.routing import component_route
from planpath.depth_calibration.components import (
    R27HistoPLUSAnalyzer,
    R27ROISelector,
    RouteBudget,
    route_budget as r27_route_budget,
)
from planpath.core.plip import SelectedROI
from planpath.core.schemas import EvidencePlan

def route_budget(route: str, clinical_domain: str = "") -> RouteBudget:
    """Keep r27 except for the validated r23 gastric nodal coverage."""

    if route == "nodal" and "gastric" in clinical_domain.casefold():
        # The paired r23_5_93/r27 audit localized the Yes/No regression to
        # regional-node questions.  Restore the r23 coverage only on this
        # route; the stricter r28 morphology gate controls false positives.
        return RouteBudget(12, 16, 8, "r23_gastric_nodal_recall")
    return r27_route_budget(route, clinical_domain)


class R28ROISelector(R27ROISelector):
    """Restore r23 nodal coverage without changing unrelated route budgets."""

    algorithm_version = "r28-full-gastric-nodal-recall-v2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics["r28_nodal_full_recall_events"] = 0
        self.metrics["r28_nodal_full_recall_extra_rois"] = 0

    @staticmethod
    def _low_power_budget(route: str) -> int:
        return route_budget(route, R28ROISelector._domain()).low_power

    @staticmethod
    def _high_power_budget(route: str, magnification: str) -> int:
        budget = route_budget(route, R28ROISelector._domain())
        return budget.forty_x if magnification == "40x" else budget.twenty_x

    def select_high(
        self,
        slide: dict[str, Any],
        parents: list[Any],
        magnification: Literal["10x", "20x", "40x"],
        query: str,
        *,
        ensure_parent_diversity: bool = True,
        plan: EvidencePlan | None = None,
    ) -> list[SelectedROI]:
        route = component_route()
        domain = self._domain().casefold()
        full_recall = bool(
            plan is not None
            and route == "nodal"
            and "gastric" in domain
        )
        result = super().select_high(
            slide,
            parents,
            magnification,
            query,
            ensure_parent_diversity=ensure_parent_diversity,
            plan=plan,
        )
        if full_recall:
            baseline = 6 if magnification == "40x" else 10
            self.metrics["r28_nodal_full_recall_events"] += 1
            self.metrics["r28_nodal_full_recall_extra_rois"] += max(
                0,
                len(result) - baseline,
            )
        return result


class R28HistoPLUSAnalyzer(R27HistoPLUSAnalyzer):
    """Retain r27 measurements with an explicit r28 cache identity."""

    algorithm_version = "r28-full-nodal-routing-only-histoplus-v2"


__all__ = [
    "R28HistoPLUSAnalyzer",
    "R28ROISelector",
    "RouteBudget",
    "route_budget",
]
