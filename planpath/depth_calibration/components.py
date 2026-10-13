"""r27 component adapters retaining r26 hard efficiency bounds."""

from __future__ import annotations

from typing import Any, Literal

from planpath.efficiency.components import (
    R26HistoPLUSAnalyzer,
    R26ROISelector,
    RouteBudget,
    route_budget as r26_route_budget,
)
from planpath.evidence_fusion.routing import component_route
from planpath.core.plip import SelectedROI
from planpath.core.schemas import EvidencePlan

from .context import depth_priority_parent_ids


def route_budget(route: str, clinical_domain: str = "") -> RouteBudget:
    """Keep r26 costs except for a small nodal recall reserve."""

    if route == "nodal":
        return RouteBudget(8, 10, 6, "bounded_nodal_recall")
    return r26_route_budget(route, clinical_domain)


class R27ROISelector(R26ROISelector):
    """Use r27 budgets behind the r26 low/high hard-cap boundary."""

    algorithm_version = "r27-evidence-calibrated-budget-v2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics["r27_depth_candidate_expansion_uses"] = 0
        self.metrics["r27_depth_priority_reserves"] = 0

    @staticmethod
    def _low_power_budget(route: str) -> int:
        return route_budget(route, R27ROISelector._domain()).low_power

    @staticmethod
    def _high_power_budget(route: str, magnification: str) -> int:
        budget = route_budget(route, R27ROISelector._domain())
        return budget.forty_x if magnification == "40x" else budget.twenty_x

    def _depth_priority_cap(
        self,
        output: list[SelectedROI],
        target: int,
        priority_parents: tuple[str, ...],
    ) -> list[SelectedROI]:
        """Keep two children for one morphology-nominated depth interface."""

        target = min(max(0, int(target)), len(output))
        if len(output) <= target or target == 0:
            return output[:target]

        self.metrics["r26_high_budget_hard_cap_events"] += 1
        self.metrics["r26_high_budget_dropped_rois"] += len(output) - target
        by_parent: dict[str, list[int]] = {}
        for index, item in enumerate(output):
            parent = str(item.roi.parent_roi_id or f"__unparented_{index}")
            by_parent.setdefault(parent, []).append(index)
        for indices in by_parent.values():
            indices.sort(
                key=lambda index: self._high_priority(output[index], index),
                reverse=True,
            )

        chosen: list[int] = []
        for priority_index, parent in enumerate(priority_parents):
            quota = 2 if priority_index == 0 else 1
            for index in by_parent.get(parent, [])[:quota]:
                if index not in chosen:
                    chosen.append(index)
        reserved = len(chosen)

        representatives = sorted(
            (
                indices[0]
                for parent, indices in by_parent.items()
                if parent not in priority_parents and indices
            ),
            key=lambda index: self._high_priority(output[index], index),
            reverse=True,
        )
        chosen.extend(representatives[: max(0, target - len(chosen))])
        if len(chosen) < target:
            chosen_set = set(chosen)
            remaining = sorted(
                (index for index in range(len(output)) if index not in chosen_set),
                key=lambda index: self._high_priority(output[index], index),
                reverse=True,
            )
            chosen.extend(remaining[: target - len(chosen)])

        if reserved > 1:
            self.metrics["r27_depth_priority_reserves"] += reserved - 1
        return [output[index] for index in sorted(chosen[:target])]

    def _hard_cap_high_output(
        self,
        output: list[SelectedROI],
        target: int,
    ) -> list[SelectedROI]:
        priority = depth_priority_parent_ids()
        if component_route() == "depth_interface" and priority:
            return self._depth_priority_cap(output, target, priority)
        return super()._hard_cap_high_output(output, target)

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
        """Rerank two extra candidates, while retaining the same final cap.

        PLIP already embeds the full child grid before ranking.  Asking the
        inherited selector to expose two additional candidates therefore
        adds no Patho-R1 or HistoPLUS calls; the r26 final boundary still
        sends exactly eight 20x fields to those expensive components.
        """

        priority = depth_priority_parent_ids()
        is_priority_depth = bool(
            plan is not None
            and magnification == "20x"
            and component_route() == "depth_interface"
            and priority
        )
        previous = self.high_power_top_k
        if is_priority_depth:
            target = self._high_power_budget("depth_interface", magnification)
            self.high_power_top_k = max(previous, target + 2)
            self.metrics["r27_depth_candidate_expansion_uses"] += 1
        try:
            return super().select_high(
                slide,
                parents,
                magnification,
                query,
                ensure_parent_diversity=ensure_parent_diversity,
                plan=plan,
            )
        finally:
            self.high_power_top_k = previous


class R27HistoPLUSAnalyzer(R26HistoPLUSAnalyzer):
    """The same bounded r26 analyzer with an explicit r27 trace identity."""

    algorithm_version = "r27-evidence-calibrated-histoplus-v2"


__all__ = [
    "R27HistoPLUSAnalyzer",
    "R27ROISelector",
    "RouteBudget",
    "route_budget",
]
