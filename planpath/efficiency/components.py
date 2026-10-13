"""Efficiency adapters for r26.

r26 deliberately changes only the number of fields inspected at each scale.
The r25 Tissue Map strata, HistoPLUS hotspot logic, Patho-R1 transcription,
and evidence rules are inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from planpath.evidence_fusion.components import R25HistoPLUSAnalyzer, R25ROISelector
from planpath.evidence_fusion.context import current_question
from planpath.evidence_fusion.routing import component_route
from planpath.core.plip import SelectedROI
from planpath.core.schemas import EvidencePlan


@dataclass(frozen=True)
class RouteBudget:
    """Maximum first-pass Patho-R1 fields for one slide and one route."""

    low_power: int
    twenty_x: int
    forty_x: int
    precision_guard: str


def route_budget(route: str, clinical_domain: str = "") -> RouteBudget:
    """Return the r26 route-aware observation budget.

    The common gastric pN and pT routes use eight fields at 5x and 20x.  Rare
    single-focus relations (PNI/LVI and renal boundary involvement) retain a
    wider search.  This is intentionally not a uniform budget reduction.
    """

    domain = clinical_domain.casefold()
    if route in {"vascular_interface", "neural_interface"}:
        return RouteBudget(10, 12, 6, "microinvasion_recall")
    if route == "renal_boundary":
        return RouteBudget(10, 12, 6, "outer_boundary_recall")
    if route in {"nodal", "depth_interface"}:
        return RouteBudget(8, 8, 6, "all_candidate_slides")
    if route in {"margin_interface", "collecting_system_interface"}:
        return RouteBudget(8, 8, 6, "interface_recall")
    if route == "cellular_extremum":
        return RouteBudget(8, 8, 6, "extremum_recall")
    if route in {
        "tissue_composition",
        "tumor_bed_response",
        "tumor_distribution",
    }:
        return RouteBudget(6, 6, 5, "stratified_tissue_recall")
    if route == "tumor_phenotype":
        count = 6 if "gastric" in domain else 5
        return RouteBudget(count, count, 5, "phenotype_strata")
    return RouteBudget(5, 5, 5, "r23_5_baseline")


class R26ROISelector(R25ROISelector):
    """Use a route-aware first pass while retaining r25 evidence selection."""

    algorithm_version = "r26-route-aware-budget-v2-hard-cap"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics["r26_low_budget_hard_cap_events"] = 0
        self.metrics["r26_low_budget_dropped_rois"] = 0
        self.metrics["r26_high_budget_hard_cap_events"] = 0
        self.metrics["r26_high_budget_dropped_rois"] = 0

    @staticmethod
    def _domain() -> str:
        return str((current_question() or {}).get("clinical_domain") or "")

    @staticmethod
    def _low_power_budget(route: str) -> int:
        return route_budget(route, R26ROISelector._domain()).low_power

    @staticmethod
    def _high_power_budget(route: str, magnification: str) -> int:
        budget = route_budget(route, R26ROISelector._domain())
        return budget.forty_x if magnification == "40x" else budget.twenty_x

    def _hybrid_indices(
        self,
        coords: Any,
        plip_scores: Any,
        fusion_scores: Any,
        tissue_scores: Any,
        interface_scores: Any,
        heterogeneity_scores: Any,
        count: int,
        **kwargs: Any,
    ) -> tuple[list[int], dict[int, str]]:
        """Make the low-power route budget an invariant.

        The r25 portfolio can already be full after PLIP anchors and one
        member from each Tissue Map stratum.  Its routed-fill loop then adds
        one more item before checking equality, turning a six-field gastric
        phenotype budget into seven or more fields.  r26 owns the efficiency
        policy, so it applies a final bound without changing r25 source.
        """

        target = min(max(0, int(count)), len(coords))
        selected, routes = super()._hybrid_indices(
            coords,
            plip_scores,
            fusion_scores,
            tissue_scores,
            interface_scores,
            heterogeneity_scores,
            count,
            **kwargs,
        )
        if len(selected) > target:
            self.metrics["r26_low_budget_hard_cap_events"] += 1
            self.metrics["r26_low_budget_dropped_rois"] += len(selected) - target
        bounded = list(selected[:target])
        return bounded, {
            index: routes[index] for index in bounded if index in routes
        }

    @staticmethod
    def _high_priority(item: SelectedROI, original_index: int) -> tuple[Any, ...]:
        """Rank a candidate for the bounded high-power portfolio."""

        navigation = item.roi.tissue_navigation
        selection_route = str(
            navigation.selection_route if navigation is not None else ""
        ).casefold()
        if "outer_interface" in selection_route or "adipose_interface" in selection_route:
            route_priority = 4
        elif "tissue_stratum" in selection_route:
            route_priority = 3
        elif "tissue_interface" in selection_route:
            route_priority = 2
        elif navigation is not None and float(navigation.interface_score) > 0.0:
            route_priority = 1
        else:
            route_priority = 0
        return (
            route_priority,
            float(navigation.fusion_score) if navigation is not None else 0.0,
            float(navigation.interface_score) if navigation is not None else 0.0,
            float(navigation.target_score) if navigation is not None else 0.0,
            float(item.roi.retrieval_score),
            -original_index,
        )

    def _hard_cap_high_output(
        self,
        output: list[SelectedROI],
        target: int,
    ) -> list[SelectedROI]:
        """Bound observer work while preserving routed parent coverage.

        One strongest child per parent is selected before any parent receives
        a second child.  Interface and Tissue Map-stratum lineages win ties,
        then fusion and PLIP retrieval scores fill the remaining positions.
        The chosen set is returned in its original order so evidence ordering
        remains as stable as possible relative to r25.
        """

        target = min(max(0, int(target)), len(output))
        if len(output) <= target:
            return output

        self.metrics["r26_high_budget_hard_cap_events"] += 1
        self.metrics["r26_high_budget_dropped_rois"] += len(output) - target
        if target == 0:
            return []

        best_by_parent: dict[str, int] = {}
        for index, item in enumerate(output):
            parent_id = str(item.roi.parent_roi_id or f"__unparented_{index}")
            previous = best_by_parent.get(parent_id)
            if previous is None or self._high_priority(
                item, index
            ) > self._high_priority(output[previous], previous):
                best_by_parent[parent_id] = index

        representatives = sorted(
            best_by_parent.values(),
            key=lambda index: self._high_priority(output[index], index),
            reverse=True,
        )
        chosen = representatives[:target]
        if len(chosen) < target:
            chosen_set = set(chosen)
            remaining = sorted(
                (index for index in range(len(output)) if index not in chosen_set),
                key=lambda index: self._high_priority(output[index], index),
                reverse=True,
            )
            chosen.extend(remaining[: target - len(chosen)])

        return [output[index] for index in sorted(chosen)]

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
        """Apply an unavoidable final cap before cellular/observer analysis."""

        target = (
            self._high_power_budget(component_route(), magnification)
            if plan is not None
            else self.high_power_top_k
        )
        output = super().select_high(
            slide,
            parents,
            magnification,
            query,
            ensure_parent_diversity=ensure_parent_diversity,
            plan=plan,
        )
        return self._hard_cap_high_output(output, target)


class R26HistoPLUSAnalyzer(R25HistoPLUSAnalyzer):
    """The validated r25 analyzer, explicitly versioned for r26 traces."""

    algorithm_version = "r26-budget-aware-histoplus-v2-hard-cap"
