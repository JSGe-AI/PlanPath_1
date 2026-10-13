"""r26 wrapper around the unchanged r25/r23_5 clinical workflow."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from planpath.evidence_fusion.workflow import PlanPathEvidenceFusion

from .components import route_budget


METHOD = "PlanPath Efficient Fusion"


class PlanPathEfficientFusion(PlanPathEvidenceFusion):
    """Add auditable efficiency metadata without changing answer semantics."""

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        trace = super().run(question, slides, checkpoint=checkpoint)
        trace["method"] = METHOD

        integration = trace.get("r25_component_integration") or {}
        route = str(integration.get("route") or "quantitative")
        domain = str(
            question.get("clinical_domain")
            or self.base_r23_5_agent.config.clinical_domain
            or ""
        )
        budget = route_budget(route, domain)
        observations_by_magnification: Counter[str] = Counter()
        slides_with_visual_work = 0
        for slide_round in trace.get("slide_rounds", []):
            low = (slide_round.get("low_power") or {}).get("observations", [])
            if low:
                slides_with_visual_work += 1
                observations_by_magnification["5x"] += len(low)
            for zoom in slide_round.get("zoom_rounds", []):
                magnification = str(zoom.get("magnification") or "unknown")
                observations_by_magnification[magnification] += len(
                    zoom.get("observations", [])
                )

        visual_metrics = (trace.get("metrics") or {}).get("visual") or {}
        trace["r26_efficiency"] = {
            "version": "r26-route-aware-budget-v2-hard-cap",
            "route": route,
            "first_pass_budget_per_slide": {
                "5x": budget.low_power,
                "20x": budget.twenty_x,
                "40x_if_requested": budget.forty_x,
            },
            "precision_guard": budget.precision_guard,
            "all_candidate_slide_coverage_preserved": route
            in {
                "nodal",
                "depth_interface",
                "vascular_interface",
                "neural_interface",
                "renal_boundary",
            },
            "adaptive_rescue": (
                "The unchanged evidence review may request a bounded 40x pass "
                "when 20x evidence is insufficient or ambiguous."
            ),
            "tissue_map_strata_unchanged_from_r25": True,
            "histoplus_hotspot_logic_unchanged_from_r25": True,
            "low_power_budget_hard_cap": True,
            "high_power_budget_hard_cap": True,
            "hard_cap_metrics": {
                "low_power_events": int(
                    visual_metrics.get("r26_low_budget_hard_cap_events", 0)
                ),
                "low_power_dropped_rois": int(
                    visual_metrics.get("r26_low_budget_dropped_rois", 0)
                ),
                "high_power_events": int(
                    visual_metrics.get("r26_high_budget_hard_cap_events", 0)
                ),
                "high_power_dropped_rois": int(
                    visual_metrics.get("r26_high_budget_dropped_rois", 0)
                ),
            },
            "observed_slides": slides_with_visual_work,
            "observations_by_magnification": dict(
                sorted(observations_by_magnification.items())
            ),
            "observations_total": sum(observations_by_magnification.values()),
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace
