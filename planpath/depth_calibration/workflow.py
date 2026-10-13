"""r27 evidence-calibrated wrapper around the efficient r26 workflow."""

from __future__ import annotations

from typing import Any, Callable

from planpath.efficiency.workflow import PlanPathEfficientFusion

from .components import route_budget


METHOD = "PlanPath Depth Calibration"


class PlanPathDepthCalibration(PlanPathEfficientFusion):
    """Retag r26 traces and record the calibrated r27 evidence policy."""

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
        efficiency = trace.setdefault("r26_efficiency", {})
        visual_metrics = (trace.get("metrics") or {}).get("visual") or {}
        efficiency.update(
            {
                "version": "r27-evidence-calibrated-budget-v2",
                "route": route,
                "first_pass_budget_per_slide": {
                    "5x": budget.low_power,
                    "20x": budget.twenty_x,
                    "40x_if_requested": budget.forty_x,
                },
                "precision_guard": budget.precision_guard,
                "r26_low_high_hard_caps_retained": True,
                "gastric_depth_hard_cap_unchanged_from_r26": True,
                "gastric_depth_20x_observer_budget_unchanged": True,
                "depth_priority_candidate_rerank_only": True,
                "r27_depth_rerank_metrics": {
                    "candidate_expansion_uses": int(
                        visual_metrics.get("r27_depth_candidate_expansion_uses", 0)
                    ),
                    "extra_priority_children_retained": int(
                        visual_metrics.get("r27_depth_priority_reserves", 0)
                    ),
                },
            }
        )
        trace["r27_evidence_calibration"] = {
            "version": "r27-tumor-lineage-depth-calibration-v2",
            "gastric_depth_policy": (
                "a reconciled explicit 5x tumour-compartment relation, one "
                "joint-qualified strict high-power relation, or a five-slide/"
                "three-high-power cross-scale morphology consensus"
            ),
            "histoplus_is_required_for_anatomic_depth": False,
            "unsupported_depth_defaults_to_pt1": False,
            "generic_cells_are_tumour_depth_evidence": False,
            "negated_or_equivocal_relations_are_positive_evidence": False,
            "independent_shallow_layer_recovery": True,
            "quantitative_branch_retained": True,
            "ordinary_outer_workflow": "r23_5_via_r25_r26",
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace


__all__ = ["METHOD", "PlanPathDepthCalibration"]
