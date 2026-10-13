"""r28 wrapper retaining r27 outside gastric regional-node questions."""

from __future__ import annotations

from typing import Any, Callable

from planpath.depth_calibration.workflow import PlanPathDepthCalibration

from .components import route_budget


METHOD = "PlanPath Nodal Calibration"


class PlanPathNodalCalibration(PlanPathDepthCalibration):
    """Record the route-local r28 nodal recall and precision policy."""

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
        active = route == "nodal" and "gastric" in domain.casefold()
        budget = route_budget(route, domain)
        visual_metrics = (trace.get("metrics") or {}).get("visual") or {}
        trace["r28_nodal_calibration"] = {
            "version": "r28-full-gastric-nodal-recall-v2",
            "active": active,
            "budget_per_slide": {
                "5x": budget.low_power,
                "20x": budget.twenty_x,
                "40x_if_requested": budget.forty_x,
            },
            "full_recall_events": int(
                visual_metrics.get("r28_nodal_full_recall_events", 0)
            ),
            "full_recall_extra_rois": int(
                visual_metrics.get("r28_nodal_full_recall_extra_rois", 0)
            ),
            "recall_reference": "r23_5_93 12x5x + 16x20x",
            "positive_evidence_contract": (
                "Patho-R1 must describe epithelial deposit architecture plus "
                "malignant morphology; HistoPLUS labels only route fields and "
                "cannot authorize metastatic carcinoma."
            ),
            "cross_lineage_guard": (
                "Separate ROI roots require destructive deposit architecture "
                "and at least two malignant morphology features."
            ),
            "complete_scan_reuse": (
                "Retained for identical neutral nodal scans; every reused scan "
                "is re-evaluated by the r28 positive-evidence contract."
            ),
            "r27_depth_calibration_retained": True,
            "quantitative_branch_retained": True,
            "non_gastric_nodal_routes_unchanged": True,
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace


__all__ = ["METHOD", "PlanPathNodalCalibration"]
