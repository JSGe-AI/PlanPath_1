"""r25 wrapper: r23_5 state machine plus upgraded components and r24 quantification."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Protocol

from planpath.measurement.controller import R24CoreController

from .context import reset_current_question, set_current_question
from .quantification import R25Quantifier
from .routing import (
    component_route,
    histoplus_applicable,
    tissue_navigation_applicable,
)


METHOD = "PlanPath Evidence Fusion"


class BaseAgent(Protocol):
    config: Any

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...


class PlanPathEvidenceFusion:
    """Preserve the r23_5 outer workflow and replace only component use."""

    def __init__(
        self,
        base_r23_5_agent: BaseAgent,
        controller: R24CoreController,
        quantifier: R25Quantifier,
    ) -> None:
        self.base_r23_5_agent = base_r23_5_agent
        self.controller = controller
        self.quantifier = quantifier

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        decision = self.controller.route(question)
        clean_question = {
            key: question[key]
            for key in ("patient_id", "qa_id", "question_type", "question", "options")
            if key in question
        }
        if self.base_r23_5_agent.config.clinical_domain:
            clean_question["clinical_domain"] = str(
                self.base_r23_5_agent.config.clinical_domain
            )
        if decision.route == "r23_5":
            token = set_current_question(clean_question)
            try:
                trace = self.base_r23_5_agent.run(
                    question, slides, checkpoint=checkpoint
                )
                zoom_rounds = [
                    zoom
                    for slide_round in trace.get("slide_rounds", [])
                    for zoom in slide_round.get("zoom_rounds", [])
                ]
                zoom_rois = [
                    roi for zoom in zoom_rounds for roi in zoom.get("rois", [])
                ]
                low_rois = [
                    roi
                    for slide_round in trace.get("slide_rounds", [])
                    for roi in (slide_round.get("low_power") or {}).get("rois", [])
                ]
                all_visual_rois = low_rois + zoom_rois
                tissue_strata = Counter(
                    route.split(":", 1)[1]
                    for roi in all_visual_rois
                    if (
                        route := str(
                            (roi.get("tissue_navigation") or {}).get(
                                "selection_route"
                            )
                            or ""
                        )
                    ).startswith(
                        (
                            "r25_tissue_stratum:",
                            "r25_tissue_stratum_descendant:",
                        )
                    )
                )
                zoom_observations = [
                    observation
                    for zoom in zoom_rounds
                    for observation in zoom.get("observations", [])
                ]
                histoplus_observations = [
                    item
                    for item in zoom_observations
                    if item.get("cellular_analysis") is not None
                ]
                joint_observations = [
                    item
                    for item in histoplus_observations
                    if any(
                        "Tissue Map context check" in str(value)
                        for value in item["cellular_analysis"].get("limitations", [])
                    )
                ]
                trace["r25_component_integration"] = {
                    "version": "r25-question-anchored-cross-scale-v3",
                    "route": component_route(),
                    "tissue_navigation_applicable": tissue_navigation_applicable(),
                    "histoplus_applicable": histoplus_applicable(),
                    "tissue_navigation_enabled": bool(
                        self.base_r23_5_agent.config.tissue_navigation_enabled
                    ),
                    "histoplus_enabled": bool(
                        self.base_r23_5_agent.config.histoplus_enabled
                    ),
                    "ordinary_outer_workflow": "r23_5",
                    "tissue_navigated_total_rois": sum(
                        roi.get("tissue_navigation") is not None
                        for roi in all_visual_rois
                    ),
                    "tissue_navigated_zoom_rois": sum(
                        roi.get("tissue_navigation") is not None for roi in zoom_rois
                    ),
                    "histoplus_supported_observations": len(
                        histoplus_observations
                    ),
                    "joint_tissue_histoplus_observations": len(
                        joint_observations
                    ),
                    "histoplus_hotspot_recentered_rois": sum(
                        str((roi.get("tissue_navigation") or {}).get("selection_route"))
                        == "r25_joint_tissue_histoplus_hotspot"
                        for roi in zoom_rois
                    ),
                    "tissue_stratified_roi_lineages": dict(
                        sorted(tissue_strata.items())
                    ),
                    "diagnostic_contract": (
                        "Tissue Map localizes compartments/interfaces; HistoPLUS "
                        "qualifies and prioritizes cellular foci; Patho-R1 visible "
                        "morphology remains required for the answer."
                    ),
                }
                if checkpoint is not None:
                    checkpoint(trace)
                return trace
            finally:
                reset_current_question(token)

        assert decision.profile is not None
        result = self.quantifier.measure(clean_question, slides, decision.profile)
        trace = {
            "method": METHOD,
            "schema_version": "1.0",
            "status": "complete",
            "patient_id": str(question["patient_id"]),
            "qa_id": str(question["qa_id"]),
            "question": clean_question,
            "reference_answer_used_in_prompt": False,
            "site_labels_used": False,
            "patho_r1_used_by_quantitative_branch": False,
            "fixed_language_to_percentage_mapping_used": False,
            "controller_route": {
                "version": "r25-explicit-quantitative-route-v1",
                "route": "r25_quantitative",
                "reason": decision.reason,
            },
            "r25_component_integration": {
                "tissue_navigation_enabled": self.quantifier.tissue_navigation_enabled,
                "histoplus_enabled": self.quantifier.histoplus_enabled,
                "ordinary_outer_workflow": "r23_5",
            },
            "quantitative_evidence": result.as_dict(),
            "final_answer": {
                "schema_version": "1.0",
                "answer": result.formatted_answer,
                "confidence": (
                    "medium" if result.status.startswith("authorized") else "low"
                ),
                "evidence_sufficiency": (
                    "sufficient"
                    if result.status.startswith("authorized")
                    else "insufficient"
                ),
                "termination_reason": result.status,
                "supporting_evidence_ids": ["Q0001"],
                "brief_rationale": result.measurement_method,
            },
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace
