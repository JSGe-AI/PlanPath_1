"""Final PlanPath workflow with renal-subtype MCQ bias correction."""

from __future__ import annotations

from typing import Any, Callable

from planpath.nodal_calibration.workflow import PlanPathNodalCalibration

from .policy import is_renal_subtype_mcq


METHOD = "PlanPath"


class PlanPath(PlanPathNodalCalibration):
    """Run the complete PlanPath inference workflow."""

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        trace = super().run(question, slides, checkpoint=checkpoint)
        contextual_question = dict(question)
        if self.base_r23_5_agent.config.clinical_domain:
            contextual_question["clinical_domain"] = str(
                self.base_r23_5_agent.config.clinical_domain
            )
        active = is_renal_subtype_mcq(contextual_question)
        trace["method"] = METHOD
        trace["planpath_option_bias_control"] = {
            "version": "planpath-renal-subtype-mcq-balanced-retrieval-v1",
            "active": active,
            "scope": "canonical renal histological-subtype MCQ only",
            "reference_answer_used": False,
            "retrieval_policy": (
                "balanced morphology axes spanning every supplied subtype"
                if active
                else "base workflow unchanged"
            ),
            "single_option_defines_roi_query": False if active else None,
            "clear_cell_tie_priority": False if active else None,
            "first_option_structured_failure_fallback": (
                False if active else None
            ),
            "roi_budget": "base workflow unchanged",
            "evidence_aggregation": "base workflow unchanged except Clear ties abstain",
            "gastric_routes": "base workflow unchanged",
            "quantitative_branch": "base workflow unchanged",
            "tissue_navigation": "base workflow unchanged",
            "histoplus": "base workflow unchanged",
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace


__all__ = ["METHOD", "PlanPath"]
