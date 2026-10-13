"""r24 wrapper that delegates every non-quantitative task to r23_5."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .controller import R24CoreController, R24_ROUTE_VERSION
from .quantification import R24Quantifier


METHOD = "PlanPath Measurement"


class BaseAgent(Protocol):
    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...


class PlanPathMeasurementRouter:
    """Add one guarded branch without changing r23_5 implementation files."""

    def __init__(
        self,
        base_r23_5_agent: BaseAgent | None,
        controller: R24CoreController,
        quantifier: R24Quantifier,
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
        if decision.route == "r23_5":
            if self.base_r23_5_agent is None:
                raise RuntimeError("ordinary r23_5 route requested without a base agent")
            return self.base_r23_5_agent.run(question, slides, checkpoint=checkpoint)
        assert decision.profile is not None
        result = self.quantifier.measure(question, slides, decision.profile)
        clean_question = {
            key: question[key]
            for key in ("patient_id", "qa_id", "question_type", "question", "options")
            if key in question
        }
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
                "version": R24_ROUTE_VERSION,
                "route": decision.route,
                "reason": decision.reason,
            },
            "quantitative_evidence": result.as_dict(),
            "final_answer": {
                "schema_version": "1.0",
                "answer": result.formatted_answer,
                "confidence": (
                    "medium" if result.status.startswith("authorized") else "low"
                ),
                "evidence_sufficiency": (
                    "sufficient" if result.status.startswith("authorized") else "insufficient"
                ),
                "termination_reason": result.status,
                "supporting_evidence_ids": ["Q0001"],
                "brief_rationale": result.measurement_method,
            },
        }
        if checkpoint is not None:
            checkpoint(trace)
        return trace
