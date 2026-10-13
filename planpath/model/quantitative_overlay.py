"""Strictly isolated continuous-quantitative overlay for PlanPath.

The PlanPath clinical workflow remains the source of truth for every ordinary
question.  Only the two quantitative families for which the high-resolution
dual-evidence implementation has a validated estimator are delegated:

* gastric primary-tumour component fractions; and
* independent primary renal-tumour focus counts.

Distance, density, treatment-bed, staging, node-count, and all qualitative
questions stay on the unmodified PlanPath path. The overlay also removes reference
fields before delegation so a quantitative backend cannot accidentally consume
the answer or report-derived diagnostic evidence.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from planpath.measurement.quantification import (
    QuantificationProfile,
    quantitative_profile,
)


METHOD = "PlanPath"
QUANTITATIVE_BRANCH_VERSION = "planpath-quantitative-calibrated-continuous-v1"

_COMPONENT_PHENOTYPES = frozenset({"signet_ring", "mucinous", "hepatoid"})
_REFERENCE_FIELDS = frozenset(
    {"answer", "reference_answer", "diagnostic_evidence"}
)


class Agent(Protocol):
    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...


def allowlisted_quantitative_profile(
    question: dict[str, Any],
) -> QuantificationProfile | None:
    """Return the profile only for the narrow quantitative allowlist."""

    profile = quantitative_profile(question)
    if profile is None:
        return None
    if (
        profile.kind == "component_fraction"
        and profile.phenotype in _COMPONENT_PHENOTYPES
    ):
        return profile
    if profile.kind == "focus_count":
        return profile
    return None


def is_quantitative_question(question: dict[str, Any]) -> bool:
    return allowlisted_quantitative_profile(question) is not None


def question_without_reference_fields(
    question: dict[str, Any],
) -> dict[str, Any]:
    """Make the no-reference-answer boundary explicit and testable."""

    return {
        key: value
        for key, value in question.items()
        if key not in _REFERENCE_FIELDS
    }


def _tag_quantitative_trace(
    trace: dict[str, Any], profile: QuantificationProfile
) -> dict[str, Any]:
    trace["method"] = METHOD
    trace["planpath_quantitative"] = {
        "version": QUANTITATIVE_BRANCH_VERSION,
        "active": True,
        "task_kind": profile.kind,
        "phenotype": profile.phenotype,
        "scope": (
            "gastric component fractions and independent renal focus counts only"
        ),
        "ordinary_planpath_workflow_changed": False,
        "reference_answer_available_to_backend": False,
        "estimator_mode": "calibrated_continuous",
        "component_calibration_bins_enabled": True,
        "five_point_output_rounding_enabled": False,
        "backend_contract": (
            "whole-slide tissue denominator plus same-lineage high-power "
            "Patho-R1 morphology and HistoPLUS quality/cell confirmation"
        ),
    }
    return trace


class PlanPathQuantitativeOverlay:
    """Dispatch an allowlisted quantitative task without changing other tasks."""

    def __init__(self, base_agent: Agent, quantitative_agent: Agent) -> None:
        self.base_agent = base_agent
        self.quantitative_agent = quantitative_agent

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        profile = allowlisted_quantitative_profile(question)
        if profile is None:
            # Deliberately return the exact base object. Do not add metadata,
            # retag the method, or call the quantitative backend.
            return self.base_agent.run(question, slides, checkpoint=checkpoint)

        clean_question = question_without_reference_fields(question)

        def quantitative_checkpoint(value: dict[str, Any]) -> None:
            if checkpoint is not None:
                checkpoint(_tag_quantitative_trace(value, profile))

        trace = self.quantitative_agent.run(
            clean_question,
            slides,
            checkpoint=quantitative_checkpoint if checkpoint is not None else None,
        )
        return _tag_quantitative_trace(trace, profile)


__all__ = [
    "METHOD",
    "QUANTITATIVE_BRANCH_VERSION",
    "PlanPathQuantitativeOverlay",
    "allowlisted_quantitative_profile",
    "is_quantitative_question",
    "question_without_reference_fields",
]
