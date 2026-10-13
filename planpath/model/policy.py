"""Narrow r29 policy for renal-subtype MCQ option-bias control.

r29 deliberately keeps the complete r28 workflow.  The only execution-plan
change is for the canonical renal histological-subtype single-choice task:
one Clear-cell-shaped PLIP query is replaced by a compact, option-balanced
set of morphology axes.  No reference answer is inspected.
"""

from __future__ import annotations

from planpath.core.controller import normalize_plan_for_question as r28_normalize_plan
from planpath.core.schemas import EvidencePlan


RENAL_SUBTYPE_OPTIONS = frozenset(
    {
        "Clear cell renal cell carcinoma",
        "Papillary renal cell carcinoma",
        "Chromophobe renal cell carcinoma",
        "Renal oncocytoma",
    }
)


def is_renal_subtype_mcq(question: dict[str, object]) -> bool:
    """Return whether the narrowly scoped r29 correction applies."""

    question_type = str(question.get("question_type") or "").casefold()
    text = str(question.get("question") or "").casefold()
    domain = str(question.get("clinical_domain") or "").casefold()
    options = {str(item) for item in question.get("options", []) or []}
    return bool(
        question_type in {"single_choice", "multiple_choice", "mcq"}
        and "renal" in domain
        and any(
            term in text
            for term in ("histological type", "pathological type", "tumor type")
        )
        and RENAL_SUBTYPE_OPTIONS <= options
    )


def normalize_plan_for_question(
    plan: EvidencePlan,
    question: dict[str, object],
) -> EvidencePlan:
    """Apply r28 normalization, then replace only biased subtype queries."""

    normalized = r28_normalize_plan(plan, question)
    if not is_renal_subtype_mcq(question):
        return normalized
    return normalized.model_copy(
        update={
            "target_finding": (
                "renal epithelial tumor subtype from balanced hallmark morphology"
            ),
            "question_attributes": [
                "renal pathology",
                "primary renal epithelial tumor",
                "balanced subtype morphology",
            ],
            # EvidencePlan permits three slide descriptions.  Each line spans
            # more than one answer hypothesis so no single option defines the
            # TITAN search direction.
            "slide_retrieval_descriptions": [
                (
                    "Renal epithelial neoplasm with nested alveolar papillary "
                    "or tubular architecture and representative viable tumor"
                ),
                (
                    "Renal epithelial neoplasm spanning clear pale and granular "
                    "eosinophilic cytoplasmic patterns with vascular stroma"
                ),
                (
                    "Renal epithelial neoplasm showing fibrovascular cores, "
                    "perinuclear halos, wrinkled nuclei, distinct membranes, "
                    "or a central scar"
                ),
            ],
            "low_power_query": (
                "renal epithelial neoplasm with nested alveolar papillary tubular "
                "or solid architecture, delicate vessels, fibrovascular cores, "
                "and central scar"
            ),
            "high_power_query": (
                "renal epithelial tumor cells spanning clear pale and granular "
                "eosinophilic cytoplasm, perinuclear halos, wrinkled nuclei, "
                "distinct borders, and fibrovascular papillae"
            ),
            "evidence_goal": (
                "compare directly observed hallmark architecture and cytoplasm "
                "across all supplied renal subtype hypotheses without allowing "
                "one answer option to define ROI retrieval"
            ),
        }
    )


__all__ = [
    "RENAL_SUBTYPE_OPTIONS",
    "is_renal_subtype_mcq",
    "normalize_plan_for_question",
]
