"""r29 renal-subtype evidence policy layered strictly on r28."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from planpath.nodal_calibration.evidence import patch_r28_component_hooks
from planpath.core.controller import _GROUNDABLE_FEATURES, _feature_polarities
from planpath.core.schemas import EvidenceReview, ROIObservation

from .policy import RENAL_SUBTYPE_OPTIONS, is_renal_subtype_mcq, normalize_plan_for_question


PreferenceFunction = Callable[
    [dict[str, Any], list[ROIObservation], list[EvidenceReview] | None, str | None],
    str | None,
]
FallbackFunction = Callable[..., str]

_preference_delegate: PreferenceFunction | None = None
_fallback_delegate: FallbackFunction | None = None


def renal_subtype_scores(
    observations: list[ROIObservation],
) -> dict[str, int]:
    """Return the unchanged r28 feature scores without its Clear tie rule."""

    scores = {item: 0 for item in RENAL_SUBTYPE_OPTIONS}
    for item in observations:
        if item.image_quality != "adequate":
            continue
        morphology = item.morphology
        lowered = morphology.casefold()
        high_power = item.magnification in {"20x", "40x"}
        clear_present = True in _feature_polarities(
            morphology,
            _GROUNDABLE_FEATURES["clear_cytoplasm"],
        )
        nested = bool(re.search(r"\b(?:nested|nests?|alveolar)\b", lowered))
        delicate_vessels = bool(
            re.search(
                r"\b(?:delicate|thin-walled)\b.{0,45}"
                r"\b(?:vascular|vessels?|sept\w*)\b|"
                r"\bfibrovascular\s+sept\w*\b",
                lowered,
            )
        )
        papillary = True in _feature_polarities(
            morphology,
            _GROUNDABLE_FEATURES["papillary"],
        )
        fibrovascular_core = bool(
            re.search(r"\bfibrovascular\s+cores?\b", lowered)
        )
        perinuclear = True in _feature_polarities(
            morphology,
            _GROUNDABLE_FEATURES["perinuclear_halo"],
        )
        raisinoid = bool(re.search(r"\b(?:raisinoid|wrinkled)\b", lowered))
        granular_eosinophilic = bool(
            re.search(
                r"\bgranular\s+eosinophilic\s+cytoplasm\b|"
                r"\beosinophilic\s+granular\s+cytoplasm\b",
                lowered,
            )
        )
        pale_vacuolated = bool(
            re.search(
                r"\b(?:pale|clear)\w*\s+(?:and\s+)?vacuolat\w*\s+cytoplasm\b|"
                r"\bcytoplasm\w*.{0,35}\b(?:pale|clear|vacuolat\w*)\b|"
                r"\b(?:pale|clear)\s+(?:material|space)\w*.{0,35}"
                r"\b(?:intracellular|within\s+(?:a\s+)?nucleated\s+cell)\b",
                lowered,
            )
        )
        central_scar = bool(
            re.search(r"\bcentral\s+(?:fibrous\s+)?scar\b", lowered)
        )

        if clear_present:
            scores["Clear cell renal cell carcinoma"] += 4 if high_power else 2
        elif pale_vacuolated and high_power:
            scores["Clear cell renal cell carcinoma"] += 2
        if nested and delicate_vessels:
            scores["Clear cell renal cell carcinoma"] += 3
        if papillary and fibrovascular_core:
            scores["Papillary renal cell carcinoma"] += 4
        elif papillary:
            scores["Papillary renal cell carcinoma"] += (
                1 if item.magnification == "5x" else 2
            )
        if perinuclear:
            scores["Chromophobe renal cell carcinoma"] += 3
        if raisinoid:
            scores["Chromophobe renal cell carcinoma"] += 2
        if granular_eosinophilic:
            scores["Renal oncocytoma"] += 2
        if granular_eosinophilic and nested:
            scores["Renal oncocytoma"] += 1
        if central_scar:
            scores["Renal oncocytoma"] += 3
    return scores


def balanced_renal_subtype_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> str | None:
    """Authorize only a unique morphology winner; Clear never wins a tie."""

    if not is_renal_subtype_mcq(question):
        return None
    scores = renal_subtype_scores(observations)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner, winning_score = ordered[0]
    if winning_score < 2 or winning_score == ordered[1][1]:
        return None
    return winner


def question_specific_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    reviews: list[EvidenceReview] | None = None,
    termination_reason: str | None = None,
) -> str | None:
    if is_renal_subtype_mcq(question):
        return balanced_renal_subtype_preference(question, observations)
    if _preference_delegate is None:
        return None
    return _preference_delegate(
        question,
        observations,
        reviews,
        termination_reason,
    )


def safe_final_fallback_answer(
    question: dict[str, Any],
    *,
    preferred_answer: str | None,
    authorized_answer_candidate: str | None,
    review_history: list[EvidenceReview],
    renal_grade_floor: str | None,
    renal_grade_ceiling: str | None,
) -> str:
    """Remove the first-option fallback only for the corrected MCQ family."""

    if is_renal_subtype_mcq(question):
        for value in (preferred_answer, authorized_answer_candidate):
            if str(value or "").strip():
                return str(value).strip()
        for review in reversed(review_history):
            if str(review.answer_candidate or "").strip():
                return str(review.answer_candidate).strip()
        # This branch is reached only after all structured-generation repairs
        # fail.  A stable hash avoids silently converting that rare failure
        # into an A/Clear prediction.  It is not used as diagnostic evidence.
        options = sorted(str(item) for item in question.get("options", []))
        if options:
            task_key = f"{question.get('patient_id', '')}/{question.get('qa_id', '')}"
            digest = hashlib.sha256(task_key.encode("utf-8")).digest()
            return options[int.from_bytes(digest[:8], "big") % len(options)]
    if _fallback_delegate is None:
        raise RuntimeError("r29 fallback delegate was not installed")
    return _fallback_delegate(
        question,
        preferred_answer=preferred_answer,
        authorized_answer_candidate=authorized_answer_candidate,
        review_history=review_history,
        renal_grade_floor=renal_grade_floor,
        renal_grade_ceiling=renal_grade_ceiling,
    )


def patch_r29_component_hooks() -> None:
    """Install r28 first, then overlay only the renal MCQ correction."""

    import planpath.core.controller as controller_module
    import planpath.core.workflow as workflow_module

    global _preference_delegate, _fallback_delegate

    patch_r28_component_hooks()
    active_preference = controller_module._question_specific_preference
    if active_preference is not question_specific_preference:
        _preference_delegate = active_preference
    active_fallback = controller_module._safe_final_fallback_answer
    if active_fallback is not safe_final_fallback_answer:
        _fallback_delegate = active_fallback

    # The controller continues to generate/cache the unmodified r28 plan.
    # Only the workflow execution projection is changed, preventing r29 from
    # poisoning the shared r28 plan cache.
    workflow_module.normalize_plan_for_question = normalize_plan_for_question
    controller_module._question_specific_preference = question_specific_preference
    controller_module._safe_final_fallback_answer = safe_final_fallback_answer


__all__ = [
    "balanced_renal_subtype_preference",
    "patch_r29_component_hooks",
    "question_specific_preference",
    "renal_subtype_scores",
    "safe_final_fallback_answer",
]
