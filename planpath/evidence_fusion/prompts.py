"""r25 prompt adapters exposing bounded cross-scale focus support."""

from __future__ import annotations

import json
from typing import Any

from planpath.core.prompts import (
    final_messages as r23_5_final_messages,
    review_messages as r23_5_review_messages,
)
from planpath.core.schemas import EvidencePlan, EvidenceReview, ROIObservation

from .routing import component_route


def _tissue_qualified(item: ROIObservation) -> bool:
    analysis = item.cellular_analysis
    return bool(
        analysis is not None
        and any("Tissue Map context check" in value for value in analysis.limitations)
    )


def _cross_scale_summary(item: ROIObservation) -> dict[str, Any] | None:
    analysis = item.cellular_analysis
    if analysis is None:
        return None
    grouped = analysis.grouped_counts
    return {
        "evidence_id": item.evidence_id,
        "roi_id": item.roi_id,
        "quality": analysis.quality,
        "instance_count": analysis.instance_count,
        "neoplastic_like_count": int(grouped.get("neoplastic_like", 0)),
        "epithelial_count": int(grouped.get("epithelial", 0)),
        "immune_count": int(grouped.get("immune", 0)),
        "stromal_count": int(grouped.get("stromal", 0)),
        "cellular_event_count": int(grouped.get("cellular_events", 0)),
        "cancer_immune_contact_fraction": analysis.cancer_immune_contact_fraction,
        "cancer_endothelial_contact_fraction": (
            analysis.cancer_endothelial_contact_fraction
        ),
        "tissue_context_qualified": _tissue_qualified(item),
        "role": "focus quality and representativeness only",
    }


_CROSS_SCALE_RULES = """

r25 cross-scale evidence rules:
- r25_cross_scale_focus_support contains bounded HistoPLUS measurements. Use it only
  to judge whether a linked morphology field is cellularly representative and which
  already-transcribed field deserves priority. Never translate a cell count, density,
  nuclear size, or contact fraction directly into a diagnosis, grade, stage, invasion,
  margin status, or absence.
- tissue_context_qualified means that the cellular field also passed an independent
  Tissue Map compartment/interface check. It raises localization reliability but is
  not itself positive diagnostic evidence. A field lacking that flag can still be used
  when its visible morphology is direct and adequate, but do not let unqualified cell
  predictions overturn better tissue-qualified morphology.
- For grade or phenotype, cellular measurements nominate representative tumor-rich or
  event-rich fields; the named grade or type must still follow the visible morphology.
  For node/interface tasks, co-occurrence or contact only nominates a field for review;
  the same-ROI morphology must visibly establish the requested spatial relationship.
- When morphology and the auxiliary measurements disagree, morphology controls.
"""


def review_messages(
    question: dict[str, Any],
    plan: EvidencePlan,
    observations: list[ROIObservation],
    *,
    current_slide_id: str,
    remaining_candidate_slide_ids: list[str],
    available_focus_roi_ids: list[str],
    attempted_magnifications: list[str],
) -> list[dict[str, str]]:
    messages = r23_5_review_messages(
        question,
        plan,
        observations,
        current_slide_id=current_slide_id,
        remaining_candidate_slide_ids=remaining_candidate_slide_ids,
        available_focus_roi_ids=available_focus_roi_ids,
        attempted_magnifications=attempted_magnifications,
    )
    messages[0]["content"] += _CROSS_SCALE_RULES
    payload = json.loads(messages[1]["content"])
    payload["r25_component_route"] = component_route()
    payload["r25_cross_scale_focus_support"] = [
        summary
        for item in observations
        if item.slide_id == current_slide_id
        and (summary := _cross_scale_summary(item)) is not None
    ]
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
    return messages


def final_messages(
    question: dict[str, Any],
    plan: EvidencePlan,
    observations: list[ROIObservation],
    *,
    termination_reason: str,
    evidence_sufficient: bool,
    last_review: EvidenceReview | None,
    review_history: list[EvidenceReview],
    authorized_answer_candidate: str | None,
) -> list[dict[str, str]]:
    messages = r23_5_final_messages(
        question,
        plan,
        observations,
        termination_reason=termination_reason,
        evidence_sufficient=evidence_sufficient,
        last_review=last_review,
        review_history=review_history,
        authorized_answer_candidate=authorized_answer_candidate,
    )
    messages[0]["content"] += _CROSS_SCALE_RULES
    payload = json.loads(messages[1]["content"])
    payload["r25_component_route"] = component_route()
    payload["r25_cross_scale_focus_support"] = [
        summary
        for item in observations
        if (summary := _cross_scale_summary(item)) is not None
    ]
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
    return messages
