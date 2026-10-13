"""Gastric nodal evidence calibration for PlanPath r28.

The r23_5-to-r27 paired audit found that the gastric Yes/No regression was
confined to regional-node metastasis questions.  This module leaves every
other decision rule unchanged.  It prevents a HistoPLUS class label or a
generic epithelial focus from authorizing metastasis, while retaining strong
same-lineage and cross-lineage Patho-R1 morphology.
"""

from __future__ import annotations

import re
from typing import Any

from planpath.depth_calibration.evidence import (
    gastric_invasion_layer,
    patch_r27_component_hooks,
    question_specific_focus_roi_ids as r27_focus_roi_ids,
)
from planpath.core.controller import (
    _lymph_node_metastasis_support_groups as r27_nodal_support_groups,
)
from planpath.core.schemas import ROIObservation
_EPITHELIAL_IDENTITY = re.compile(
    r"\b(?:epithelial|gland\w*|acini|acinar|tubul\w*)\b",
    re.IGNORECASE,
)
_DEPOSIT_ARCHITECTURE = re.compile(
    r"\b(?:gland\w*|acini|acinar|tubul\w*|nests?|sheets?|clusters?|"
    r"cohesive\s+(?:cells?|clusters?))\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_ARCHITECTURE = re.compile(
    r"\b(?:disorganiz\w*|loss\s+of\s+polarity|architectural\s+disarray|"
    r"stromal\s+invasion|infiltrat\w*|irregular\s+(?:gland\w*|nests?|clusters?)|"
    r"angulated\s+gland\w*)\b",
    re.IGNORECASE,
)
_CONDITIONAL_OR_BENIGN = re.compile(
    r"\bepithelial\s+(?:component|cells?)\s*,?\s+if\s+present\b|"
    r"\b(?:bland|benign)\s+(?:epithelial|gland\w*)\b|"
    r"\bno\s+(?:epithelial|gland\w*)\b|"
    r"\b(?:epithelial|gland\w*)\b.{0,100}\b(?:spatially\s+separate|"
    r"maintain\w*\s+(?:their\s+)?normal\s+spatial\s+relationship)\b",
    re.IGNORECASE,
)
_MALIGNANT_FEATURES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\batyp\w*\b",
        r"\bpleomorph\w*\b",
        r"\birregular\s+nuclear\b",
        r"\bmitotic\s+figures?\b",
        r"\bhyperchrom\w*\b",
        r"\bcoarse\s+chromatin\b",
        r"\bprominent\s+nucleol\w*\b",
        r"\b(?:enlarged|large)\s+nuclei\b",
        r"\bloss\s+of\s+polarity\b",
        r"\bdisorganiz\w*\b",
    )
)


def _root_roi_id(
    item: ROIObservation,
    by_roi: dict[str, ROIObservation],
) -> str:
    current = item
    seen: set[str] = set()
    while current.parent_roi_id and current.parent_roi_id not in seen:
        seen.add(current.parent_roi_id)
        parent = by_roi.get(current.parent_roi_id)
        if parent is None:
            return current.parent_roi_id
        current = parent
    return current.roi_id


def _malignant_feature_count(text: str) -> int:
    return sum(bool(pattern.search(text)) for pattern in _MALIGNANT_FEATURES)


def _qualified_gastric_nodal_group(
    group: list[str],
    by_evidence: dict[str, ROIObservation],
    by_roi: dict[str, ROIObservation],
) -> bool:
    """Require deposit architecture in Patho-R1 morphology, not cell labels.

    Same-root evidence needs two malignant features or an explicitly
    destructive arrangement.  Joining separate ROI trees is allowed only for
    a stronger field with two malignant features and destructive epithelial
    architecture.  This preserves replacement-node metastases while rejecting
    a generic epithelial focus elsewhere on the same physical section.
    """

    items = [by_evidence[value] for value in group if value in by_evidence]
    if len(items) < 2:
        return False
    for deposit in items:
        text = deposit.morphology
        if (
            deposit.magnification not in {"20x", "40x"}
            or deposit.image_quality != "adequate"
            or not _EPITHELIAL_IDENTITY.search(text)
            or not _DEPOSIT_ARCHITECTURE.search(text)
            or _CONDITIONAL_OR_BENIGN.search(text)
        ):
            continue
        feature_count = _malignant_feature_count(text)
        destructive = bool(_DESTRUCTIVE_ARCHITECTURE.search(text))
        deposit_root = _root_roi_id(deposit, by_roi)
        same_root = any(
            other.evidence_id != deposit.evidence_id
            and _root_roi_id(other, by_roi) == deposit_root
            for other in items
        )
        if same_root and (feature_count >= 2 or destructive):
            return True
        if not same_root and feature_count >= 2 and destructive:
            return True
    return False


def gastric_nodal_support_groups(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> list[list[str]]:
    """Filter only gastric nodal positives through the r28 morphology gate."""

    groups = r27_nodal_support_groups(question, observations)
    domain = str(question.get("clinical_domain") or "").casefold()
    if "gastric" not in domain or not groups:
        return groups
    by_evidence = {item.evidence_id: item for item in observations}
    by_roi = {item.roi_id: item for item in observations}
    return [
        group
        for group in groups
        if _qualified_gastric_nodal_group(group, by_evidence, by_roi)
    ]


def gastric_nodal_support_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> list[str]:
    """Return the first r28-qualified gastric nodal deposit pair."""

    groups = gastric_nodal_support_groups(question, observations)
    return groups[0] if groups else []


def question_specific_focus_roi_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
    current_slide_id: str,
    available_roi_ids: list[str],
) -> list[str]:
    """Retain r27 focus ordering under the full gastric nodal budget."""

    return r27_focus_roi_ids(
        question,
        observations,
        current_slide_id,
        available_roi_ids,
    )


def patch_r28_component_hooks() -> None:
    """Install r27 depth hooks plus r28 gastric nodal calibration."""

    import planpath.core.controller as controller_module
    import planpath.core.workflow as workflow_module

    patch_r27_component_hooks()
    controller_module._lymph_node_metastasis_support_groups = (
        gastric_nodal_support_groups
    )
    controller_module._lymph_node_metastasis_support_ids = gastric_nodal_support_ids
    workflow_module._lymph_node_metastasis_support_ids = gastric_nodal_support_ids
    controller_module.question_specific_focus_roi_ids = question_specific_focus_roi_ids
    workflow_module.question_specific_focus_roi_ids = question_specific_focus_roi_ids


__all__ = [
    "gastric_invasion_layer",
    "gastric_nodal_support_groups",
    "gastric_nodal_support_ids",
    "patch_r28_component_hooks",
    "question_specific_focus_roi_ids",
]
