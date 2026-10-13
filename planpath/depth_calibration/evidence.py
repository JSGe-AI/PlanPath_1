"""Evidence calibration for the efficient r27 gastric depth route."""

from __future__ import annotations

import re
from typing import Any

from planpath.evidence_fusion.evidence import (
    _GASTRIC_DEEP_STRUCTURE,
    _joint_qualified,
    r23_5_gastric_invasion_layer,
    question_specific_focus_roi_ids as r25_focus_roi_ids,
)
from planpath.core.schemas import ROIObservation

from .context import set_depth_priority_parent_ids


_TUMOR_LINEAGE = re.compile(
    r"\b(?:carcinoma|tumou?r|neoplastic|malignant|"
    r"epithelial\s+(?:proliferation|cells?|clusters?|nests?)|"
    r"(?:irregular|atypical|crowded|haphazard|angulated|disorganized)\w*\s+"
    r"(?:gland|tubul)\w*|"
    r"glands?\s+(?:display|show|appear|extend|infiltrat|invad|cross)\w*)\b",
    re.IGNORECASE,
)
_INVASIVE_RELATION = re.compile(
    r"\b(?:infiltrat|inva(?:d|s)|extend|reach|breach|cross|travers|"
    r"penetrat|disrupt)\w*\b",
    re.IGNORECASE,
)
_NEGATED_RELATION = re.compile(
    r"\b(?:no|not|without|lack\w*|absen\w*|negative\s+for|free\s+of)\b"
    r".{0,65}\b(?:infiltrat|inva(?:sion|d\w*)|extend|reach|breach|cross|"
    r"travers|penetrat|disrupt)\w*\b|"
    r"\b(?:does?|did)\s+not\b.{0,35}\b(?:infiltrat|extend|reach|breach|"
    r"cross|travers|penetrat|disrupt)\w*\b|"
    r"\bnon[-\s]?infiltrative\b|"
    r"\b(?:separat\w*\s+from|pushing\s+border)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RELATION = re.compile(
    r"\b(?:appear|suggest|suspicious|possible|partial|questionable)\w*\b|"
    r"\b(?:not|no)\s+(?:clearly|definitively|fully|reliably)\b|"
    r"\b(?:cannot|unable|difficult|challenging|indeterminate|equivocal)\b|"
    r"\bfull\s+extent\b.{0,45}\bnot\b",
    re.IGNORECASE,
)
_LAYER_STRUCTURE: dict[str, re.Pattern[str]] = {
    **_GASTRIC_DEEP_STRUCTURE,
    "Muscularis propria": re.compile(r"\bmuscularis\s+propria\b", re.IGNORECASE),
    "Submucosa": re.compile(r"\bsubmucos\w*\b", re.IGNORECASE),
}
_INTERFACE_NOUN = re.compile(
    r"\b(?:interface|edge|border|boundary|glands?)\b", re.IGNORECASE
)
_VERIFIED_OUTER_SURFACE = re.compile(
    r"\bepithelial\s+proliferation\s+visibly\s+"
    r"(?:reaches|traverses)\s+the\s+true\s+outer\s+free\s+tissue\s+surface\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.!?;])\s+", text)
        if value.strip()
    ]


def _strict_tumor_relation(item: ROIObservation, layer: str) -> bool:
    """Require an explicit, non-negated tumour-to-compartment relation.

    The inherited r25 matcher deliberately had high recall and included the
    generic token ``cells``.  On gastric depth traces that promoted lymphoid
    cells and explicitly non-infiltrative interfaces to tumour invasion.  The
    r27 matcher requires a tumour/epithelial lineage and evaluates negation
    and uncertainty in the local relation sentence.
    """

    if layer == "Serosa of the gastric wall" and _VERIFIED_OUTER_SURFACE.search(
        item.morphology
    ):
        return True
    structure = _LAYER_STRUCTURE[layer]
    sentences = _sentences(item.morphology)
    for index, sentence in enumerate(sentences):
        if not structure.search(sentence) or not _INVASIVE_RELATION.search(sentence):
            continue
        previous = sentences[index - 1] if index else ""
        lineage_present = bool(_TUMOR_LINEAGE.search(sentence)) or bool(
            _TUMOR_LINEAGE.search(previous) and _INTERFACE_NOUN.search(sentence)
        )
        if not lineage_present or _NEGATED_RELATION.search(sentence):
            continue
        # The following sentence often contains the qualification that the
        # apparent crossing cannot actually be resolved.  Keep that local
        # qualifier attached to the claim rather than treating it as support.
        local_claim = " ".join(sentences[index : index + 2])
        if _UNCERTAIN_RELATION.search(local_claim):
            continue
        return True
    return False


def _candidate_tumor_relation(item: ROIObservation, layer: str) -> bool:
    """Nominate an equivocal low-power interface for another bounded zoom."""

    structure = _LAYER_STRUCTURE[layer]
    sentences = _sentences(item.morphology)
    for index, sentence in enumerate(sentences):
        if not structure.search(sentence) or not _INVASIVE_RELATION.search(sentence):
            continue
        previous = sentences[index - 1] if index else ""
        if not (
            _TUMOR_LINEAGE.search(sentence)
            or (_TUMOR_LINEAGE.search(previous) and _INTERFACE_NOUN.search(sentence))
        ):
            continue
        if not _NEGATED_RELATION.search(sentence):
            return True
    return False


def _independently_confirmed_shallow_layer(
    observations: list[ROIObservation],
) -> str | None:
    """Recover a shallower layer only from its own qualified morphology.

    This is not the old peel-away behavior: rejecting a fat claim does not
    expose whatever the inherited aggregator returns next.  Muscularis or
    submucosa must independently have a tumour-specific direct relation.
    """

    for layer in ("Muscularis propria", "Submucosa"):
        direct = [item for item in observations if _strict_tumor_relation(item, layer)]
        low = [item for item in direct if item.magnification == "5x"]
        high = [item for item in direct if item.magnification in {"20x", "40x"}]
        if low or any(_joint_qualified(item) for item in high):
            return layer
    return None


def gastric_invasion_layer(
    observations: list[ROIObservation],
) -> str | None:
    """Accept reproducible depth evidence without making HistoPLUS mandatory.

    r25 rejected otherwise valid 5x deep relations whenever *any* unrelated
    field had a joint Tissue Map/HistoPLUS measurement.  That made the answer
    fall back to mucosa/pT1 even when a reconciled tissue-level field showed a
    direct tumour-fat relation.  r27 accepts that strict 5x anatomy, retains
    the single-field joint guard, and keeps a broader five-slide/three-high-
    power consensus for wording variants.

    The broader rule is deliberately cross-scale and cross-slide.  It cannot
    be satisfied by duplicating neighboring fields from one parent, and an
    unsupported deepest claim is rejected.  A shallower claim is returned
    only when it independently passes the same tumour-specific relation gate.
    """

    adequate = [
        item for item in observations if item.image_quality == "adequate"
    ]
    layer = r23_5_gastric_invasion_layer(adequate)
    if layer not in _GASTRIC_DEEP_STRUCTURE:
        return layer

    strict = [item for item in adequate if _strict_tumor_relation(item, layer)]
    strict_high = [
        item for item in strict if item.magnification in {"20x", "40x"}
    ]
    strict_low = [item for item in strict if item.magnification == "5x"]
    if layer == "Serosa of the gastric wall" and any(
        _VERIFIED_OUTER_SURFACE.search(item.morphology) for item in strict_high
    ):
        return layer
    if any(_joint_qualified(item) for item in strict_high):
        return layer
    # A 5x field preserves the tissue-level anatomical relationship and has
    # already passed the independent high-risk transcript reconciliation.
    # High-power cell-only claims without that context may not stage depth.
    if strict_low:
        return layer

    broad = [
        item
        for item in adequate
        if r23_5_gastric_invasion_layer([item]) == layer
    ]
    broad_high = [
        item for item in broad if item.magnification in {"20x", "40x"}
    ]
    if (
        len({item.slide_id for item in broad}) >= 5
        and len({item.slide_id for item in broad_high}) >= 3
    ):
        return layer
    return _independently_confirmed_shallow_layer(adequate)


def question_specific_focus_roi_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
    current_slide_id: str,
    available_roi_ids: list[str],
) -> list[str]:
    """Spend the fixed depth budget twice on the best unresolved interface."""

    base = r25_focus_roi_ids(
        question,
        observations,
        current_slide_id,
        available_roi_ids,
    )
    domain = str(question.get("clinical_domain") or "").casefold()
    text = str(question.get("question") or "").casefold()
    is_extent = "gastric" in domain and (
        (
            "deepest" in text
            and any(value in text for value in ("layer", "level", "tissue"))
        )
        or any(
            value in text
            for value in (
                "pt category",
                "pt stage",
                "pathological t category",
                "pathological t (pt) stage",
                "primary-tumor category (pt)",
            )
        )
    )
    if not is_extent:
        set_depth_priority_parent_ids([])
        return base

    available = set(available_roi_ids)
    candidates: list[tuple[int, int, str]] = []
    for index, item in enumerate(observations):
        if (
            item.slide_id != current_slide_id
            or item.roi_id not in available
            or item.magnification != "5x"
        ):
            continue
        score = 0
        for layer in _LAYER_STRUCTURE:
            if _strict_tumor_relation(item, layer):
                score = max(score, 3)
            elif _candidate_tumor_relation(item, layer):
                score = max(score, 2)
        if score:
            candidates.append((score, -index, item.roi_id))
    candidates.sort(reverse=True)
    priority = [roi_id for _, _, roi_id in candidates[:2]]
    set_depth_priority_parent_ids(priority)
    return priority + [roi_id for roi_id in base if roi_id not in priority]


def patch_r27_component_hooks() -> None:
    """Install r25 hooks plus the r27 depth aggregator in this process."""

    import planpath.core.controller as controller_module
    import planpath.core.workflow as workflow_module

    from planpath.evidence_fusion.components import patch_r23_5_component_hooks

    patch_r23_5_component_hooks()
    controller_module._gastric_invasion_layer = gastric_invasion_layer
    controller_module.question_specific_focus_roi_ids = question_specific_focus_roi_ids
    workflow_module.question_specific_focus_roi_ids = question_specific_focus_roi_ids


__all__ = [
    "gastric_invasion_layer",
    "patch_r27_component_hooks",
    "question_specific_focus_roi_ids",
]
