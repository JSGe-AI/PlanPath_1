"""Patho-R1 ROI description and a thin visual-analysis facade."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import AgentConfig
from .io_utils import read_json, write_json_atomic
from .plip import ROISelector, SelectedROI
from .prompts import (
    OBSERVATION_PROMPT_VERSION,
    OBSERVATION_VERIFICATION_CACHE_FAMILY_VERSION,
    OBSERVATION_VERIFICATION_PROMPT_VERSION,
    observation_messages,
    observation_verification_messages,
    outer_surface_verification_messages,
)
from .schemas import (
    EvidencePlan,
    ObservationContent,
    OuterSurfaceAssessment,
    ROI,
    ROIObservation,
)
from .tissue_routing import routing_policy
from .structured import StructuredGenerator


_DIAGNOSTIC_LANGUAGE = re.compile(
    r"\b(?:diagnos\w*|malignan\w*|benign\w*|neoplas\w*|tumou?r\w*|dysplas\w*|"
    r"carcinoma\w*|adenocarcinoma\w*|lymphoma\w*|leukemia\w*|sarcoma\w*|"
    r"melanoma\w*|adenoma\w*|fibroadenoma\w*|lipoma\w*|oncocytoma\w*|"
    r"lymphoproliferative\w*|"
    r"metasta(?:sis|tic)|grade\s*[1-4]|stage\s*[ivx0-9]|p[tnm]\s*[0-9x]|"
    r"positive\s+margin|negative\s+margin|(?:low|high)[-\s]+grade|"
    r"(?:peri)?neural\s+invasion|lymphovascular\s+invasion|vascular\s+invasion|"
    r"capsul(?:e|ar)\s+invasion)\b",
    re.IGNORECASE,
)
_LOW_POWER_OVERCLAIM = re.compile(
    r"\b(?:mito(?:sis|ses|tic)|nucleol(?:us|i|ar)|chromatin|nucle(?:us|i)|nuclear|"
    r"cytolog(?:y|ic)|atypia|atypical|pleomorph(?:ism|ic))\b",
    re.IGNORECASE,
)
_CONCLUSION_LANGUAGE = re.compile(
    r"\b(?:consistent\s+with|compatible\s+with|suggest(?:s|ive)?|aligns?\s+with|"
    r"indicat(?:e|es|ed|ing)|likely|favor(?:s|ed)?|"
    r"argues?\s+against|concerning\s+for|rule(?:s|d)?\s+out|definitive\s+classification|"
    r"histologically\s+unremarkable|clinical\s+(?:context|correlation)|patient\s+history|"
    r"confined\s+to|uninvolved|"
    r"no\s+evidence\s+of(?:\s+\w+){0,8}\s+(?:invasion|infiltrat\w*)|"
    r"without(?:\s+\w+){0,8}\s+(?:invasion|infiltrat\w*)|"
    r"lack\w*(?:\s+\w+){0,8}\s+(?:invasion|infiltrat\w*)|"
    r"characteristic\s+of|supports?\s+(?:this|the|a|an)\s+(?:interpretation|diagnosis)|"
    r"ancillary\s+(?:study|studies|test|testing)|adjunct(?:ive)?\s+(?:study|studies|test|testing)|"
    r"immunohistochem|clinical\s+suspicion|"
    r"CAIX|PAX8|CK7|CD\d+|BCL[26]|Ki[-\s]?67|AE1/?AE3|CAM5\.2|EMA|vimentin|"
    r"renal|kidney|adrenal|gastric|stomach|pulmonary|lung|pneumocyt\w*|hepatocyt\w*)",
    re.IGNORECASE,
)
_LOCAL_NEGATION = re.compile(
    r"\b(?:no|not|without|lack\w*|absen\w*|inconspicuous|not\s+assessable)\b",
    re.IGNORECASE,
)
_LOCAL_POST_NEGATION = re.compile(
    r"^\s*(?:(?:is|are|was|were|appears?|remains?)\s+)?"
    r"(?:directly\s+)?(?:no|not|without|absen\w*)\b",
    re.IGNORECASE,
)
_INFERRED_SITE_MODIFIER = re.compile(
    r"\b(?:renal|kidney|adrenal|gastric|stomach|pulmonary|lung|"
    r"endometrial|pneumocyt\w*|hepatocyt\w*)\b",
    re.IGNORECASE,
)

# These expressions deliberately describe generic visible structures rather
# than a diagnosis.  Their co-occurrence is only a reason to ask the
# independent transcriber for a precise local relationship; it is not itself
# evidence of perineural or lymphovascular invasion.
_EPITHELIAL_COMPONENT = re.compile(
    r"\b(?:epithelial|glandular)\s+"
    r"(?:cells?|groups?|clusters?|nests?|glands?|proliferation|structures?)\b|"
    r"\b(?:irregular\s+(?:epithelial\s+)?glands?|cohesive\s+(?:epithelial\s+)?"
    r"(?:cells?|groups?|clusters?|nests?)|(?:nests?|clusters?|groups?)\s+of\s+"
    r"(?:atypical\s+)?epithelial\s+cells?)\b",
    re.IGNORECASE,
)
_NERVE_COMPONENT = re.compile(
    r"\b(?:nerve(?:\s+(?:fascicle|profile|sheath|bundle))?|neural\s+structure|"
    r"perineur\w*|epineur\w*)\b",
    re.IGNORECASE,
)
_VASCULAR_COMPONENT = re.compile(
    r"\b(?:blood\s+vessels?|vessels?(?:\s+lumen)?|"
    r"vascular\s+(?:spaces?|channels?|lumina?|structures?)|"
    r"lymphatic\s+(?:spaces?|channels?|lumina?|structures?)|"
    r"endothelial[-\s]+lined\s+(?:spaces?|channels?|lumina?|vessels?)|"
    r"endothelial\s+(?:cells?|lining|spaces?|channels?|lumina?))\b",
    re.IGNORECASE,
)
_CAPSULE_CELLULAR_COMPONENT = re.compile(
    r"\b(?:epithelial|glandular)\s+"
    r"(?:cells?|groups?|clusters?|nests?|proliferation|structures?)\b|"
    r"\b(?:cellular\s+(?:nests?|proliferation)|solid\s+(?:nests?|sheets?)|"
    r"nested\s+(?:cells?|architecture)|clear\s+cells?|polygonal\s+cells?|"
    r"cohesive\s+(?:cells?|groups?|clusters?|nests?))\b",
    re.IGNORECASE,
)
_CAPSULE_BOUNDARY_COMPONENT = re.compile(
    r"\b(?:fibrous\s+capsul\w*|(?:broad|thick|dense|continuous|outer)\s+"
    r"(?:collagenous\s+)?fibrous\s+(?:band|layer|boundary|rim)|"
    r"(?:mesothelial[-\s]+lined\s+)?(?:outer|free)"
    r"(?:\s+(?:outer|free))?\s+(?:tissue\s+)?surface|serosal\s+surface)\b",
    re.IGNORECASE,
)
_ADIPOSE_COMPONENT = re.compile(
    r"\b(?:mature\s+adipocytes?|adipocytes?|adipose\s+(?:tissue|compartment|lobules?)|"
    r"fat\s+(?:tissue|lobules?))\b",
    re.IGNORECASE,
)
_OUTER_SURFACE_CONTEXT = re.compile(
    r"\b(?:outer|free)(?:\s+(?:outer|free))?\s+(?:tissue\s+)?surface\b|"
    r"\bserosal\s+surface\b|\btissue\s+periphery\b",
    re.IGNORECASE,
)
_EXPLICIT_SURFACE_SEPARATION = re.compile(
    r"\b(?:epithelial|gland\w*)\b[^.;:!?]{0,100}"
    r"\b(?:remain\w*\s+)?separat\w*\s+from\b[^.;:!?]{0,55}"
    r"\b(?:outer|free|serosal)\b[^.;:!?]{0,35}\bsurface\b|"
    r"\b(?:epithelial|gland\w*)\b[^.;:!?]{0,100}"
    r"\b(?:do|does)\s+not\s+(?:reach|breach|cross|travers)\w*\b"
    r"[^.;:!?]{0,55}\b(?:outer|free|serosal)\b[^.;:!?]{0,35}\bsurface\b|"
    r"\b(?:outer|free|serosal)\b[^.;:!?]{0,35}\bsurface\b"
    r"[^.;:!?]{0,100}\b(?:not\s+(?:reached|breached|crossed|traversed)|"
    r"remain\w*\s+intact)\b",
    re.IGNORECASE,
)
_SYNTHETIC_VERIFIED_SURFACE_CLAUSE = re.compile(
    r"(?:^|\s+)The epithelial proliferation visibly (?:reaches|traverses) "
    r"the true outer free tissue surface\.",
    re.IGNORECASE,
)
_SURFACE_EPITHELIAL_RISK_FEATURES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:irregular|crowded|complex|angulated|distorted)\s+"
        r"(?:epithelial\s+)?(?:glands?|nests?|clusters?)\b",
        r"\b(?:disorganiz\w*|haphazard)\s+(?:growth|arrangement|architecture|pattern)\b",
        r"\b(?:infiltrat|invad|interdigitat|penetrat|destruct)\w*\b",
        r"\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|irregular\s+nuclear\s+"
        r"(?:membran\w*|contours?)|prominent\s+nucleol\w*)\b",
        r"\b(?:loss|lack)\w*\s+(?:of\s+)?(?:normal\s+)?(?:polarity|cohesion)\b",
        r"\bdesmoplas\w*\b",
    )
)

_STRICT_PERINEURAL_RELATION = re.compile(
    rf"\b(?:{_EPITHELIAL_COMPONENT.pattern})\b[^.;:!?]{{0,90}}"
    rf"\b(?:surround|encas|encircl|wrap|within|inside|invad|infiltrat|"
    rf"penetrat|enter)\w*\b[^.;:!?]{{0,55}}"
    rf"\b(?:{_NERVE_COMPONENT.pattern})\b|"
    rf"\b(?:{_NERVE_COMPONENT.pattern})\b[^.;:!?]{{0,70}}"
    rf"\b(?:is\s+|are\s+|being\s+)?(?:surround|encas|encircl|wrap|invad|"
    rf"infiltrat|penetrat|contain|enclos)\w*\s+(?:by\s+|with\s+)?"
    rf"(?:an?\s+|the\s+)?(?:{_EPITHELIAL_COMPONENT.pattern})\b",
    re.IGNORECASE,
)
_STRICT_LYMPHOVASCULAR_RELATION = re.compile(
    rf"\b(?:{_EPITHELIAL_COMPONENT.pattern})\b[^.;:!?]{{0,90}}"
    rf"\b(?:within|inside|intraluminal|intravascular|intralymphatic|occup|fill)\w*\b"
    rf"[^.;:!?]{{0,55}}\b(?:{_VASCULAR_COMPONENT.pattern})\b|"
    rf"\b(?:{_EPITHELIAL_COMPONENT.pattern})\b[^.;:!?]{{0,60}}"
    rf"\b(?:surround|encas|enclos)\w*\s+by\s+"
    rf"(?:an?\s+|the\s+)?endothelial\s+(?:cells?|lining)\b|"
    rf"\b(?:{_VASCULAR_COMPONENT.pattern})\b[^.;:!?]{{0,75}}"
    rf"\b(?:contain|enclos|harbou?r|hous|occup|fill)\w*"
    rf"\s+(?:by\s+|with\s+)?(?:an?\s+|the\s+)?"
    rf"(?:{_EPITHELIAL_COMPONENT.pattern})\b",
    re.IGNORECASE,
)
_STRICT_CAPSULE_INTERFACE_RELATION = re.compile(
    rf"\b(?:{_CAPSULE_CELLULAR_COMPONENT.pattern})\b[^.;:!?]{{0,95}}"
    rf"\b(?:enter|extend|infiltrat|penetrat|breach|cross|interdigitat|travers|span)\w*\b"
    rf"[^.;:!?]{{0,65}}\b(?:{_CAPSULE_BOUNDARY_COMPONENT.pattern})\b|"
    rf"\b(?:{_CAPSULE_BOUNDARY_COMPONENT.pattern})\b[^.;:!?]{{0,75}}"
    rf"\b(?:contain|enclos|travers|enter|infiltrat|penetrat|breach|cross|"
    rf"interdigitat|span)\w*\b[^.;:!?]{{0,55}}"
    rf"\b(?:{_CAPSULE_CELLULAR_COMPONENT.pattern})\b",
    re.IGNORECASE,
)
_STRICT_ADIPOSE_INTERFACE_RELATION = re.compile(
    rf"\b(?:{_CAPSULE_CELLULAR_COMPONENT.pattern})\b[^.;:!?]{{0,95}}"
    rf"\b(?:enter|extend|infiltrat|penetrat|cross|interdigitat|travers|span)\w*\b"
    rf"[^.;:!?]{{0,65}}\b(?:{_ADIPOSE_COMPONENT.pattern})\b|"
    rf"\b(?:{_ADIPOSE_COMPONENT.pattern})\b[^.;:!?]{{0,80}}"
    rf"\b(?:contain|enclos|interspers|surround)\w*\b[^.;:!?]{{0,60}}"
    rf"\b(?:{_CAPSULE_CELLULAR_COMPONENT.pattern})\b",
    re.IGNORECASE,
)

_ATYPICAL_EPITHELIAL_DEPOSIT = re.compile(
    r"\b(?:epithelial\s+(?:cells?|clusters?|groups?|proliferation)|"
    r"glandular\s+(?:cells?|clusters?|proliferation)|irregular\s+glands?)\b"
    r"[^;\n]{0,360}\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|"
    r"mitotic\s+figures?|nuclear\s+crowd\w*|irregular\s+nuclear\s+"
    r"(?:membran\w*|contours?)|coarse(?:ly\s+clumped)?\s+chromatin|"
    r"nuclei\s+(?:are\s+)?(?:enlarged|variable)|prominent\s+nucleol\w*)\b|"
    r"\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|mitotic\s+figures?|"
    r"nuclear\s+crowd\w*|irregular\s+nuclear\s+(?:membran\w*|contours?)|"
    r"coarse(?:ly\s+clumped)?\s+chromatin|nuclei\s+(?:are\s+)?"
    r"(?:enlarged|variable)|prominent\s+nucleol\w*)\b"
    r"[^;\n]{0,360}\b(?:epithelial\s+(?:cells?|clusters?|groups?|proliferation)|"
    r"glandular\s+(?:cells?|clusters?|proliferation)|irregular\s+glands?)\b",
    re.IGNORECASE,
)

_HIGH_RISK_FEATURES: dict[str, re.Pattern[str]] = {
    "atypical_epithelial_deposit": _ATYPICAL_EPITHELIAL_DEPOSIT,
    "prominent_nucleoli": re.compile(
        r"\b(?:(?:prominent|conspicuous|eosinophilic)\w*(?:\s+\w+){0,2}\s+nucleol\w*|"
        r"nucleol\w*(?:\s+\w+){0,3}\s+(?:prominent|conspicuous|eosinophilic)\w*)\b",
        re.IGNORECASE,
    ),
    "necrosis": re.compile(r"\b(?:necrosis|necrotic)\b", re.IGNORECASE),
    "infiltrative_interface": re.compile(
        r"\b(?:infiltrat|invad|travers|cross|enter|extend|interdigitat|"
        r"penetrat|breach|span)\w*[^.;]{0,80}"
        r"\b(?:interface|adipose|adipocytes?|fat|smooth\s+muscle|stroma|capsul\w*|"
        r"collagen\w*|fibrous\s+(?:band|layer|boundary|rim)|"
        r"(?:mesothelial[-\s]+lined\s+)?(?:outer|free)"
        r"(?:\s+(?:outer|free))?\s+(?:tissue\s+)?surface|serosal\s+surface)\b|"
        r"\bcontinu\w*\s+across[^.;]{0,55}\b(?:capsul\w*|collagen\w*|"
        r"fibrous\s+(?:band|layer|boundary|rim)|interface|boundary)\b|"
        r"\b(?:interface|adipose|adipocytes?|fat|smooth\s+muscle|stroma|capsul\w*|"
        r"collagen\w*|fibrous\s+(?:band|layer|boundary|rim)|"
        r"(?:mesothelial[-\s]+lined\s+)?(?:outer|free)"
        r"(?:\s+(?:outer|free))?\s+(?:tissue\s+)?surface|serosal\s+surface)"
        r"\b[^.;]{0,80}"
        r"\b(?:infiltrat|invad|travers|cross|enter|extend|interdigitat|"
        r"penetrat|breach|span)\w*\b",
        re.IGNORECASE,
    ),
    "papillary_fibrovascular_core": re.compile(
        r"\bpapillar\w*.{0,80}\bfibrovascular\s+cores?\b|"
        r"\bfibrovascular\s+cores?\b.{0,80}\bpapillar\w*\b",
        re.IGNORECASE,
    ),
    "extreme_grade_morphology": re.compile(
        r"\b(?:rhabdoid|sarcomatoid|multinucleated\s+giant|"
        r"extreme(?:ly)?\s+pleomorph)\w*\b",
        re.IGNORECASE,
    ),
    # A direct epithelial-nerve/channel relationship is consequential enough
    # that a first-pass claim must also be independently re-transcribed.
    "perineural_relation": _STRICT_PERINEURAL_RELATION,
    "lymphovascular_relation": _STRICT_LYMPHOVASCULAR_RELATION,
}

_EPITHELIAL_LINEAGE_CONFIRMATION = re.compile(
    r"\b(?:epithelial|glandular|glands?)\b",
    re.IGNORECASE,
)
_LUMEN_ARCHITECTURE_CONFIRMATION = re.compile(
    r"\b(?:lumen|luminal|lumen[-\s]?like\s+space)\b[^.;]{0,100}"
    r"\b(?:surrounded|lined)\b[^.;]{0,55}\b(?:cells?|cellular\s+layer)\b|"
    r"\b(?:cells?|cellular\s+layer)\b[^.;]{0,55}\b(?:surround|line)\w*\b"
    r"[^.;]{0,100}\b(?:lumen|luminal)\b|"
    # Qwen sometimes puts the visible lumen and the cells around it in two
    # adjacent sentences: ``a lumen-like space ... . The surrounding cells``.
    # This is morphologically equivalent to ``surrounded by cells`` and must
    # not become a stochastic false negative merely because of punctuation.
    r"\b(?:lumen|luminal|lumen[-\s]?like\s+space)\b[^;\n]{0,220}"
    r"\b(?:surrounding|peripheral)\s+cells?\b",
    re.IGNORECASE,
)
_UNCERTAIN_LUMEN = re.compile(
    r"\b(?:may|might|could)\s+(?:represent|be)\b[^.;]{0,55}\b(?:lumina?|luminal)\b|"
    r"\b(?:possible|potential|indeterminate|equivocal)\b[^.;]{0,45}"
    r"\b(?:lumina?|luminal)\b",
    re.IGNORECASE,
)
_ATYPICAL_CYTOLOGY_CONFIRMATION = re.compile(
    r"\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|mitotic\s+(?:activity|figures?)|"
    r"nuclear\s+crowd\w*|irregular\s+nuclear\s+(?:contours?|membran\w*)|"
    r"enlarged\s+nuclei|coarse(?:ly\s+clumped)?\s+chromatin|"
    r"nuclei\s+(?:are\s+)?(?:enlarged|variable)|prominent\s+nucleol\w*)\b",
    re.IGNORECASE,
)

# Reconciliation changes can alter whether a high-risk positive clause is
# retained even when the vision prompts and models are unchanged.  Keep that
# semantic version in each cache record so affected high-risk entries can be
# rechecked without invalidating millions of ordinary neutral observations.
OBSERVATION_CONSENSUS_VERSION = "v9-local-adipose-interface-verification-20260907"
_ATYPICAL_CONSENSUS_COMPATIBLE_VERSIONS = {
    "v2-morphology-equivalent-lineage-20260906",
    "v6-focused-outer-surface-assessment-20260906",
    "v7-adipocyte-interface-consensus-20260907",
    OBSERVATION_CONSENSUS_VERSION,
}
_INFILTRATIVE_CONSENSUS_COMPATIBLE_VERSIONS = {
    OBSERVATION_CONSENSUS_VERSION,
}


def _has_positive_local_match(text: str, pattern: re.Pattern[str]) -> bool:
    """Require an unnegated local morphology mention for a verifier feature."""

    for match in pattern.finditer(text):
        prefix = re.split(r"[.;,]", text[max(0, match.start() - 60) : match.start()])[-1]
        suffix = re.split(r"[.;,]", text[match.end() : match.end() + 45])[0]
        if not (
            _LOCAL_NEGATION.search(prefix)
            or _LOCAL_NEGATION.search(match.group(0))
            or _LOCAL_POST_NEGATION.search(suffix)
        ):
            return True
    return False


def _has_positive_lumen_architecture(text: str) -> bool:
    for match in _LUMEN_ARCHITECTURE_CONFIRMATION.finditer(text):
        window = text[max(0, match.start() - 70) : match.end() + 30]
        if _UNCERTAIN_LUMEN.search(window):
            continue
        prefix = re.split(r"[;,]", text[max(0, match.start() - 60) : match.start()])[-1]
        if not _LOCAL_NEGATION.search(prefix):
            return True
    return False


def _positive_feature_names(text: str) -> set[str]:
    output: set[str] = set()
    for name, pattern in _HIGH_RISK_FEATURES.items():
        for match in pattern.finditer(text):
            prefix = re.split(r"[.;,]", text[max(0, match.start() - 60) : match.start()])[-1]
            suffix = re.split(r"[.;,]", text[match.end() : match.end() + 45])[0]
            # Reverse-order relation patterns can contain their negator after
            # match.start(), e.g. ``the fibrous band is not infiltrated``.
            # A caption may also negate the just-named relation as
            # ``crossing is not shown``; inspect that short suffix too.
            if not (
                _LOCAL_NEGATION.search(prefix)
                or _LOCAL_NEGATION.search(match.group(0))
                or _LOCAL_POST_NEGATION.search(suffix)
            ):
                output.add(name)
                break
    return output


def _microinvasion_context_names(text: str) -> set[str]:
    """Return neutral same-ROI structure pairs that merit retranscription."""

    output: set[str] = set()
    if _has_positive_local_match(text, _EPITHELIAL_COMPONENT):
        if _has_positive_local_match(text, _NERVE_COMPONENT):
            output.add("perineural_context_v1")
        if _has_positive_local_match(text, _VASCULAR_COMPONENT):
            output.add("lymphovascular_context_v1")
    if (
        _has_positive_local_match(text, _CAPSULE_CELLULAR_COMPONENT)
        and _has_positive_local_match(text, _CAPSULE_BOUNDARY_COMPONENT)
    ):
        output.add("capsule_interface_context_v1")
    if (
        _has_positive_local_match(text, _CAPSULE_CELLULAR_COMPONENT)
        and _has_positive_local_match(text, _ADIPOSE_COMPONENT)
    ):
        output.add("adipose_interface_context_v1")
    return output


def _has_external_blank_glass(png: bytes) -> bool:
    """Require a large glass field that reaches an image side, not an internal cleft."""

    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    if pixels.ndim != 3 or min(pixels.shape[:2]) < 8:
        return False
    # H&E scanner background is slightly pink/blue rather than pure white.
    # Requiring all RGB channels above 205 rejects pale stroma and most luminal
    # material while retaining the broad glass field at a true tissue edge.
    blank = np.min(pixels, axis=2) > 205
    blank_fraction = float(blank.mean())
    border_fraction = max(
        float(blank[0, :].mean()),
        float(blank[-1, :].mean()),
        float(blank[:, 0].mean()),
        float(blank[:, -1].mean()),
    )
    return blank_fraction >= 0.15 and border_fraction >= 0.50


def _is_outer_surface_candidate(roi: ROI, png: bytes) -> bool:
    """Use navigation metadata only to decide whether a surface check is worth running."""

    navigation = roi.tissue_navigation
    if navigation is None or roi.magnification not in {"20x", "40x"}:
        return False
    return (
        navigation.selection_route == "outer_interface_descendant"
        and navigation.class_fractions.get("artifact_or_background", 0.0) >= 0.02
        and _has_external_blank_glass(png)
    )


def _strip_synthetic_outer_surface_relation(
    content: ObservationContent,
) -> ObservationContent:
    morphology = _SYNTHETIC_VERIFIED_SURFACE_CLAUSE.sub(" ", content.morphology)
    morphology = re.sub(r"\s{2,}", " ", morphology).strip()
    return content if morphology == content.morphology else content.model_copy(
        update={"morphology": morphology}
    )


def _surface_epithelial_candidate(content: ObservationContent) -> bool:
    """Exclude bland surface epithelium before any serosal relation is materialized."""

    if not _has_positive_local_match(content.morphology, _EPITHELIAL_COMPONENT):
        return False
    feature_count = sum(
        _has_positive_local_match(content.morphology, pattern)
        for pattern in _SURFACE_EPITHELIAL_RISK_FEATURES
    )
    return feature_count >= 2


def _risk_features(
    content: ObservationContent,
    *,
    surface_candidate: bool = False,
) -> set[str]:
    output = _positive_feature_names(content.morphology)
    output.update(_microinvasion_context_names(content.morphology))
    if (
        surface_candidate
        and _surface_epithelial_candidate(content)
        and _has_positive_local_match(content.morphology, _OUTER_SURFACE_CONTEXT)
    ):
        output.add("outer_surface_context_v1")
    return output


def _append_verified_outer_surface_relation(
    primary: ObservationContent,
    reconciled: ObservationContent,
    assessment: OuterSurfaceAssessment,
) -> ObservationContent:
    """Materialize only a two-model, explicitly surface-reaching relation."""

    if (
        assessment.boundary_identity != "anatomic_free_surface"
        or assessment.epithelial_relation
        not in {"reaches_surface", "traverses_surface"}
        or _EXPLICIT_SURFACE_SEPARATION.search(primary.morphology)
        or not _surface_epithelial_candidate(primary)
        or not _has_positive_local_match(reconciled.morphology, _EPITHELIAL_COMPONENT)
    ):
        return reconciled
    verb = (
        "traverses" if assessment.epithelial_relation == "traverses_surface" else "reaches"
    )
    clause = f"The epithelial proliferation visibly {verb} the true outer free tissue surface."
    morphology = " ".join([reconciled.morphology, clause]).strip()[:900]
    return sanitize_neutral_observation(
        reconciled.model_copy(update={"morphology": morphology}), "20x"
    )


def _corroborated_feature_names(text: str) -> set[str]:
    """Allow morphology-equivalent wording in the independent transcription."""

    output = _positive_feature_names(text)
    if (
        _has_positive_local_match(text, _ATYPICAL_CYTOLOGY_CONFIRMATION)
        and (
            _has_positive_local_match(text, _EPITHELIAL_LINEAGE_CONFIRMATION)
            or _has_positive_lumen_architecture(text)
        )
    ):
        output.add("atypical_epithelial_deposit")
    return output


def reconcile_observation_consensus(
    primary: ObservationContent,
    verification: ObservationContent,
    magnification: str,
) -> tuple[ObservationContent, list[str]]:
    """Remove high-risk positive claims absent from an independent transcription."""

    primary_features = _positive_feature_names(primary.morphology)
    verification_features = _corroborated_feature_names(verification.morphology)
    rejected = sorted(primary_features - verification_features)
    if not rejected:
        return primary, []
    rejected_patterns = [_HIGH_RISK_FEATURES[name] for name in rejected]
    clauses = [
        item.strip()
        for item in re.split(r"(?<=[.!?;])\s+|[\r\n]+", primary.morphology)
        if item.strip()
    ]
    reject_multisentence_epithelial = "atypical_epithelial_deposit" in rejected

    def clause_was_rejected(clause: str) -> bool:
        if any(
            pattern.search(clause)
            and name in _positive_feature_names(clause)
            for name, pattern in zip(rejected, rejected_patterns)
        ):
            return True
        if reject_multisentence_epithelial and (
            _EPITHELIAL_LINEAGE_CONFIRMATION.search(clause)
            or _ATYPICAL_CYTOLOGY_CONFIRMATION.search(clause)
        ):
            # A multi-sentence claim may put epithelial identity in one
            # sentence and its nuclear atypia in the next. If the independent
            # transcriber does not reproduce both parts, retaining either
            # sentence would let the downstream node aggregator reconstruct
            # the rejected carcinoma claim again.
            return True
        return False

    kept = [clause for clause in clauses if not clause_was_rejected(clause)]
    morphology = " ".join(kept).strip()
    if not morphology:
        morphology = verification.morphology
    limitations = list(primary.limitations)
    limitations.append(
        "Independent neutral retranscription did not corroborate: "
        + ", ".join(rejected)
        + "."
    )
    reconciled = primary.model_copy(
        update={"morphology": morphology[:900], "limitations": limitations[:4]}
    )
    return sanitize_neutral_observation(reconciled, magnification), rejected


def _matching_positive_clause(
    text: str, pattern: re.Pattern[str]
) -> str | None:
    """Extract one independently observed, unnegated relation clause."""

    clauses = [
        item.strip()
        for item in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text)
        if item.strip()
    ]
    for clause in clauses:
        if pattern.search(clause) and _has_positive_local_match(clause, pattern):
            return clause
    return None


def _merge_verified_microinvasion_relations(
    primary: ObservationContent,
    verification: ObservationContent,
    reconciled: ObservationContent,
    magnification: str,
) -> ObservationContent:
    """Add a verifier-only relation only after primary structure localization.

    The primary transcriber must independently see both generic structures in
    the same ROI.  The verifier must then explicitly state their strict local
    spatial relationship.  Mere co-occurrence, retrieval scores, and cellular
    model labels can never add a relation.
    """

    if magnification not in {"20x", "40x"}:
        return reconciled
    contexts = _microinvasion_context_names(primary.morphology)
    candidates = (
        (
            "perineural_context_v1",
            "perineural_relation",
            _STRICT_PERINEURAL_RELATION,
        ),
        (
            "lymphovascular_context_v1",
            "lymphovascular_relation",
            _STRICT_LYMPHOVASCULAR_RELATION,
        ),
        (
            "adipose_interface_context_v1",
            "infiltrative_interface",
            _STRICT_ADIPOSE_INTERFACE_RELATION,
        ),
        (
            "capsule_interface_context_v1",
            "infiltrative_interface",
            _STRICT_CAPSULE_INTERFACE_RELATION,
        ),
    )
    additions: list[str] = []
    retained_features = _positive_feature_names(reconciled.morphology)
    for context_name, relation_name, pattern in candidates:
        if context_name not in contexts or relation_name in retained_features:
            continue
        clause = _matching_positive_clause(verification.morphology, pattern)
        if clause is not None:
            additions.append(clause)
    if not additions:
        return reconciled
    morphology = " ".join([reconciled.morphology, *additions]).strip()[:900]
    merged = reconciled.model_copy(update={"morphology": morphology})
    return sanitize_neutral_observation(merged, magnification)


def neutral_observation_errors(
    value: ObservationContent, magnification: str
) -> dict[str, str]:
    """Reject diagnostic conclusions and resolution-inappropriate claims."""

    errors: dict[str, str] = {}
    if _DIAGNOSTIC_LANGUAGE.search(value.morphology):
        errors["morphology"] = (
            "must be diagnosis-free morphology; remove disease, grade, stage, metastasis, "
            "and margin conclusions"
        )
    if magnification == "5x" and _LOW_POWER_OVERCLAIM.search(value.morphology):
        errors["magnification"] = (
            "5x cannot support mitotic, nucleolar, chromatin, or nuclear-grade claims"
        )
    if _CONCLUSION_LANGUAGE.search(value.morphology):
        errors["conclusion"] = (
            "remove diagnostic inference, clinical-history speculation, and recommendations; "
            "retain only pixel-visible morphology"
        )
    return errors


def sanitize_neutral_observation(
    value: ObservationContent, magnification: str
) -> ObservationContent:
    """Keep descriptive sentences and remove downstream diagnostic conclusions."""

    clauses = [
        item.strip()
        for item in re.split(
            r"(?<=[.!?;])\s+|[\r\n]+|(?<=[,;])\s+(?=(?:and\s+)?[A-Z])",
            value.morphology,
        )
        if item.strip()
    ]
    kept: list[str] = []
    for clause in clauses:
        # Preserve the pixel-visible architecture while removing an inferred
        # organ modifier. Dropping the whole sentence would also discard the
        # useful neutral finding (for example, crowded irregular glands).
        clause = _INFERRED_SITE_MODIFIER.sub("", clause)
        clause = re.sub(r"\s{2,}", " ", clause).strip()
        if not clause:
            continue
        if _DIAGNOSTIC_LANGUAGE.search(clause) or _CONCLUSION_LANGUAGE.search(clause):
            continue
        if magnification == "5x" and _LOW_POWER_OVERCLAIM.search(clause):
            continue
        kept.append(clause)
    morphology = " ".join(kept).strip()
    if not morphology:
        limitations = list(value.limitations)
        limitations.append(
            "The generated text contained only diagnostic or resolution-inappropriate claims."
        )
        return value.model_copy(
            update={
                "image_quality": "limited",
                "morphology": (
                    "Tissue is visible, but no reliable diagnosis-free morphology could be "
                    "retained from the transcription."
                ),
                "limitations": limitations[:4],
            }
        )
    return value.model_copy(update={"morphology": morphology[:900]})


def image_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(image_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def bytes_data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_observation_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize common Patho-R1 key/style deviations without inventing fields."""

    key_map = {
        "imagequality": "image_quality",
        "morphology": "morphology",
        "description": "morphology",
        "morphologicdescription": "morphology",
        "limitations": "limitations",
    }
    dictionaries: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            dictionaries.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(candidate)
    source = max(
        dictionaries,
        key=lambda item: sum(
            re.sub(r"[^a-z]", "", str(key).casefold())
            in {*key_map, "visiblefindings", "tissuecontext"}
            for key in item
        ),
        default=candidate,
    )
    normalized: dict[str, Any] = {}
    context = ""
    findings: list[str] = []
    for key, value in source.items():
        compact = re.sub(r"[^a-z]", "", str(key).casefold())
        canonical = key_map.get(compact)
        if canonical:
            normalized[canonical] = value
        elif compact == "tissuecontext":
            context = str(value).strip()
        elif compact == "visiblefindings":
            values = value if isinstance(value, list) else [value]
            findings = [str(item).strip() for item in values if str(item).strip()]
    if not str(normalized.get("morphology") or "").strip():
        legacy_parts = [item for item in [context, "; ".join(findings)] if item]
        if legacy_parts:
            normalized["morphology"] = " ".join(legacy_parts)
    # Patho-R1 sometimes emits a valid full morphology paragraph as the value
    # of its only ``image_quality`` key.  Preserve that paragraph before the
    # categorical quality normalizer below reduces it to ``adequate``.
    raw_quality = normalized.get("image_quality")
    if (
        not str(normalized.get("morphology") or "").strip()
        and isinstance(raw_quality, str)
        and (len(raw_quality.split()) >= 8 or any(mark in raw_quality for mark in ".;"))
    ):
        normalized["morphology"] = raw_quality.strip()
    if "morphology" in normalized:
        value = normalized["morphology"]
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if str(item).strip())
        normalized["morphology"] = str(value).strip()[:900]
    limitations = normalized.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    if isinstance(limitations, list):
        normalized["limitations"] = [
            str(item).strip()[:300] for item in limitations[:4] if str(item).strip()
        ]
    quality = str(normalized.get("image_quality") or "").casefold()
    for allowed in ("inadequate", "limited", "adequate"):
        if allowed in quality:
            normalized["image_quality"] = allowed
            break
    return normalized


class PathoObserver:
    def __init__(
        self,
        generator: StructuredGenerator,
        config: AgentConfig,
        verifier_generator: StructuredGenerator | None = None,
    ) -> None:
        self.generator = generator
        self.verifier_generator = verifier_generator
        self.config = config
        self.audit: list[dict[str, Any]] = []
        self.requests = 0
        self.calls = 0
        self.failures = 0
        self.cache_hits = 0
        self.cache_writes = 0
        self.sanitized_outputs = 0
        self.verification_calls = 0
        self.verification_failures = 0
        self.verification_rejections = 0
        self.surface_verification_calls = 0
        self.surface_verification_failures = 0
        self.surface_verification_positives = 0
        # Disk caching avoids repeated vision calls across runs. Keep the
        # canonical content seen by this process as well: a shared cache file
        # can be refreshed by another worker between two task questions, but
        # identical pixels within one run must never acquire two descriptions.
        self._process_observation_cache: OrderedDict[str, ObservationContent] = (
            OrderedDict()
        )
        self._process_observation_cache_limit = 8192
        cache_signature = hashlib.sha256(
            json.dumps(
                {
                    "prompt": OBSERVATION_PROMPT_VERSION,
                    "verification_prompt": OBSERVATION_VERIFICATION_CACHE_FAMILY_VERSION,
                    "model": config.patho_model,
                    "verification_model": config.qwen_model,
                    "seed": config.generation_seed,
                    "verification_seed": config.generation_seed + 1,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        self.cache_root = (
            config.cache_root
            / "planpath.core"
            / "neutral_roi_observations"
            / cache_signature
        )

    def _cache_path(self, png: bytes, magnification: str) -> Path:
        key = hashlib.sha256(
            OBSERVATION_PROMPT_VERSION.encode("utf-8")
            + b"\0"
            + OBSERVATION_VERIFICATION_CACHE_FAMILY_VERSION.encode("utf-8")
            + b"\0"
            + self.config.patho_model.encode("utf-8")
            + b"\0"
            + self.config.qwen_model.encode("utf-8")
            + b"\0"
            + str(self.config.generation_seed).encode("ascii")
            + b"\0"
            + magnification.encode("ascii")
            + b"\0"
            + png
        ).hexdigest()
        return self.cache_root / key[:2] / f"{key}.json"

    def _verify_high_risk_content(
        self,
        content: ObservationContent,
        png: bytes,
        magnification: str,
        *,
        surface_candidate: bool = False,
    ) -> tuple[ObservationContent, list[str], list[dict[str, Any]], set[str]]:
        """Independently re-transcribe only newly detected high-risk claims."""

        risk_features = _risk_features(
            content, surface_candidate=surface_candidate
        )
        if not risk_features or self.verifier_generator is None:
            return content, [], [], set()
        self.verification_calls += 1
        try:
            verification_result = self.verifier_generator.generate(
                ObservationContent,
                observation_verification_messages(
                    bytes_data_url(png), magnification
                ),
                self.config.patho_max_tokens,
                "neutral_observation_qwen_verification",
                candidate_normalizer=normalize_observation_candidate,
            )
            verification = sanitize_neutral_observation(
                verification_result.value, magnification
            )
            reconciled, rejected_features = reconcile_observation_consensus(
                content, verification, magnification
            )
            reconciled = _merge_verified_microinvasion_relations(
                content, verification, reconciled, magnification
            )
            if (
                "outer_surface_context_v1" in risk_features
                and magnification in {"20x", "40x"}
            ):
                self.surface_verification_calls += 1
                try:
                    surface_result = self.verifier_generator.generate(
                        OuterSurfaceAssessment,
                        outer_surface_verification_messages(
                            bytes_data_url(png), magnification
                        ),
                        self.config.patho_max_tokens,
                        "neutral_outer_surface_verification",
                    )
                    before = reconciled.morphology
                    reconciled = _append_verified_outer_surface_relation(
                        content, reconciled, surface_result.value
                    )
                    self.surface_verification_positives += (
                        reconciled.morphology != before
                    )
                    verification_result.audit.extend(surface_result.audit)
                except Exception as error:
                    self.surface_verification_failures += 1
                    verification_result.audit.extend(getattr(error, "audit", []))
            self.verification_rejections += len(rejected_features)
            return (
                reconciled,
                rejected_features,
                verification_result.audit,
                risk_features,
            )
        except Exception as error:
            self.verification_failures += 1
            verification_audit = list(getattr(error, "audit", []))
            empty_verification = ObservationContent(
                image_quality="limited",
                morphology=(
                    "The independent neutral retranscription was unavailable; no high-risk "
                    "positive morphology is retained."
                ),
                limitations=[f"{type(error).__name__}: {error}"[:300]],
            )
            reconciled, rejected_features = reconcile_observation_consensus(
                content, empty_verification, magnification
            )
            self.verification_rejections += len(rejected_features)
            return reconciled, rejected_features, verification_audit, risk_features

    def _one(
        self,
        selected: SelectedROI,
        evidence_id: str,
    ) -> tuple[ROIObservation, list[dict[str, Any]]]:
        png = image_png_bytes(selected.image)
        cache_path = self._cache_path(png, selected.roi.magnification)
        surface_candidate = _is_outer_surface_candidate(selected.roi, png)
        cache_key = str(cache_path) + ("#outer-surface" if surface_candidate else "")
        process_cached = self._process_observation_cache.get(cache_key)
        if process_cached is not None:
            self._process_observation_cache.move_to_end(cache_key)
            self.cache_hits += 1
            return (
                ROIObservation(
                    **process_cached.model_dump(),
                    evidence_id=evidence_id,
                    slide_id=selected.roi.slide_id,
                    roi_id=selected.roi.roi_id,
                    magnification=selected.roi.magnification,
                    parent_roi_id=selected.roi.parent_roi_id,
                ),
                [],
            )
        if cache_path.is_file():
            cached = read_json(cache_path)
            if (
                cached.get("prompt_version") == OBSERVATION_PROMPT_VERSION
                and cached.get("patho_model") == self.config.patho_model
                and cached.get("generation_seed") == self.config.generation_seed
            ):
                content = ObservationContent.model_validate(cached["observation"])
                if not surface_candidate:
                    stripped = _strip_synthetic_outer_surface_relation(content)
                    if stripped.morphology != content.morphology:
                        content = stripped
                        cached["observation"] = content.model_dump(mode="json")
                        write_json_atomic(cache_path, cached)
                        self.cache_writes += 1
                sanitized = sanitize_neutral_observation(
                    content, selected.roi.magnification
                )
                if sanitized.morphology != content.morphology:
                    content = sanitized
                    cached["observation"] = content.model_dump(mode="json")
                    write_json_atomic(cache_path, cached)
                    self.sanitized_outputs += 1
                verification_audit: list[dict[str, Any]] = []
                checked_features = {
                    str(item)
                    for item in cached.get("consensus_checked_features", [])
                }
                cached_rejected_features = {
                    str(item)
                    for item in cached.get("consensus_rejected_features", [])
                }
                primary_payload = cached.get("primary_observation")
                primary_content = (
                    sanitize_neutral_observation(
                        ObservationContent.model_validate(primary_payload),
                        selected.roi.magnification,
                    )
                    if isinstance(primary_payload, dict)
                    else content
                )
                if not _surface_epithelial_candidate(primary_content):
                    stripped = _strip_synthetic_outer_surface_relation(content)
                    if stripped.morphology != content.morphology:
                        content = stripped
                        cached["observation"] = content.model_dump(mode="json")
                        write_json_atomic(cache_path, cached)
                        self.cache_writes += 1
                consensus_version = str(
                    cached.get("consensus_algorithm_version") or ""
                )
                legacy_high_risk_consensus = (
                    (
                        consensus_version
                        not in _ATYPICAL_CONSENSUS_COMPATIBLE_VERSIONS
                        and "atypical_epithelial_deposit"
                        in (checked_features | cached_rejected_features)
                    )
                    or (
                        consensus_version
                        not in _INFILTRATIVE_CONSENSUS_COMPATIBLE_VERSIONS
                        and {
                            "infiltrative_interface",
                            "adipose_interface_context_v1",
                        }
                        & (checked_features | cached_rejected_features)
                    )
                )
                if legacy_high_risk_consensus:
                    # An older consensus may already have removed the primary
                    # epithelial clause. Recover it once from Patho-R1 when no
                    # preserved primary transcription exists; accepted legacy
                    # entries can be reverified directly from their intact
                    # cached content.
                    primary_audit: list[dict[str, Any]] = []
                    if (
                        not isinstance(primary_payload, dict)
                        and not (
                            {
                                "atypical_epithelial_deposit",
                                "infiltrative_interface",
                            }
                            & _positive_feature_names(primary_content.morphology)
                        )
                    ):
                        self.calls += 1
                        primary_result = self.generator.generate(
                            ObservationContent,
                            observation_messages(
                                bytes_data_url(png), selected.roi.magnification
                            ),
                            self.config.patho_max_tokens,
                            "patho_observation_consensus_migration",
                            candidate_normalizer=normalize_observation_candidate,
                        )
                        primary_content = sanitize_neutral_observation(
                            primary_result.value, selected.roi.magnification
                        )
                        primary_audit = list(primary_result.audit)
                    (
                        content,
                        newly_rejected,
                        new_verification_audit,
                        newly_checked,
                    ) = self._verify_high_risk_content(
                        primary_content,
                        png,
                        selected.roi.magnification,
                        surface_candidate=surface_candidate,
                    )
                    verification_audit = primary_audit + new_verification_audit
                    checked_features = set(newly_checked)
                    cached_rejected_features = set(newly_rejected)
                    cached.update(
                        primary_observation=primary_content.model_dump(mode="json"),
                        observation=content.model_dump(mode="json"),
                        consensus_algorithm_version=OBSERVATION_CONSENSUS_VERSION,
                        verification_prompt_version=OBSERVATION_VERIFICATION_PROMPT_VERSION,
                        verification_model=self.config.qwen_model,
                        verification_seed=self.config.generation_seed + 1,
                        consensus_checked_features=sorted(checked_features),
                        consensus_rejected_features=sorted(
                            cached_rejected_features
                        ),
                    )
                    write_json_atomic(cache_path, cached)
                    self.cache_writes += 1
                risk_features = _risk_features(
                    primary_content,
                    surface_candidate=surface_candidate,
                )
                # Detector improvements must not silently trust an older
                # cache entry that was never sent through the independent
                # verifier. Recheck only entries containing a newly detected
                # high-risk feature; ordinary cached observations stay hot.
                if risk_features - checked_features and self.verifier_generator is not None:
                    (
                        content,
                        newly_rejected,
                        verification_audit,
                        newly_checked,
                    ) = self._verify_high_risk_content(
                        primary_content,
                        png,
                        selected.roi.magnification,
                        surface_candidate=surface_candidate,
                    )
                    cached.update(
                        primary_observation=primary_content.model_dump(mode="json"),
                        observation=content.model_dump(mode="json"),
                        consensus_algorithm_version=OBSERVATION_CONSENSUS_VERSION,
                        verification_prompt_version=OBSERVATION_VERIFICATION_PROMPT_VERSION,
                        verification_model=self.config.qwen_model,
                        verification_seed=self.config.generation_seed + 1,
                        consensus_checked_features=sorted(
                            checked_features | newly_checked
                        ),
                        consensus_rejected_features=sorted(
                            set(newly_rejected)
                        ),
                    )
                    write_json_atomic(cache_path, cached)
                    self.cache_writes += 1
                self.cache_hits += 1
                self._process_observation_cache[cache_key] = content
                self._process_observation_cache.move_to_end(cache_key)
                if len(self._process_observation_cache) > self._process_observation_cache_limit:
                    self._process_observation_cache.popitem(last=False)
                observation = ROIObservation(
                    **content.model_dump(),
                    evidence_id=evidence_id,
                    slide_id=selected.roi.slide_id,
                    roi_id=selected.roi.roi_id,
                    magnification=selected.roi.magnification,
                    parent_roi_id=selected.roi.parent_roi_id,
                )
                return observation, verification_audit
        self.calls += 1
        result = self.generator.generate(
            ObservationContent,
            observation_messages(
                bytes_data_url(png), selected.roi.magnification
            ),
            self.config.patho_max_tokens,
            "patho_observation",
            candidate_normalizer=normalize_observation_candidate,
        )
        content = sanitize_neutral_observation(
            result.value, selected.roi.magnification
        )
        primary_content = content
        if content.morphology != result.value.morphology:
            self.sanitized_outputs += 1
        (
            content,
            rejected_features,
            verification_audit,
            checked_features,
        ) = self._verify_high_risk_content(
            primary_content,
            png,
            selected.roi.magnification,
            surface_candidate=surface_candidate,
        )
        write_json_atomic(
            cache_path,
            {
                "prompt_version": OBSERVATION_PROMPT_VERSION,
                "verification_prompt_version": OBSERVATION_VERIFICATION_PROMPT_VERSION,
                "patho_model": self.config.patho_model,
                "verification_model": self.config.qwen_model,
                "generation_seed": self.config.generation_seed,
                "verification_seed": self.config.generation_seed + 1,
                "magnification": selected.roi.magnification,
                "image_sha256": hashlib.sha256(png).hexdigest(),
                "primary_observation": primary_content.model_dump(mode="json"),
                "observation": content.model_dump(mode="json"),
                "consensus_algorithm_version": OBSERVATION_CONSENSUS_VERSION,
                "consensus_checked_features": sorted(checked_features),
                "consensus_rejected_features": rejected_features,
            },
        )
        self.cache_writes += 1
        self._process_observation_cache[cache_key] = content
        self._process_observation_cache.move_to_end(cache_key)
        if len(self._process_observation_cache) > self._process_observation_cache_limit:
            self._process_observation_cache.popitem(last=False)
        observation = ROIObservation(
            **content.model_dump(),
            evidence_id=evidence_id,
            slide_id=selected.roi.slide_id,
            roi_id=selected.roi.roi_id,
            magnification=selected.roi.magnification,
            parent_roi_id=selected.roi.parent_roi_id,
        )
        return observation, result.audit + verification_audit

    def describe(
        self,
        selected: list[SelectedROI],
        evidence_start: int,
    ) -> list[ROIObservation]:
        jobs = [
            (item, f"E{evidence_start + index:04d}")
            for index, item in enumerate(selected)
        ]
        with ThreadPoolExecutor(
            max_workers=min(self.config.patho_concurrency, max(1, len(jobs)))
        ) as pool:
            futures = [
                pool.submit(self._one, item, evidence_id)
                for item, evidence_id in jobs
            ]
            results: list[tuple[ROIObservation, list[dict[str, Any]]]] = []
            for (selected, evidence_id), future in zip(jobs, futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    self.failures += 1
                    self.audit.extend(getattr(error, "audit", []))
                    results.append(
                        (
                            ROIObservation(
                                evidence_id=evidence_id,
                                slide_id=selected.roi.slide_id,
                                roi_id=selected.roi.roi_id,
                                magnification=selected.roi.magnification,
                                parent_roi_id=selected.roi.parent_roi_id,
                                image_quality="inadequate",
                                morphology="No valid neutral morphology transcription was returned.",
                                limitations=[f"{type(error).__name__}: {error}"[:500]],
                            ),
                            [],
                        )
                    )
        for _, item_audit in results:
            self.audit.extend(item_audit)
        self.requests += len(results)
        return [item for item, _ in results]


class VisualAnalyzer:
    def __init__(
        self,
        selector: ROISelector,
        observer: PathoObserver,
        histoplus: Any | None = None,
    ) -> None:
        self.selector = selector
        self.observer = observer
        self.histoplus = histoplus
        observer_config = getattr(observer, "config", None)
        self._patient_visual_cache_enabled = bool(
            getattr(observer_config, "patient_visual_cache_enabled", False)
        )
        self._cached_patient_id: str | None = None
        self._low_power_pack_cache: dict[
            tuple[Any, ...], tuple[list[ROI], list[ROIObservation]]
        ] = {}
        self._zoom_pack_cache: dict[
            tuple[Any, ...], tuple[list[ROI], list[ROIObservation]]
        ] = {}
        self._patient_visual_cache_hits = 0
        self._patient_visual_cache_writes = 0

    def _prepare_patient_cache(self, slide: dict[str, Any]) -> None:
        if not self._patient_visual_cache_enabled:
            return
        patient_id = str(slide.get("patient_id") or "")
        if patient_id == self._cached_patient_id:
            return
        self._cached_patient_id = patient_id
        self._low_power_pack_cache.clear()
        self._zoom_pack_cache.clear()

    @staticmethod
    def _plan_visual_key(
        slide: dict[str, Any], plan: EvidencePlan
    ) -> tuple[Any, ...]:
        """Identify plans whose neutral ROI selection is exactly reusable."""

        policy = routing_policy(str(slide.get("cancer") or ""), plan)
        plan_text = " ".join(
            [plan.target_finding, *plan.question_attributes, plan.evidence_goal]
        ).casefold()
        renal_grade_expansion = "who/isup" in plan_text and "grade" in plan_text
        global_interface_ranking = any(
            "capsule interface" in str(value).casefold()
            or "tumor-capsule interface" in str(value).casefold()
            or "gastric wall compartment" in str(value).casefold()
            or "deepest anatomic compartment" in str(value).casefold()
            for value in [plan.target_finding, *plan.question_attributes]
        )
        return (
            plan.low_power_query,
            plan.high_power_query,
            policy.target_labels,
            policy.interface_pairs,
            policy.histoplus_policy,
            policy.histoplus_targets,
            renal_grade_expansion,
            global_interface_ranking,
        )

    @staticmethod
    def _rebase_pack(
        cached: tuple[list[ROI], list[ROIObservation]], evidence_start: int
    ) -> tuple[list[ROI], list[ROIObservation]]:
        rois, observations = cached
        rebased = [
            item.model_copy(
                update={"evidence_id": f"E{evidence_start + index:04d}"}
            )
            for index, item in enumerate(observations)
        ]
        return list(rois), rebased

    @property
    def metrics(self) -> dict[str, int]:
        metrics = {
            **self.selector.metrics,
            "patient_visual_cache_hits": self._patient_visual_cache_hits,
            "patient_visual_cache_writes": self._patient_visual_cache_writes,
            "patho_requests": self.observer.requests,
            "patho_calls": self.observer.calls,
            "qwen_visual_verification_calls": self.observer.verification_calls,
            "qwen_surface_verification_calls": self.observer.surface_verification_calls,
            "qwen_surface_verification_failures": self.observer.surface_verification_failures,
            "qwen_surface_verification_positives": self.observer.surface_verification_positives,
            "patho_failures": self.observer.failures + self.observer.verification_failures,
            "qwen_visual_verification_failures": self.observer.verification_failures,
            "neutral_consensus_rejections": self.observer.verification_rejections,
            "neutral_observation_cache_hits": self.observer.cache_hits,
            "neutral_observation_cache_writes": self.observer.cache_writes,
            "neutral_observation_sanitized": self.observer.sanitized_outputs,
        }
        if self.histoplus is not None:
            metrics.update(self.histoplus.metrics)
        return metrics

    @property
    def audit(self) -> list[dict[str, Any]]:
        output = list(self.observer.audit)
        if self.histoplus is not None:
            output.extend(
                {"phase": "histoplus", **item} for item in self.histoplus.audit
            )
        return output

    def low_power(
        self,
        slide: dict[str, Any],
        plan: EvidencePlan,
        evidence_start: int,
    ) -> tuple[list[ROI], list[ROIObservation]]:
        self._prepare_patient_cache(slide)
        cache_key = (
            str(slide.get("slide_id") or ""),
            *self._plan_visual_key(slide, plan),
        )
        if self._patient_visual_cache_enabled and cache_key in self._low_power_pack_cache:
            self._patient_visual_cache_hits += 1
            return self._rebase_pack(
                self._low_power_pack_cache[cache_key], evidence_start
            )
        selected = self.selector.select_low(slide, plan.low_power_query, plan)
        observations = self.observer.describe(selected, evidence_start)
        output = ([item.roi for item in selected], observations)
        if self._patient_visual_cache_enabled:
            self._low_power_pack_cache[cache_key] = output
            self._patient_visual_cache_writes += 1
        return output

    def zoom(
        self,
        slide: dict[str, Any],
        parents: list[ROI],
        magnification: str,
        plan: EvidencePlan,
        evidence_start: int,
    ) -> tuple[list[ROI], list[ROIObservation]]:
        self._prepare_patient_cache(slide)
        cache_key = (
            str(slide.get("slide_id") or ""),
            magnification,
            *self._plan_visual_key(slide, plan),
            tuple(
                (item.roi_id, tuple(item.coordinates), item.magnification)
                for item in parents
            ),
        )
        if self._patient_visual_cache_enabled and cache_key in self._zoom_pack_cache:
            self._patient_visual_cache_hits += 1
            return self._rebase_pack(
                self._zoom_pack_cache[cache_key], evidence_start
            )
        interface_target = any(
            "capsule interface" in str(value).casefold()
            or "tumor-capsule interface" in str(value).casefold()
            or "gastric wall compartment" in str(value).casefold()
            or "deepest anatomic compartment" in str(value).casefold()
            for value in [plan.target_finding, *plan.question_attributes]
        )
        selected = self.selector.select_high(
            slide,
            parents,
            magnification,
            plan.high_power_query,
            ensure_parent_diversity=not interface_target,
            plan=plan,
        )
        observations = self.observer.describe(selected, evidence_start)
        if self.histoplus is not None:
            cellular = self.histoplus.analyze_selected(slide, selected, plan)
            observations = [
                item.model_copy(
                    update={"cellular_analysis": cellular.get(item.roi_id)}
                )
                for item in observations
            ]
        output = ([item.roi for item in selected], observations)
        if self._patient_visual_cache_enabled:
            self._zoom_pack_cache[cache_key] = output
            self._patient_visual_cache_writes += 1
        return output
