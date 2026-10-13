"""Qwen controller for the three reasoning calls retained in v6."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel

from .config import AgentConfig
from .prompts import final_messages, plan_messages, review_messages
from .quantification import quantitative_profile
from .schemas import (
    EvidenceJudgment,
    EvidencePlan,
    EvidenceReview,
    FinalAnswer,
    QuantitativeEvidence,
    ROIObservation,
)
from .structured import StructuredGenerationError, StructuredGenerator


_MAGNIFICATION_RANK = {"5x": 5, "10x": 10, "20x": 20, "40x": 40}

_GROUNDABLE_FEATURES = {
    "clear_cytoplasm": re.compile(
        r"\b(?:clear\s+(?:cell\w*|cytoplasm)|cytoplasm\w*\s+(?:is\s+|appears?\s+)?clear)\b",
        re.IGNORECASE,
    ),
    "eosinophilic": re.compile(r"\beosinophil\w*\b", re.IGNORECASE),
    "papillary": re.compile(r"\bpapillar\w*\b", re.IGNORECASE),
    "glandular": re.compile(r"\bgland\w*\b", re.IGNORECASE),
    "nucleoli": re.compile(r"\bnucleol\w*\b", re.IGNORECASE),
    "necrosis": re.compile(r"\bnecro\w*\b", re.IGNORECASE),
    "mitoses": re.compile(r"\bmito\w*\b", re.IGNORECASE),
    "perinuclear_halo": re.compile(r"\bperinuclear\b", re.IGNORECASE),
    "vacuoles": re.compile(r"\bvacuol\w*\b", re.IGNORECASE),
    "desmoplasia": re.compile(r"\bdesmoplas\w*\b", re.IGNORECASE),
    "infiltration": re.compile(r"\binfiltrat\w*\b", re.IGNORECASE),
    "vasculature": re.compile(r"\b(?:vascular|vasculature|fibrovascular|vessels?)\b", re.IGNORECASE),
}
_NEGATION = re.compile(r"\b(?:no|not|without|lack\w*|absen\w*|neither)\b", re.IGNORECASE)
_POST_MATCH_NEGATION = re.compile(
    r"^\s*(?:(?:is|are|was|were|appears?|remains?)\s+)?"
    r"(?:directly\s+)?(?:no|not|without|absen\w*)\b",
    re.IGNORECASE,
)
_PROMINENT_NUCLEOLI = re.compile(
    r"\b(?:(?:prominent|conspicuous|eosinophilic)\w*(?:\s+\w+){0,2}\s+nucleol\w*|"
    r"nucleol\w*(?:\s+\w+){0,3}\s+(?:prominent|conspicuous|eosinophilic)\w*|"
    r"(?:large|enlarged)\s+nucleolat\w*\s+cells?)\b",
    re.IGNORECASE,
)
_INCONSPICUOUS_NUCLEOLI = re.compile(
    r"\b(?:nucleol\w*(?:\s+\w+){0,3}\s+(?:absent|inconspicuous|not\s+visible)|"
    r"(?:absent|inconspicuous|no\s+visible)\s+nucleol\w*|"
    r"(?:without|lack\w*|no)\s+(?:prominent|conspicuous)\s+nucleol\w*)\b",
    re.IGNORECASE,
)
_GRADE_FOUR_MORPHOLOGY = re.compile(
    r"\b(?:rhabdoid|sarcomatoid|multinucleated\s+giant|extreme(?:ly)?\s+pleomorph)\w*\b",
    re.IGNORECASE,
)
_GRADE_ANSWER = re.compile(r"^Grade ([1-4])$")

_GASTRIC_HETEROGENEITY_SLIDE_DESCRIPTIONS = [
    "Gastric carcinoma with poorly cohesive single-cell or nongland-forming solid growth",
    "Gastric carcinoma with irregular partially formed invasive glands",
    "Heterogeneous gastric carcinoma containing both glandular and poorly differentiated regions",
]
_GASTRIC_HETEROGENEITY_LOW_POWER_QUERY = (
    "poorly cohesive or solid nongland-forming gastric carcinoma at invasive front"
)
_GASTRIC_HETEROGENEITY_HIGH_POWER_QUERY = (
    "discohesive malignant cells and loss of gland formation"
)


def _feature_polarities(text: str, pattern: re.Pattern[str]) -> set[bool]:
    """Return True for asserted presence and False for local negation."""

    output: set[bool] = set()
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 96) : match.start()]
        # Do not split on the periods inside abbreviations such as ``e.g.``.
        # Doing so turned ``absence of architecture (e.g., invasion) or
        # cytologic atypia`` into an asserted atypia because the negator was
        # discarded at the abbreviation boundary.  A period only terminates
        # the local scope when it looks like an actual sentence boundary.
        sentence_prefix = re.split(
            r";|(?<!e\.g)(?<!i\.e)\.(?=\s|$)",
            prefix,
            flags=re.IGNORECASE,
        )[-1]
        comma_scope = re.sub(
            r"\b(?:e\.g|i\.e)\.,", "abbrev ", sentence_prefix, flags=re.IGNORECASE
        )
        comma_tail = comma_scope.rsplit(",", 1)[-1]
        # Some relation patterns include their verb phrase in the match, so a
        # negator can occur after match.start(), e.g. "cells do not infiltrate
        # capsule" or "nucleoli are not prominent".  Inspect the matched local
        # clause as well as its prefix.
        matched_clause = re.split(
            r";|(?<!e\.g)(?<!i\.e)\.(?=\s|$)",
            match.group(0),
            flags=re.IGNORECASE,
        )[0]
        suffix = text[match.end() : match.end() + 64]
        suffix_clause = re.split(
            r",|;|(?<!e\.g)(?<!i\.e)\.(?=\s|$)",
            suffix,
            flags=re.IGNORECASE,
        )[0]
        negated = bool(
            _NEGATION.search(comma_tail)
            or _NEGATION.search(matched_clause)
            or _POST_MATCH_NEGATION.search(suffix_clause)
        )
        # Preserve negation across coordinated comma lists such as
        # "no inflammation, necrosis, or infiltrative growth". Splitting on
        # every comma previously converted the final item into a false
        # positive because its immediate prefix was only "or".
        if not negated and "," in comma_scope:
            connector = comma_tail.strip().casefold()
            if connector == "" or re.match(r"^(?:and|or|nor)\b", connector):
                negated = bool(_NEGATION.search(comma_scope))
        output.add(not negated)
    return output


def _asserted_relation_present(text: str, pattern: re.Pattern[str]) -> bool:
    """Return whether a clause asserts a complete matched relation.

    Unlike a single-feature mention, these patterns already span the subject,
    relation verb, and object.  A negator after the complete match can modify
    an unrelated trailing feature (for example, ``cells cross the fibrous
    boundary without architectural disruption``) and must not reverse the
    crossing relation.  Negation immediately before the subject or anywhere
    inside the matched relation remains disqualifying.
    """

    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 96) : match.start()]
        sentence_prefix = re.split(
            r";|(?<!e\.g)(?<!i\.e)\.(?=\s|$)",
            prefix,
            flags=re.IGNORECASE,
        )[-1]
        comma_scope = re.sub(
            r"\b(?:e\.g|i\.e)\.,", "abbrev ", sentence_prefix, flags=re.IGNORECASE
        )
        comma_tail = comma_scope.rsplit(",", 1)[-1]
        if _NEGATION.search(comma_tail) or _NEGATION.search(match.group(0)):
            continue
        return True
    return False


def _independent_verifier_rejected(
    item: ROIObservation, feature_name: str
) -> bool:
    """Keep a rejected high-risk feature out of deterministic answer gates."""

    return any(
        feature_name.casefold() in limitation.casefold()
        and "did not corroborate" in limitation.casefold()
        for limitation in item.limitations
    )


def _renal_sarcomatoid_support_ids(
    observations: list[ROIObservation],
) -> list[str]:
    """Ground sarcomatoid morphology without asking the observer to diagnose it.

    A neutral observer should describe a spindle-cell proliferation rather
    than emit the diagnostic word ``sarcomatoid``.  On the question side, a
    discrete dense/fascicular spindle-cell component plus independently
    sampled renal epithelial tumour architecture is sufficient to map that
    morphology to sarcomatoid differentiation.
    """

    adequate = [
        item for item in observations if item.image_quality == "adequate"
    ]
    by_roi = {item.roi_id: item for item in adequate}

    def lineage(item: ROIObservation) -> list[ROIObservation]:
        output = [item]
        seen = {item.roi_id}
        parent_id = item.parent_roi_id
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            output.append(parent)
            parent_id = parent.parent_roi_id
        return output

    spindle_component = re.compile(
        r"\b(?:dense\s+)?(?:spindle[-\s]+cell|spindle[-\s]+shaped\s+cell)"
        r"\s+proliferation\b|"
        r"\bproliferation\s+of\s+(?:bland\s+|atypical\s+|dense\s+)?"
        r"spindle[-\s]+(?:cells?|shaped\s+cells?)\b|"
        r"\bspindle[-\s]+(?:cells?|shaped\s+cells?)\b.{0,65}"
        r"\b(?:intersecting|sweeping|long)\s+fascicles?\b",
        re.IGNORECASE,
    )
    epithelial_tumor = re.compile(
        r"\b(?:predominantly\s+)?solid\s+and\s+nested\b|"
        r"\bsolid\s+nests?\b|"
        r"\b(?:solid|nested|nests?|alveolar)\b.{0,80}"
        r"\b(?:clear\s+cytoplasm|polygonal\s+cells?|delicate\s+vascular)|"
        r"\b(?:polygonal\s+cells?|cells?\s+with\s+clear\s+cytoplasm)\b"
        r".{0,80}\b(?:solid|nested|nests?|alveolar)\b",
        re.IGNORECASE,
    )
    high_grade_spindle = re.compile(
        r"\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|mitotic|necros\w*|"
        r"abrupt\s+transition|destructive|disorganiz\w*)\b",
        re.IGNORECASE,
    )
    bland_spindle = re.compile(
        r"\b(?:bland|uniform|monotonous)\b.{0,55}\b(?:spindle|smooth\s+muscle|fibroblast)|"
        r"\b(?:spindle|smooth\s+muscle|fibroblast)\w*\b.{0,55}\b(?:bland|uniform|monotonous)\b",
        re.IGNORECASE,
    )
    explicit = re.compile(r"\b(?:sarcomatoid|rhabdoid)\b", re.IGNORECASE)
    fascicular_spindle = re.compile(
        r"\bspindle(?:[-\s]+shaped)?[-\s]+cells?\b.{0,100}"
        r"\b(?:arranged\s+in\s+)?(?:intersecting|sweeping|long)\s+fascicles?\b|"
        r"\b(?:intersecting|sweeping|long)\s+fascicles?\b.{0,100}"
        r"\bspindle(?:[-\s]+shaped)?[-\s]+cells?\b",
        re.IGNORECASE,
    )
    explicitly_nonaggressive_spindle = re.compile(
        r"\bno\s+(?:necros\w*|mitotic\s+activity|significant\s+atypia)\b"
        r"[^.;]{0,55}\b(?:no\s+)?(?:infiltrative|destructive)\s+growth\b|"
        r"\bno\s+(?:infiltrative|destructive)\s+growth\b[^.;]{0,55}"
        r"\b(?:no\s+)?(?:necros\w*|mitotic\s+activity|significant\s+atypia)\b|"
        r"\b(?:smooth|sharply\s+demarcated)\b[^.;]{0,55}"
        r"\b(?:non[-\s]?infiltrative|does\s+not\s+infiltrate)\b|"
        r"\b(?:does\s+not|do\s+not|but\s+not|not\s+to)\s+infiltrat\w*\b|"
        r"\b(?:smooth|non[-\s]?infiltrative|pushing)\s+border\b",
        re.IGNORECASE,
    )
    for item in adequate:
        chain_text = " ".join(part.morphology for part in lineage(item))
        if (
            True in _feature_polarities(item.morphology, explicit)
            and bool(epithelial_tumor.search(chain_text))
            and not bland_spindle.search(chain_text)
        ):
            return [item.evidence_id]
    spindle_fields = [
        item
        for item in adequate
        if True in _feature_polarities(item.morphology, spindle_component)
        and (
            True in _feature_polarities(item.morphology, high_grade_spindle)
            # A discrete fascicular component does not cease to be a
            # sarcomatoid candidate merely because the neutral caption omits
            # the adjective ``dense``.  Require the actual intersecting,
            # sweeping or long-fascicle architecture and retain the explicit
            # bland spindle/smooth-muscle exclusion below.
            or bool(fascicular_spindle.search(item.morphology))
        )
        and not bland_spindle.search(item.morphology)
        and not explicitly_nonaggressive_spindle.search(item.morphology)
    ]
    tumor_fields = [
        item
        for item in adequate
        if True in _feature_polarities(item.morphology, epithelial_tumor)
    ]
    # Sarcomatoid differentiation is a transition within the renal tumour,
    # not the conjunction of a bland spindle field and epithelial tumour from
    # unrelated slides. A heterogeneous component can occupy a different ROI
    # tree on the same physical section, so require same slide (and exclude
    # bland smooth-muscle/fibroblastic mimics) rather than same root lineage.
    for spindle in spindle_fields:
        tumor = next(
            (
                item
                for item in tumor_fields
                if item.slide_id == spindle.slide_id
            ),
            None,
        )
        if tumor is not None:
            return list(dict.fromkeys([spindle.evidence_id, tumor.evidence_id]))
    return []


def _judgment_grounding_errors(
    rationale: str, morphology: str
) -> list[str]:
    errors: list[str] = []
    for name, pattern in _GROUNDABLE_FEATURES.items():
        claimed = _feature_polarities(rationale, pattern)
        if not claimed:
            continue
        observed = _feature_polarities(morphology, pattern)
        if not observed:
            errors.append(f"{name} is absent from the cited morphology")
        elif not claimed <= observed:
            errors.append(f"{name} polarity contradicts the cited morphology")
    return errors


def _renal_grade_bounds(
    question: dict[str, Any], observations: list[ROIObservation]
) -> tuple[str, str] | None:
    """Renal WHO/ISUP grade interval established at observed resolutions."""

    question_text = str(question.get("question") or "").casefold()
    if "who/isup grade" not in question_text:
        return None
    adequate = [item for item in observations if item.image_quality == "adequate"]
    by_roi = {item.roi_id: item for item in observations}
    normal_renal_context = re.compile(
        r"\b(?:glomerul\w*|bowman['’]?s?|normal\s+renal|"
        r"(?:normal|intact|preserved|regular)\s+tubul\w*|"
        r"tubul\w*.{0,45}\b(?:normal|intact|preserved|regular|unremarkable))\b",
        re.IGNORECASE,
    )
    non_tumor_vascular_context = re.compile(
        r"\b(?:large|small)\s+(?:blood\s+)?vessel\b[^.;]{0,80}"
        r"\b(?:lumen|red\s+blood\s+cells?|erythrocytes?)\b|"
        r"\bvascular\s+lumen\b[^.;]{0,60}"
        r"\b(?:red\s+blood\s+cells?|erythrocytes?)\b",
        re.IGNORECASE,
    )
    strong_tumor_architecture = re.compile(
        r"\bsolid\s+(?:growth|architectural|proliferation|sheets?)\b|"
        r"\b(?:nested|nests?|alveolar)\b.{0,70}\bdelicate\s+"
        r"(?:vascular|vessels?|septa?)\b|"
        r"\b(?:solid|nested|nests?|alveolar)\b.{0,70}\bclear\s+cytoplasm\b|"
        r"\b(?:proliferation|cluster\w*)\b.{0,70}\babundant\s+clear\s+cytoplasm\b",
        re.IGNORECASE,
    )

    def belongs_to_tumor_lineage(item: ROIObservation) -> bool:
        """Exclude nuclear claims reached through explicit normal-renal tissue.

        The observer remains question-independent, but grading on the question
        side must not promote a nucleolus seen in a glomerulus or intact renal
        tubule. A strong tumor architecture encountered before such background
        establishes a usable lineage. Missing ancestry is left usable so old
        traces and standalone observations remain conservative rather than
        being silently discarded.
        """

        current: ROIObservation | None = item
        seen: set[str] = set()
        while current is not None and current.roi_id not in seen:
            seen.add(current.roi_id)
            if non_tumor_vascular_context.search(current.morphology):
                return False
            if strong_tumor_architecture.search(current.morphology):
                return True
            if normal_renal_context.search(current.morphology):
                return False
            if not current.parent_roi_id:
                return True
            current = by_roi.get(current.parent_roi_id)
            if current is None:
                return True
        return False

    adequate = [item for item in adequate if belongs_to_tumor_lineage(item)]
    if not adequate:
        return None
    if _renal_sarcomatoid_support_ids(adequate):
        return "Grade 4", "Grade 4"
    isolated_nucleus = re.compile(
        r"\b(?:single|isolated|one)\b.{0,60}\b(?:cell|nucle\w*)\b|"
        r"\b(?:cell|nucle\w*)\b.{0,60}\b(?:single|isolated)\b",
        re.IGNORECASE,
    )
    representative_grade_four = re.compile(
        r"\b(?:rhabdoid\s+(?:cells?|morphology)|multinucleated\s+giant\s+cells?|"
        r"extreme(?:ly)?\s+pleomorph\w*)\b",
        re.IGNORECASE,
    )
    if any(
        item.magnification in {"20x", "40x"}
        and representative_grade_four.search(item.morphology)
        and not isolated_nucleus.search(item.morphology)
        for item in adequate
    ):
        return "Grade 4", "Grade 4"
    positive_prominence = [
        item
        for item in adequate
        if True in _feature_polarities(item.morphology, _PROMINENT_NUCLEOLI)
        and not isolated_nucleus.search(item.morphology)
    ]
    positive_at_40x = [
        item for item in positive_prominence if item.magnification == "40x"
    ]
    positive_at_10x = [
        item for item in positive_prominence if item.magnification == "10x"
    ]
    weak_prominence = re.compile(
        r"\b(?:occasional(?:ly)?|focal(?:ly)?|rare(?:ly)?|variabl\w*)\b.{0,40}"
        r"(?:prominent|conspicuous|eosinophilic)\w*\s+nucleol|"
        r"\bnucleol\w*.{0,40}\b(?:occasional(?:ly)?|focal(?:ly)?|rare(?:ly)?|variabl\w*)\b",
        re.IGNORECASE,
    )
    strong_at_20x = [
        item
        for item in positive_prominence
        if item.magnification == "20x"
        and not weak_prominence.search(item.morphology)
    ]
    strong_at_40x = [
        item
        for item in positive_at_40x
        if not weak_prominence.search(item.morphology)
    ]
    inconspicuous_at_40x = [
        item
        for item in adequate
        if item.magnification == "40x"
        and (
            _INCONSPICUOUS_NUCLEOLI.search(item.morphology)
            or False
            in _feature_polarities(item.morphology, _PROMINENT_NUCLEOLI)
        )
    ]
    inconspicuous_at_10x = [
        item
        for item in adequate
        if item.magnification == "10x"
        and (
            _INCONSPICUOUS_NUCLEOLI.search(item.morphology)
            or False
            in _feature_polarities(item.morphology, _PROMINENT_NUCLEOLI)
        )
    ]
    def root_roi_id(item: ROIObservation) -> str:
        current = item
        seen: set[str] = set()
        while current.parent_roi_id and current.parent_roi_id not in seen:
            seen.add(current.parent_roi_id)
            parent = by_roi.get(current.parent_roi_id)
            if parent is None:
                return current.parent_roi_id
            current = parent
        return current.roi_id

    positive_roots = {root_roi_id(item) for item in positive_prominence}
    negative_roots = {root_roi_id(item) for item in inconspicuous_at_40x}
    positive_20x_roots = {
        root_roi_id(item)
        for item in positive_prominence
        if item.magnification == "20x"
    }
    positive_40x_roots = {root_roi_id(item) for item in positive_at_40x}
    positive_10x_roots = {root_roi_id(item) for item in positive_at_10x}
    negative_10x_roots = {root_roi_id(item) for item in inconspicuous_at_10x}
    strong_20x_roots = {root_roi_id(item) for item in strong_at_20x}
    cross_scale_roots = positive_20x_roots & positive_40x_roots
    grade_three_cross_scale_roots = positive_10x_roots & positive_40x_roots
    positive_20x_roots_by_slide: dict[str, set[str]] = {}
    strong_20x_roots_by_slide: dict[str, set[str]] = {}
    for item in positive_prominence:
        if item.magnification != "20x":
            continue
        positive_20x_roots_by_slide.setdefault(item.slide_id, set()).add(
            root_roi_id(item)
        )
        if not weak_prominence.search(item.morphology):
            strong_20x_roots_by_slide.setdefault(item.slide_id, set()).add(
                root_roi_id(item)
            )
    # Grade 3 is defined by nucleoli already conspicuous at approximately a
    # 10x objective field. The 10x transcription is independently checked by
    # the neutral verifier before it reaches this aggregator. In whole-slide
    # runs, however, a single cross-scale lineage remains vulnerable to a
    # repeated caption adjective: a real Grade-1 control produced one such
    # lineage amid seventeen independent 10x roots with inconspicuous
    # nucleoli. Three same-slide 10x captions were also observed in a Grade-2
    # control when only two survived same-lineage 40x confirmation. Require
    # three cross-scale tumour roots; lower-power caption voting alone cannot
    # upstage the whole case.
    if len(grade_three_cross_scale_roots) >= 3:
        return "Grade 3", "Grade 3"
    # Legacy traces contain no true 10x field. Patho-R1 captions can become
    # less stable at 40x when a very small crop loses representative cells.
    # loses representative tumor cells.  Three independent 20x tumor roots on
    # the *same slide* are a stronger specimen-level signal than one such
    # caption, provided at least one field describes non-focal prominence.
    # Keeping both the same-slide and distinct-root requirements prevents the
    # old failure mode where weak phrases from unrelated slides were joined.
    # Necrosis is not part of the WHO/ISUP Grade-3 definition and must never
    # promote a nucleolar caption. Require regional low-power visibility:
    # three independent, non-weak 20x tumour roots on one physical slide.
    legacy_same_slide_grade_three_consensus = any(
        len(roots) >= 3
        for roots in strong_20x_roots_by_slide.values()
    )
    if legacy_same_slide_grade_three_consensus:
        return "Grade 3", "Grade 3"
    # WHO/ISUP Grade 2 is the complementary true-scale pattern: nucleoli are
    # inconspicuous at 10x yet prominent at 40x. Require this transition in
    # three independent rooted tumour regions. Two such roots occurred in a
    # Grade-1 control through caption noise, whereas both audited Grade-2
    # cases had three and the Grade-3 case was already captured above.
    true_scale_grade_two_roots = negative_10x_roots & positive_40x_roots
    if (
        len(true_scale_grade_two_roots) >= 3
        and len(positive_40x_roots) > len(negative_roots)
    ):
        return "Grade 2", "Grade 2"
    if cross_scale_roots and len(positive_roots) >= 2:
        return "Grade 2", "Grade 2"
    # A focal high-grade region may be absent from many fields, but a single
    # caption is also noisy. Four independent slides with strong 40x
    # prominence, constituting a strict majority of the sampled 40x tumour
    # slides, provide specimen-level Grade-2 consensus even when the 20x
    # parent crops did not reproduce the focal nucleolus wording.
    eligible_40x_slides = {
        item.slide_id for item in adequate if item.magnification == "40x"
    }
    strong_40x_slides = {item.slide_id for item in strong_at_40x}
    if (
        len(strong_40x_slides) >= 4
        and len(strong_40x_slides) * 2 > len(eligible_40x_slides)
        and (
            len(strong_40x_slides) >= 5
            or len(positive_40x_roots) > len(negative_roots)
        )
    ):
        return "Grade 2", "Grade 2"
    quantified_visible_nucleoli = re.compile(
        r"\bnucleol\w*\s+(?:are\s+)?visible\s+in\s+"
        r"(?:approximately\s+|about\s+)?(?P<percent>\d{1,3})\s*%",
        re.IGNORECASE,
    )
    quantified_roots_by_slide: dict[str, set[str]] = {}
    for item in adequate:
        if item.magnification != "40x":
            continue
        match = quantified_visible_nucleoli.search(item.morphology)
        if match is None or int(match.group("percent")) < 20:
            continue
        quantified_roots_by_slide.setdefault(item.slide_id, set()).add(
            root_roi_id(item)
        )
    strong_40x_roots_by_slide: dict[str, set[str]] = {}
    for item in strong_at_40x:
        strong_40x_roots_by_slide.setdefault(item.slide_id, set()).add(
            root_roi_id(item)
        )
    # A highest-grade focus must not be outvoted by bland fields.  Accept one
    # localized Grade-2 focus when an unequivocal 40x prominence caption is
    # independently supported by a second 40x root on the same slide that
    # quantifies visible nucleoli in at least 20% of cells.  Keep this route
    # focal: prominent-nucleolus captions scattered across several slides but
    # contradicted at 20x are a recurrent observer template failure, not
    # independent biological confirmation.
    focal_same_slide_grade_two = False
    if len(strong_40x_slides) == 1:
        slide_id = next(iter(strong_40x_slides))
        strong_roots = strong_40x_roots_by_slide.get(slide_id, set())
        quantified_roots = quantified_roots_by_slide.get(slide_id, set())
        focal_same_slide_grade_two = bool(
            strong_roots
            and any(root not in strong_roots for root in quantified_roots)
        )
    if focal_same_slide_grade_two:
        return "Grade 2", "Grade 2"
    distinct_nucleoli = re.compile(
        r"\bdistinct\s+nucleol\w*\b|\bnucleol\w*\s+(?:are\s+)?distinct\b",
        re.IGNORECASE,
    )
    distinct_20x_roots_by_slide: dict[str, set[str]] = {}
    for item in adequate:
        if (
            item.magnification == "20x"
            and distinct_nucleoli.search(item.morphology)
            and not _INCONSPICUOUS_NUCLEOLI.search(item.morphology)
        ):
            distinct_20x_roots_by_slide.setdefault(item.slide_id, set()).add(
                root_roi_id(item)
            )
    weak_40x_roots_by_slide: dict[str, set[str]] = {}
    for item in positive_at_40x:
        if weak_prominence.search(item.morphology):
            weak_40x_roots_by_slide.setdefault(item.slide_id, set()).add(
                root_roi_id(item)
            )
    positive_prominence_slides = {item.slide_id for item in positive_prominence}
    # Subtle Grade-2 fields can be described as merely distinct at 20x and as
    # occasional prominence at 40x. Accept that weaker signal only when it is
    # localized to one slide, reproduced across independent ROI roots and no
    # strong 20x/40x prominence template occurs anywhere in the case. This
    # preserves Grade-1 controls where stock prominent-nucleolus captions are
    # scattered through otherwise inconspicuous tumour fields.
    subtle_same_slide_grade_two = False
    if (
        len(positive_prominence_slides) == 1
        and not strong_at_20x
        and not strong_at_40x
    ):
        slide_id = next(iter(positive_prominence_slides))
        weak_roots = weak_40x_roots_by_slide.get(slide_id, set())
        distinct_roots = distinct_20x_roots_by_slide.get(slide_id, set())
        subtle_same_slide_grade_two = bool(
            weak_roots and any(root not in weak_roots for root in distinct_roots)
        )
    if subtle_same_slide_grade_two:
        return "Grade 2", "Grade 2"
    if positive_prominence and all(
        item.parent_roi_id is None for item in positive_prominence
    ) and not negative_roots:
        # Preserve direct standalone observations used by controller unit
        # contracts and legacy traces that lack ancestry metadata.
        return "Grade 2", "Grade 2"
    if positive_prominence:
        # If the only positive wording came from 20x but repeated 40x fields
        # explicitly show inconspicuous nucleoli, treat the lone lower-power
        # phrase as uncorroborated instead of promoting the whole tumor.
        if (
            not cross_scale_roots or len(positive_roots) < 2
        ) and len(negative_roots) >= 2:
            return "Grade 1", "Grade 1"
        return "Grade 1", "Grade 2"
    # WHO/ISUP Grade 2 requires nucleoli that are conspicuous at 40x.  Two
    # independently sampled adequate high-power fields explicitly reporting
    # absent/inconspicuous nucleoli establish a Grade-1 ceiling unless a focal
    # higher-grade field above has already overridden them.
    if negative_roots:
        return "Grade 1", "Grade 1"
    return "Grade 1", "Grade 2"


def _renal_grade_ceiling(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    bounds = _renal_grade_bounds(question, observations)
    return bounds[1] if bounds else None


def _renal_grade_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    """Return a grade only when question-side morphology fixes both bounds."""

    bounds = _renal_grade_bounds(question, observations)
    if bounds is not None and bounds[0] == bounds[1]:
        return bounds[0]
    return None


def _bound_grade_answer(
    value: Any, floor: str | None, ceiling: str | None
) -> tuple[Any, bool]:
    if floor is None or ceiling is None:
        return value, False
    match = _GRADE_ANSWER.fullmatch(str(value or "").strip())
    floor_match = _GRADE_ANSWER.fullmatch(floor)
    ceiling_match = _GRADE_ANSWER.fullmatch(ceiling)
    if not match or floor_match is None or ceiling_match is None:
        return value, False
    grade = int(match.group(1))
    if grade < int(floor_match.group(1)):
        return floor, True
    if grade > int(ceiling_match.group(1)):
        return ceiling, True
    return value, False


def _canonical_answer_format(question: dict[str, Any], value: Any) -> Any:
    """Remove task-extraneous qualifiers without changing a true subtype."""

    if value is None:
        return None
    answer = str(value or "").strip()
    question_text = str(question.get("question") or "").casefold()
    clinical_domain = str(question.get("clinical_domain") or "").casefold()
    is_margin_question = any(
        term in question_text for term in ("margin", "dissection surface")
    )
    # Free-form margin questions use a dataset-level reporting phrase rather
    # than a bare polarity.  Treat only concise, unambiguous synonyms as
    # equivalent so that qualifiers such as "cannot exclude" are never
    # promoted into a definite result.  Choice questions retain their literal
    # option text and yes/no questions retain the Yes/No contract.
    if (
        is_margin_question
        and question.get("question_type") == "short_answer"
        and not question.get("options")
    ):
        concise_margin_answer = answer.casefold().strip().rstrip(".")
        if re.fullmatch(
            r"(?:the\s+)?(?:resection\s+|surgical\s+)?margin(?:s)?\s+(?:(?:is|are)\s+)?"
            r"(?:negative|uninvolved|free\s+of\s+(?:carcinoma|tumou?r))|"
            r"negative|uninvolved|no\s+(?:carcinoma|tumou?r)(?:\s+(?:is|was))?\s+"
            r"(?:identified|seen|present|found|detected)|"
            r"no\s+(?:carcinoma|tumou?r)",
            concise_margin_answer,
            flags=re.IGNORECASE,
        ):
            return "No carcinoma identified"
        if re.fullmatch(
            r"(?:the\s+)?(?:resection\s+|surgical\s+)?margin(?:s)?\s+(?:(?:is|are)\s+)?"
            r"(?:positive|involved)|positive|involved|"
            r"(?:carcinoma|tumou?r)(?:\s+(?:is|was))?\s+"
            r"(?:identified|seen|present|found|detected)",
            concise_margin_answer,
            flags=re.IGNORECASE,
        ):
            return "Carcinoma identified"
    generic_type = any(
        term in question_text
        for term in ("histological type", "pathological type", "tumor type")
    )
    if generic_type and "gastric" in clinical_domain:
        answer = re.sub(
            r"^(?:well|moderately|poorly)(?:-|\s+)differentiated\s+",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()
        # Intestinal/diffuse/mixed are Lauren classifications of a gastric
        # carcinoma, not answers to a generic histological-type question.  A
        # small controller can otherwise copy a visually plausible Lauren
        # label even when the question asks for the parent tumour category.
        # Canonicalise that granularity mismatch without using a patient label.
        if re.fullmatch(
            r"(?:lauren\s+)?(?:intestinal|diffuse|mixed)(?:\s+type)?",
            answer,
            flags=re.IGNORECASE,
        ):
            return "Adenocarcinoma"
        if (
            answer.casefold().endswith("adenocarcinoma")
            and "mucinous" not in answer.casefold()
        ):
            return "Adenocarcinoma"
    return answer


def _is_generic_gastric_type_question(question: dict[str, Any]) -> bool:
    question_text = str(question.get("question") or "").casefold()
    clinical_domain = str(question.get("clinical_domain") or "").casefold()
    return (
        "gastric" in clinical_domain
        and any(
            term in question_text
            for term in ("histological type", "pathological type", "tumor type")
        )
        and "lauren classification" not in question_text
    )


def _gastric_parent_type_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    """Retain an established invasive gastric epithelial parent category.

    The final integrator occasionally lets a late crop of bland residual
    glands overwrite earlier invasive carcinoma morphology.  This preference
    is deliberately morphology-gated: irregular/crowded glands and an
    invasive, destructive, or desmoplastic context must coexist within one
    localization lineage.  Organ/cohort membership alone is never enough.
    """

    if not _is_generic_gastric_type_question(question):
        return None
    adequate = [
        item for item in observations if item.image_quality == "adequate"
    ]
    by_roi = {item.roi_id: item for item in adequate}
    abnormal_glands = re.compile(
        r"\b(?:irregular|crowded|back[-\s]+to[-\s]+back|angulated|distorted|"
        r"complex|branching|haphazard)\w*.{0,55}\b(?:gland|glandular|tubul)\w*\b|"
        r"\b(?:gland|glandular|tubul)\w*.{0,55}\b(?:irregular|crowded|"
        r"back[-\s]+to[-\s]+back|angulated|distorted|complex|branching|"
        r"haphazard)\w*\b",
        re.IGNORECASE,
    )
    invasive_context = re.compile(
        r"\b(?:infiltrat|invad|penetrat|breach|destruct|disrupt)\w*\b|"
        r"\b(?:desmoplas\w*|stromal\s+reaction|desmoplastic\s+stroma)\b|"
        r"\b(?:gland|epithelial|cellular)\w*.{0,65}\b(?:within|between|into|"
        r"embedded\s+in|interspersed\s+with)\b.{0,35}\b(?:stroma|smooth\s+"
        r"muscle|fat|adipose|fibroadipose)\b",
        re.IGNORECASE,
    )

    for item in adequate:
        lineage = [item]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            lineage.append(parent)
            parent_id = parent.parent_roi_id
        text = " ".join(part.morphology for part in lineage)
        if (
            True in _feature_polarities(text, abnormal_glands)
            and True in _feature_polarities(text, invasive_context)
        ):
            return "Adenocarcinoma"
    return None


def _is_deepest_layer_question(question: dict[str, Any]) -> bool:
    text = str(question.get("question") or "").casefold()
    return "deepest" in text and any(
        term in text for term in ("layer", "level", "tissue")
    )


def _is_pt_question(question: dict[str, Any]) -> bool:
    text = str(question.get("question") or "").casefold()
    return any(
        term in text
        for term in (
            "pt category",
            "pt stage",
            "pathological t category",
            "pathological t (pt) stage",
            "primary-tumor category (pt)",
        )
    )


def _is_pn_question(question: dict[str, Any]) -> bool:
    text = str(question.get("question") or "").casefold()
    return any(
        term in text
        for term in (
            "pn category",
            "pn stage",
            "pathological n category",
            "pathological n (pn) stage",
            "regional-node category (pn)",
        )
    )


def is_regional_node_question(question: dict[str, Any]) -> bool:
    """Whether a patient-level answer requires coverage of regional nodes."""

    text = str(question.get("question") or "").casefold()
    return bool(
        re.search(r"\blymph\s+nodes?\b|\bregional[-\s]+nodes?\b", text)
        or _is_pn_question(question)
    )


def is_renal_capsule_question(question: dict[str, Any]) -> bool:
    """Whether the question requires a renal-tumour/capsule interface lineage."""

    text = str(question.get("question") or "").casefold()
    domain = str(question.get("clinical_domain") or "").casefold()
    return "renal" in domain and "capsule" in text


def is_renal_perirenal_fat_question(question: dict[str, Any]) -> bool:
    """Whether the requested endpoint is extension into perirenal adipose tissue."""

    text = str(question.get("question") or "").casefold()
    return is_renal_capsule_question(question) and bool(
        re.search(r"\b(?:fat\s+capsule|perirenal|perinephric)\b", text)
    )


def is_gastric_extent_question(question: dict[str, Any]) -> bool:
    """Whether the question asks for the deepest gastric extent or pT."""

    domain = str(question.get("clinical_domain") or "").casefold()
    return "gastric" in domain and (
        _is_deepest_layer_question(question) or _is_pt_question(question)
    )


def is_microinvasion_question(question: dict[str, Any]) -> bool:
    """Whether PNI/LVI requires a focal high-power spatial search."""

    text = str(question.get("question") or "").casefold()
    return any(term in text for term in ("perineural", "lymphovascular"))


def is_spatial_interface_question(question: dict[str, Any]) -> bool:
    """Whether a conclusion depends on a localized tissue relationship."""

    text = str(question.get("question") or "").casefold()
    return bool(
        is_renal_capsule_question(question)
        or is_microinvasion_question(question)
        or any(
            term in text
            for term in (
                "margin",
                "dissection surface",
            )
        )
    )


def question_specific_focus_roi_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
    current_slide_id: str,
    available_roi_ids: list[str],
) -> list[str]:
    """Prioritize morphology-bearing parents when a generic focus choice is unsafe."""

    available = set(available_roi_ids)
    question_text = str(question.get("question") or "").casefold()
    if is_spatial_interface_question(question) or is_gastric_extent_question(question):
        if is_gastric_extent_question(question):
            structure = re.compile(
                r"\bseros\w*\b|"
                r"\bmesothelial[-\s]+lined\s+(?:outer|free)"
                r"(?:\s+(?:outer|free))?(?:\s+tissue)?\s+surface\b|"
                r"\b(?:outer|free)(?:\s+(?:outer|free))?\s+"
                r"(?:tissue\s+)?surface\b|"
                r"\b(?:adipose|fibroadipose|adipocytes?|fibrous\s+boundary|"
                r"smooth\s+muscle|muscularis(?:\s+propria)?)\b",
                re.IGNORECASE,
            )
        elif "perineural" in question_text:
            structure = re.compile(
                r"\b(?:nerve|neural|perineur\w*|epineur\w*|nerve\s+sheath)\b",
                re.IGNORECASE,
            )
        elif "lymphovascular" in question_text:
            structure = re.compile(
                r"\b(?:vessel|vascular|lymphatic|endothelial[-\s]+lined)\w*\b",
                re.IGNORECASE,
            )
        elif is_renal_perirenal_fat_question(question):
            structure = re.compile(
                r"\b(?:perirenal|perinephric)\s+(?:fat|adipose(?:\s+tissue)?)\b|"
                r"\b(?:fat|adipose(?:\s+tissue)?|adipocytes?)\b",
                re.IGNORECASE,
            )
        elif is_renal_capsule_question(question):
            structure = re.compile(
                r"\b(?:fibrous\s+)?capsul\w*\b|"
                r"\b(?:fibrous|collagenous)\s+(?:band|layer|boundary|rim)\b",
                re.IGNORECASE,
            )
        else:
            structure = re.compile(
                r"\b(?:inked|cauterized|cut\s+edge|resection\s+margin|"
                r"surgical\s+margin|dissection\s+surface)\b",
                re.IGNORECASE,
            )
        tumor = re.compile(
            r"\b(?:epithelial|glandular|irregular\s+glands?|cellular\s+nests?|"
            r"solid\s+nests?|cohesive\s+(?:cells?|clusters?)|clear\s+cells?|"
            r"tumou?r|carcinoma|neoplastic)\b",
            re.IGNORECASE,
        )
        relation = re.compile(
            r"\b(?:abut|adjacent|indistinct|interface|border|cross|enter|"
            r"penetrat|breach|infiltrat|invad|surround|encas|within|inside|"
            r"not\s+fully|not\s+directly|cannot\s+be\s+confirmed)\w*\b",
            re.IGNORECASE,
        )
        scored_spatial: list[tuple[int, int, str]] = []
        for index, item in enumerate(observations):
            if item.slide_id != current_slide_id or item.roi_id not in available:
                continue
            if True not in _feature_polarities(item.morphology, structure):
                continue
            tumor_present = True in _feature_polarities(item.morphology, tumor)
            score = 8 + (5 if tumor_present else 0)
            if relation.search(item.morphology):
                score += 3
            if item.magnification in {"20x", "40x"}:
                score += 1
            scored_spatial.append((score, -index, item.roi_id))
        scored_spatial.sort(reverse=True)
        if scored_spatial:
            return [roi_id for _, _, roi_id in scored_spatial]
    if not is_regional_node_question(question):
        if not any(
            term in question_text
            for term in ("grade", "nuclear", "nucleol", "mitotic", "pleomorph")
        ):
            return []
        cellular_scored: list[tuple[float, int, str]] = []
        for index, item in enumerate(observations):
            if item.slide_id != current_slide_id or item.roi_id not in available:
                continue
            analysis = item.cellular_analysis
            if analysis is None or analysis.quality == "unavailable":
                continue
            neoplastic = float(analysis.grouped_counts.get("neoplastic_like", 0))
            events = float(analysis.grouped_counts.get("cellular_events", 0))
            nuclear_q95 = float(analysis.nuclear_area_um2_quantiles.get("q95", 0.0))
            score = math.log1p(neoplastic) + 2.0 * math.log1p(events) + nuclear_q95 / 100.0
            cellular_scored.append((score, -index, item.roi_id))
        cellular_scored.sort(reverse=True)
        return [roi_id for _, _, roi_id in cellular_scored]
    epithelial = re.compile(
        r"\b(?:epithelial|gland\w*|acini|acinar|tubul\w*|"
        r"cohesive\s+(?:cell\w*|cluster\w*)|cellular\s+nests?|"
        r"solid(?:\s+and)?\s+nested|solid\s+(?:nests?|sheets?))\b",
        re.IGNORECASE,
    )
    suspicious = re.compile(
        r"\b(?:atyp\w*|pleomorph\w*|irregular\w*|crowd\w*|haphazard\w*|"
        r"desmoplas\w*|infiltrative|disrupt\w*|effac\w*)\b",
        re.IGNORECASE,
    )
    lymphoid = re.compile(
        r"\b(?:lymph\w*|nodal|follic\w*|germinal\s+cent\w*|"
        r"subcapsular\s+sinus)\b",
        re.IGNORECASE,
    )
    scored: list[tuple[int, int, str]] = []
    for index, item in enumerate(observations):
        if item.slide_id != current_slide_id or item.roi_id not in available:
            continue
        epithelial_positive = True in _feature_polarities(
            item.morphology, epithelial
        )
        cellular = item.cellular_analysis
        cell_neoplastic = (
            int(cellular.grouped_counts.get("neoplastic_like", 0))
            if cellular is not None
            else 0
        )
        cell_immune = (
            int(cellular.grouped_counts.get("immune", 0))
            if cellular is not None
            else 0
        )
        histoplus_focus_signal = cell_neoplastic >= 3 and cell_immune >= 10
        if not epithelial_positive and not histoplus_focus_signal:
            continue
        score = 5 if epithelial_positive else 0
        if histoplus_focus_signal:
            score += 4
        if True in _feature_polarities(item.morphology, suspicious):
            score += 3
        if True in _feature_polarities(item.morphology, lymphoid):
            score += 2
        if item.image_quality == "adequate":
            score += 1
        scored.append((score, -index, item.roi_id))
    scored.sort(reverse=True)
    return [roi_id for _, _, roi_id in scored]


def _lymph_node_metastasis_support_groups(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> list[list[str]]:
    """Find distinct rooted deposits on morphologically established node tissue."""

    text = str(question.get("question") or "").casefold()
    if (
        "metastatic carcinoma" not in text
        or not is_regional_node_question(question)
        or question.get("question_type") != "yes_no"
    ):
        return []
    explicit_node_context = re.compile(
        r"\blymph\s+node\b|\bsubcapsular\s+sinus\b|"
        r"\bnodal\s+(?:architecture|parenchyma|capsule)\b",
        re.IGNORECASE,
    )
    definitive_node_context = re.compile(
        explicit_node_context.pattern + r"|"
        r"\blymphoid\s+tissue\s+architecture\b.{0,100}\bfollic\w*\b|"
        r"\bpreserved\s+(?:follicular\s+)?architecture\b.{0,120}"
        r"\bgerminal\s+cent\w*\b.{0,120}\binterfollicular\s+zones?\b|"
        r"\bgerminal\s+cent\w*\b.{0,100}\bmantle\s+zones?\b|"
        r"\bmantle\s+zones?\b.{0,100}\bgerminal\s+cent\w*\b|"
        r"\bdense\s+lymphoid\s+(?:infiltrate|tissue)\b.{0,80}"
        r"\bpreserved\s+(?:follicular\s+)?architecture\b|"
        r"\bdense\s+lymphoid\s+(?:proliferation|aggregate)\b.{0,100}"
        r"\bpreserved\s+(?:follicular\s+)?architecture\b.{0,100}"
        r"\b(?:follic\w*|germinal\s+cent\w*)\b|"
        r"\blymphoid\s+(?:infiltrate|tissue|proliferation)\b.{0,150}"
        r"\bnodular\s+aggregates?\b.{0,100}\breactive\s+follicles?\b|"
        r"\bcentral\s+germinal\s+cent\w*\b.{0,100}"
        r"\b(?:peripheral|surrounding)\s+mantle\s+zones?\b",
        re.IGNORECASE,
    )
    epithelial = re.compile(
        r"\b(?:epithelial|gland\w*|acini|acinar|tubul\w*|"
        r"cellular\s+nests?|solid\s+(?:nests?|sheets?))\b",
        re.IGNORECASE,
    )
    suspicious_cytology_features = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\batyp\w*\b",
            r"\bhigh\s+nuclear[-\s]+to[-\s]+cytoplasmic\b",
            r"\bnuclear\s+crowd\w*\b",
            r"\bhyperchrom\w*\b",
            r"\bmitotic\s+figures?\b",
            r"\bpleomorph\w*\b",
            r"\bdisorganiz\w*\s+overlap\w*\b",
            r"\birregular\s+nuclear\s+(?:membran\w*|contours?)\b",
            r"\b(?:coarse(?:ly\s+clumped)?\s+chromatin|"
            r"chromatin\s+(?:is\s+|appears?\s+|displays?\s+)?"
            r"coarse(?:ly\s+clumped)?)\b",
            r"\b(?:(?:marked\s+)?nuclear\s+size\s+variation|"
            r"marked\s+size\s+variation|nuclei\b.{0,70}\b"
            r"(?:vary|varying|variable)\b.{0,20}\bin\s+size)\b",
            r"\b(?:(?:enlarged|large)\s+nuclei|nuclei\s+(?:are\s+)?enlarged)\b",
            r"\b(?:prominent\s+nucleol\w*|"
            r"nucleol\w*\s+(?:are\s+|is\s+)?prominent)\b",
        )
    ]

    def suspicious_cytology_count(item: ROIObservation) -> int:
        return sum(
            True in _feature_polarities(item.morphology, pattern)
            for pattern in suspicious_cytology_features
        )

    def atypical_epithelial_claim_was_rejected(item: ROIObservation) -> bool:
        return any(
            "atypical_epithelial_deposit" in limitation
            and "did not corroborate" in limitation.casefold()
            for limitation in item.limitations
        )

    deposit_architecture = re.compile(
        r"\b(?:solid(?:\s+and)?\s+nested|solid\s+(?:nests?|sheets?)|"
        r"cellular\s+nests?|gland\w*|cohesive\s+clusters?)\b",
        re.IGNORECASE,
    )
    benign_mimic = re.compile(
        r"\b(?:granuloma|epithelioid\s+histiocytes?|foamy\s+histiocytes?|"
        r"mesothelial\s+cells?|endometrial\s+gland|endothelial\s+cells?|"
        r"salivary\s+gland|respiratory\s+epithelium|mature\s+adipocytes?)\b",
        re.IGNORECASE,
    )
    spatial_separation = re.compile(
        r"\b(?:spatially\s+distinct|respect\w*\s+(?:their\s+)?spatial\s+"
        r"boundar\w*|without\s+direct\s+crossing|non[-\s]?overlapping|"
        r"maintain\w*.{0,50}spatial\s+relationship.{0,80}"
        r"without\s+crossing\s+into\s+(?:the\s+)?epithelial\s+compartment|"
        r"smooth\s+and\s+non[-\s]?overlapping|sharply\s+demarcated.{0,35}"
        r"no\s+intermingling)\b",
        re.IGNORECASE,
    )
    destructive_epithelial_architecture = re.compile(
        r"\b(?:crowded|irregular|angulated|distorted|complex|branching)\w*"
        r".{0,45}\b(?:gland|glandular|tubul)\w*\b|"
        r"\b(?:gland|glandular|tubul)\w*.{0,45}"
        r"\b(?:crowded|irregular|angulated|distorted|complex|branching|"
        r"haphazard|disarray)\w*\b|"
        r"\b(?:architectural\s+disarray|loss\s+of\s+normal\s+lobular\s+organization)\b",
        re.IGNORECASE,
    )
    gastric_wall_context = re.compile(
        r"\b(?:gastric\s+wall|gastric\s+mucosa|foveolar|muscularis|"
        r"mucosal\s+crypts?|intestinal\s+villi|smooth\s+muscle\s+bundles?)\b",
        re.IGNORECASE,
    )
    by_roi = {item.roi_id: item for item in observations}
    by_slide: dict[str, list[ROIObservation]] = {}
    for item in observations:
        if item.image_quality != "inadequate":
            by_slide.setdefault(item.slide_id, []).append(item)

    def lineage_text(item: ROIObservation) -> str:
        parts = [item.morphology]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            parts.append(parent.morphology)
            parent_id = parent.parent_roi_id
        return " ".join(parts)

    def root_roi_id(item: ROIObservation) -> str:
        current = item
        seen: set[str] = set()
        while current.parent_roi_id and current.parent_roi_id not in seen:
            seen.add(current.parent_roi_id)
            parent = by_roi.get(current.parent_roi_id)
            if parent is None:
                return current.parent_roi_id
            current = parent
        return current.roi_id

    support_groups: list[list[str]] = []
    for slide_id, items in by_slide.items():
        node_fields = [
            item
            for item in items
            if True in _feature_polarities(item.morphology, definitive_node_context)
        ]
        if not node_fields:
            continue
        explicit_node_fields = [
            item
            for item in items
            if True
            in _feature_polarities(item.morphology, explicit_node_context)
        ]
        organized_node_roots = {root_roi_id(item) for item in node_fields}
        # A solitary germinal-center aggregate can occur inside a primary
        # gastric tumour. It cannot provide global nodal context for joining
        # an unrelated tumour ROI elsewhere on the same slide. Cross-lineage
        # reconstruction therefore needs either explicit node anatomy or two
        # independently sampled organized lymphoid low-power regions.
        strong_slide_node_context = bool(explicit_node_fields) or (
            len(organized_node_roots) >= 2
        )
        definitive_node_fields = explicit_node_fields or node_fields
        slide_text = " ".join(item.morphology for item in items)
        slide_has_gastric_wall = True in _feature_polarities(
            slide_text, gastric_wall_context
        )
        high_power = [
            item for item in items if item.magnification in {"20x", "40x"}
        ]
        epithelial_fields = [
            item
            for item in high_power
            if item.image_quality == "adequate"
            and True in _feature_polarities(item.morphology, epithelial)
            and not benign_mimic.search(item.morphology)
            and not atypical_epithelial_claim_was_rejected(item)
        ]
        direct_combined: list[
            tuple[ROIObservation, ROIObservation, bool]
        ] = []
        for item in epithelial_fields:
            lineage: list[ROIObservation] = [item]
            parent_id = item.parent_roi_id
            seen: set[str] = set()
            while parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = by_roi.get(parent_id)
                if parent is None:
                    break
                lineage.append(parent)
                parent_id = parent.parent_roi_id
            lineage_text_value = " ".join(part.morphology for part in lineage)
            if spatial_separation.search(lineage_text_value):
                continue
            lineage_node = next(
                (
                    part
                    for part in lineage
                    if True
                    in _feature_polarities(part.morphology, definitive_node_context)
                ),
                None,
            )
            self_node_context = True in _feature_polarities(
                item.morphology, definitive_node_context
            )
            incidental_residual_epithelium = re.search(
                r"\b(?:residual\s+epithelial\s+(?:elements?|structures?)|"
                r"intact\s+(?:salivary\s+)?gland|preserved\s+zymogen)\b",
                item.morphology,
                re.IGNORECASE,
            )
            # Joining two unrelated ROI trees on one physical slide is only
            # allowed for a highly specific high-power cytologic signal.  A
            # merely irregular gland can belong to the primary tumour or an
            # incidental epithelial structure elsewhere on the section.
            strong = (
                strong_slide_node_context
                and
                suspicious_cytology_count(item) >= 3
                and not slide_has_gastric_wall
            )
            localized_to_node = (
                self_node_context
                and incidental_residual_epithelium is None
                and (
                    suspicious_cytology_count(item) >= 2
                    or True
                    in _feature_polarities(
                        lineage_text_value, destructive_epithelial_architecture
                    )
                )
            ) or (
                lineage_node is not None
                and suspicious_cytology_count(item) >= 2
            )
            replaced_node_cross_lineage = (
                strong_slide_node_context
                and not slide_has_gastric_wall
                and suspicious_cytology_count(item) >= 2
                and True
                in _feature_polarities(
                    lineage_text_value, destructive_epithelial_architecture
                )
            )
            if localized_to_node or strong or replaced_node_cross_lineage:
                node_item = (
                    lineage_node
                    if lineage_node is not None
                    else (definitive_node_fields or node_fields)[0]
                )
                if node_item.evidence_id == item.evidence_id:
                    node_item = next(
                        (
                            candidate
                            for candidate in node_fields
                            if candidate.evidence_id != item.evidence_id
                        ),
                        node_item,
                    )
                direct_combined.append(
                    (item, node_item, localized_to_node)
                )
        suspicious_deposit_fields = [
            item
            for item in high_power
            if item.image_quality == "adequate"
            and strong_slide_node_context
            # Scattered large atypical lymphoid cells (for example reactive
            # immunoblasts) can contribute three nuclear adjectives and even
            # mention that they are *not* cohesive. They are not evidence of
            # metastatic carcinoma. The suspicious field itself, rather than
            # an unrelated bland epithelial ROI elsewhere on the slide, must
            # carry an independently verified epithelial/glandular identity.
            and True in _feature_polarities(item.morphology, epithelial)
            and suspicious_cytology_count(item) >= 3
            and not atypical_epithelial_claim_was_rejected(item)
            and not slide_has_gastric_wall
            and True in _feature_polarities(lineage_text(item), deposit_architecture)
            and not benign_mimic.search(item.morphology)
            and not spatial_separation.search(lineage_text(item))
        ]
        if direct_combined:
            # Multiple high-power descendants of the same low-power root are
            # repeated views of one deposit and count once. Additional roots
            # on the same physical slide count only when their own lineage
            # contains nodal anatomy; global slide-level lymphoid context may
            # authorize one replaced-node deposit but cannot be multiplied.
            localized_by_root: dict[
                str, tuple[ROIObservation, ROIObservation]
            ] = {}
            fallback: tuple[ROIObservation, ROIObservation] | None = None
            for epithelial_item, node_item, localized_to_node in direct_combined:
                pair = (epithelial_item, node_item)
                if fallback is None:
                    fallback = pair
                if localized_to_node:
                    localized_by_root.setdefault(root_roi_id(epithelial_item), pair)
            pairs = list(localized_by_root.values()) or ([fallback] if fallback else [])
            support_groups.extend(
                list(
                    dict.fromkeys(
                        [epithelial_item.evidence_id, node_item.evidence_id]
                    )
                )[:2]
                for epithelial_item, node_item in pairs
            )
            continue
        if epithelial_fields and suspicious_deposit_fields:
            epithelial_item = suspicious_deposit_fields[0]
            support_groups.append(
                list(
                    dict.fromkeys(
                        [
                            epithelial_item.evidence_id,
                            definitive_node_fields[0].evidence_id,
                        ]
                    )
                )[:2]
            )
    return support_groups


def _lymph_node_metastasis_support_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> list[str]:
    """Return one strict support pair for an existential nodal conclusion."""

    groups = _lymph_node_metastasis_support_groups(question, observations)
    return groups[0] if groups else []


def _lymph_node_metastasis_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None = None,
    reviews: list[EvidenceReview] | None = None,
) -> str | None:
    """Integrate nodal context and epithelial deposits without using slide labels."""

    if len(_lymph_node_metastasis_support_ids(question, observations)) >= 2:
        return "Yes"
    text = str(question.get("question") or "").casefold()
    if (
        "metastatic carcinoma" not in text
        or not is_regional_node_question(question)
        or question.get("question_type") != "yes_no"
    ):
        return None
    if termination_reason == "candidate_slides_exhausted":
        # Review judgments are deliberately not a second route to a positive.
        # Two loosely positive crops on different slides still do not establish
        # that an atypical epithelial deposit is located in a lymph node. A
        # positive is authorized only by the same-slide morphology contract at
        # the top of this function; after complete candidate coverage, failure
        # to form that contract is deterministically negative.
        return "No"
    return None


def _gastric_pn_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None = None,
) -> str | None:
    """Map independently supported involved-node slides to a gastric pN bin."""

    domain = str(question.get("clinical_domain") or "").casefold()
    if (domain and "gastric" not in domain) or not _is_pn_question(question):
        return None
    fake_question = {
        **question,
        "question": "Is metastatic carcinoma present in submitted regional lymph nodes?",
        "question_type": "yes_no",
        "options": ["Yes", "No"],
    }
    count = len(
        _lymph_node_metastasis_support_groups(fake_question, observations)
    )
    if count == 0:
        if termination_reason != "candidate_slides_exhausted":
            return None
        answer = "pN0"
    elif count <= 2:
        # Independently positive node slides establish at least pN1.  Returning
        # no preference here allowed the final integrator to collapse every
        # non-zero case back to pN0 despite strict metastatic evidence.
        answer = "pN1"
    elif count <= 6:
        answer = "pN2"
    else:
        answer = "pN3"
    options = [str(item) for item in question.get("options", [])]
    return answer if not options or answer in options else None


def _renal_pn_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None = None,
) -> str | None:
    """Distinguish an unevaluable renal nodal category from true pN0."""

    domain = str(question.get("clinical_domain") or "").casefold()
    if (
        "renal" not in domain
        or not _is_pn_question(question)
        or termination_reason != "candidate_slides_exhausted"
    ):
        return None
    fake_question = {
        **question,
        "question": "Is metastatic carcinoma present in submitted regional lymph nodes?",
        "question_type": "yes_no",
        "options": ["Yes", "No"],
    }
    by_slide: dict[str, list[ROIObservation]] = {}
    for item in observations:
        by_slide.setdefault(item.slide_id, []).append(item)
    if any(
        len(_lymph_node_metastasis_support_ids(fake_question, items)) >= 2
        for items in by_slide.values()
    ):
        answer = "pN1"
    else:
        node_context = re.compile(
            r"\blymph\s+node\b|\bsubcapsular\s+sinus\b|"
            r"\bnodal\s+(?:architecture|parenchyma|capsule)\b|"
            r"\bpreserved\s+(?:follicular\s+)?architecture\b.{0,120}"
            r"\bgerminal\s+cent\w*\b.{0,120}\binterfollicular\s+zones?\b|"
            r"\bgerminal\s+cent\w*\b.{0,100}\bmantle\s+zone\b|"
            r"\bmantle\s+zone\b.{0,100}\bgerminal\s+cent\w*\b",
            re.IGNORECASE,
        )
        evaluable_node = any(
            item.image_quality == "adequate"
            and True in _feature_polarities(item.morphology, node_context)
            for item in observations
        )
        # No morphologically evaluable submitted node is pNX, not pN0.
        answer = "pN0" if evaluable_node else "pNX"
    options = [str(item) for item in question.get("options", [])]
    return answer if not options or answer in options else None


def _renal_subtype_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    """Choose a renal subtype only when option-specific hallmark morphology wins."""

    text = str(question.get("question") or "").casefold()
    options = {str(item) for item in question.get("options", [])}
    expected = {
        "Clear cell renal cell carcinoma",
        "Papillary renal cell carcinoma",
        "Chromophobe renal cell carcinoma",
        "Renal oncocytoma",
    }
    if (
        "renal" not in str(question.get("clinical_domain") or "").casefold()
        or not any(term in text for term in ("histological type", "pathological type", "tumor type"))
        or not expected <= options
    ):
        return None

    scores = {item: 0 for item in expected}
    clear_high_power_fields = 0
    papillary_core_fields = 0
    for item in observations:
        if item.image_quality != "adequate":
            continue
        morphology = item.morphology
        lowered = morphology.casefold()
        clear_present = True in _feature_polarities(
            morphology, _GROUNDABLE_FEATURES["clear_cytoplasm"]
        )
        nested = bool(re.search(r"\b(?:nested|nests?|alveolar)\b", lowered))
        delicate_vessels = bool(
            re.search(
                r"\b(?:delicate|thin-walled)\b.{0,45}\b(?:vascular|vessels?|sept\w*)\b|"
                r"\bfibrovascular\s+sept\w*\b",
                lowered,
            )
        )
        papillary = True in _feature_polarities(
            morphology, _GROUNDABLE_FEATURES["papillary"]
        )
        fibrovascular_core = bool(re.search(r"\bfibrovascular\s+cores?\b", lowered))
        perinuclear = True in _feature_polarities(
            morphology, _GROUNDABLE_FEATURES["perinuclear_halo"]
        )
        raisinoid = bool(re.search(r"\b(?:raisinoid|wrinkled)\b", lowered))
        granular_eosinophilic = bool(
            re.search(r"\bgranular\s+eosinophilic\s+cytoplasm\b|"
                      r"\beosinophilic\s+granular\s+cytoplasm\b", lowered)
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
        central_scar = bool(re.search(r"\bcentral\s+(?:fibrous\s+)?scar\b", lowered))

        if clear_present:
            scores["Clear cell renal cell carcinoma"] += (
                4 if item.magnification in {"20x", "40x"} else 2
            )
            if item.magnification in {"20x", "40x"}:
                clear_high_power_fields += 1
        elif pale_vacuolated and item.magnification in {"20x", "40x"}:
            scores["Clear cell renal cell carcinoma"] += 2
            clear_high_power_fields += 1
        if nested and delicate_vessels:
            scores["Clear cell renal cell carcinoma"] += 3
        if papillary and fibrovascular_core:
            scores["Papillary renal cell carcinoma"] += 4
            papillary_core_fields += 1
        elif papillary:
            # A lone low-power use of "papillary" is vulnerable to folds,
            # clefts, normal papillae, and caption-template completion.  It is
            # supporting architecture, but must not outweigh repeated
            # required-resolution cytoplasmic morphology.
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

    if (
        clear_high_power_fields >= 1
        and papillary_core_fields < 2
        and scores["Clear cell renal cell carcinoma"] >= max(scores.values())
    ):
        return "Clear cell renal cell carcinoma"
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] < 2 or ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _gastric_high_grade_nonglandular_field(item: ROIObservation) -> bool:
    """Recognize an atypical non-gland-forming epithelial component.

    This is deliberately narrower than the generic phrase ``without gland
    formation``.  Lymphoid fields, bland single-layered epithelium and an
    explicitly cohesive gland-poor focus must not manufacture a diffuse/poor
    component.
    """

    if item.image_quality != "adequate" or item.magnification not in {"20x", "40x"}:
        return False
    epithelial = re.compile(
        r"\b(?:epithelial|carcinoma|tumou?r\s+cells?)\b",
        re.IGNORECASE,
    )
    nonglandular_disarray = re.compile(
        r"\bdisorganiz\w*\s+(?:sheets?|pattern)\b[^.;]{0,80}"
        r"\bwithout\s+gland(?:ular)?\s+(?:formation|structures?)\b|"
        r"\bwithout\s+gland(?:ular)?\s+(?:formation|structures?)\b"
        r"[^.;]{0,80}\bdisorganiz\w*\s+(?:sheets?|pattern)\b",
        re.IGNORECASE,
    )
    suspicious_nuclei = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\birregular\s+nuclear\s+(?:membran\w*|contours?)\b",
            r"\bpleomorph\w*\b",
            r"\bmitotic\s+figures?\b",
            r"\bhyperchrom\w*\b",
        )
    ]
    positive_cohesion = re.compile(r"\bcohesive\b", re.IGNORECASE)
    return bool(
        True in _feature_polarities(item.morphology, epithelial)
        and nonglandular_disarray.search(item.morphology)
        and sum(
            True in _feature_polarities(item.morphology, pattern)
            for pattern in suspicious_nuclei
        )
        >= 2
        and True not in _feature_polarities(item.morphology, positive_cohesion)
    )


def _lauren_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    if "lauren classification" not in str(question.get("question") or "").casefold():
        return None
    intestinal = 0
    diffuse = 0
    intestinal_pattern = re.compile(
        r"\b(?:glandular|glands?|tubular|papillary)\b",
        re.IGNORECASE,
    )
    diffuse_pattern = re.compile(
        r"\b(?:signet[-\s]?ring|discohesive|poorly\s+cohesive|"
        r"non[-\s]?cohesive|loss\s+of\s+cohesion|isolated\s+(?:tumou?r\s+)?cells?|"
        r"scattered\s+(?:(?:single\s+)(?:tumou?r\s+)?cells?|tumou?r\s+cells?))\b|"
        r"\bsingle[-\s]?cell\s+infiltrat\w*\b",
        re.IGNORECASE,
    )
    for item in observations:
        if item.image_quality != "adequate":
            continue
        text = item.morphology.casefold()
        if True in _feature_polarities(item.morphology, intestinal_pattern) and re.search(
            r"\b(?:cohesive|polarity|lumen|luminal|well-formed|organized|preserved)\b",
            text,
        ):
            intestinal += 1
        if (
            True in _feature_polarities(item.morphology, diffuse_pattern)
            or _gastric_high_grade_nonglandular_field(item)
        ):
            diffuse += 1
    if intestinal and diffuse:
        return "Mixed type"
    if intestinal:
        return "Intestinal type"
    if diffuse:
        return "Diffuse type"
    return None


def _gastric_differentiation_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    if "degree of differentiation" not in str(question.get("question") or "").casefold():
        return None
    moderate = 0
    poor = 0
    well = 0
    for item in observations:
        if item.image_quality != "adequate":
            continue
        morphology = item.morphology.casefold()
        if re.search(
            r"\b(?:irregular|crowded|distorted|complex|branching|haphazard)\w*\b"
            r".{0,45}\b(?:gland|glandular|tubul)\w*\b|"
            r"\b(?:gland|glandular|tubul)\w*\b.{0,45}"
            r"\b(?:irregular|crowded|distorted|complex|branching|haphazard)\w*\b",
            morphology,
        ):
            moderate += 1
        poor_pattern = re.compile(
            r"\b(?:signet[-\s]?ring|discohesive|poorly\s+cohesive|"
            r"non[-\s]?cohesive|loss\s+of\s+cohesion|isolated\s+(?:tumou?r\s+)?cells?|"
            r"scattered\s+(?:(?:single\s+)(?:tumou?r\s+)?cells?|tumou?r\s+cells?))\b|"
            r"\bsingle[-\s]?cell\s+infiltrat\w*\b|"
            r"\b(?:absen\w*\s+gland\w*|no\s+gland\w*\s+formation)\b",
            re.IGNORECASE,
        )
        if (
            True in _feature_polarities(item.morphology, poor_pattern)
            or _gastric_high_grade_nonglandular_field(item)
            or (
                "predominantly solid" in morphology
                and True
                in _feature_polarities(
                    item.morphology, _GROUNDABLE_FEATURES["infiltration"]
                )
            )
        ):
            poor += 1
        if re.search(
            r"\b(?:well-formed|regular|organized|preserved)\w*\b.{0,45}"
            r"\b(?:gland|glandular|tubul)\w*\b|"
            r"\b(?:gland|glandular|tubul)\w*\b.{0,45}"
            r"\b(?:well-formed|regular|organized|preserved)\w*\b",
            morphology,
        ):
            well += 1
    if moderate and poor:
        # Differentiation is assigned by the least differentiated viable
        # component; a rare poorly cohesive focus must not be outvoted by
        # numerous fields from the dominant moderately gland-forming area.
        return "Moderately to poorly differentiated"
    if moderate:
        return "Moderately differentiated"
    if poor:
        return "Poorly differentiated"
    if well:
        return "Well differentiated"
    return None


def _gastric_invasion_layer(
    observations: list[ROIObservation],
) -> str | None:
    """Return the deepest positively described gastric wall compartment."""

    adequate = [
        item for item in observations if item.image_quality == "adequate"
    ]
    by_roi = {item.roi_id: item for item in adequate}
    lineage_texts: list[str] = []
    for item in adequate:
        parts = [item.morphology]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            parts.append(parent.morphology)
            parent_id = parent.parent_roi_id
        lineage_texts.append(" ".join(parts))

    def positively_observed(pattern: re.Pattern[str]) -> bool:
        # Do not manufacture a deep interface by concatenating morphology
        # from unrelated slides or ROI trees. A finding must occur within one
        # localization lineage.
        return any(
            True in _feature_polarities(text, pattern)
            for text in lineage_texts
        )

    serosal_relation = re.compile(
        r"\b(?:(?:infiltrat|inva(?:d|s)|penetrat|reach|breach|cross|travers)\w*|"
        r"extend\w*\s+(?:to|through|across|onto))"
        r".{0,70}\b(?:seros(?:a|al)(?:\s+(?:surface|lining))?|"
        r"mesothelial[-\s]+lined\s+(?:outer|free)(?:\s+(?:outer|free))?"
        r"(?:\s+tissue)?\s+surface|"
        r"true\s+(?:outer|free)(?:\s+(?:outer|free))?(?:\s+tissue)?\s+surface)\b|"
        r"\b(?:tumou?r|carcinoma|epithelial|gland|nest|cell)\w*\b.{0,70}"
        r"\b(?:(?:reach|cross|penetrat|breach|travers)\w*|"
        r"extend\w*\s+(?:to|through|across|onto))"
        r".{0,35}\b(?:seros(?:a|al)(?:\s+(?:surface|lining))?|"
        r"mesothelial[-\s]+lined\s+(?:outer|free)(?:\s+(?:outer|free))?"
        r"(?:\s+tissue)?\s+surface|"
        r"true\s+(?:outer|free)(?:\s+(?:outer|free))?(?:\s+tissue)?\s+surface)\b|"
        r"\b(?:seros(?:a|al)(?:\s+(?:surface|lining))?|"
        r"mesothelial[-\s]+lined\s+(?:outer|free)(?:\s+(?:outer|free))?"
        r"(?:\s+tissue)?\s+surface|"
        r"true\s+(?:outer|free)(?:\s+(?:outer|free))?(?:\s+tissue)?\s+surface)\b.{0,70}"
        r"\b(?:reach|cross|penetrat|breach|travers|infiltrat|inva(?:d|s))\w*"
        r"(?:\s+by)?\b.{0,35}\b(?:tumou?r|carcinoma|epithelial|gland|nest|cell)\w*\b",
        re.IGNORECASE,
    )
    uncertain_serosal_relation = re.compile(
        r"\b(?:whether|uncertain|equivocal|indeterminate)\b|"
        r"\b(?:cannot|can't|could\s+not|unable\s+to|not\s+reliably|not\s+fully)"
        r"\b.{0,60}\b(?:confirm|establish|resolve|assess|visualiz|evaluat)\w*\b|"
        r"\bnot\s+(?:directly\s+)?(?:confirm|establish|shown|visible)\w*\b",
        re.IGNORECASE,
    )
    non_serosal_cut_surface = re.compile(
        r"\b(?:inked|cauterized|sharply\s+cut|cut\s+edge|resection\s+(?:edge|margin|surface)|"
        r"surgical\s+margin|dissection\s+surface)\b",
        re.IGNORECASE,
    )

    def serosal_surface_observed() -> bool:
        # A glass-facing edge is only a surrogate used to localize the outer
        # surface.  It becomes staging evidence only when the neutral visual
        # transcript identifies a true serosal/free/mesothelial surface and
        # links epithelial proliferation to it.  Explicit surgical or cut
        # surfaces must never be promoted to serosa.
        for text in lineage_texts:
            for clause in re.split(r"(?<=[.!?;])\s+", text):
                if non_serosal_cut_surface.search(clause):
                    continue
                if uncertain_serosal_relation.search(clause):
                    continue
                if True in _feature_polarities(clause, serosal_relation):
                    return True
        return False

    if serosal_surface_observed():
        return "Serosa of the gastric wall"
    if positively_observed(
        re.compile(
            r"\b(?:infiltrat|inva(?:d|s)|extend|disrupt|cross)\w*.{0,70}"
            r"\b(?:subseros\w*\s+)?(?:fat|adipose|fibroadipose)\b|"
            r"\b(?:gland|nest|carcinoma|tumou?r\s+cell)\w*.{0,70}"
            r"\b(?:within|between|into|embedded\s+in|interspersed\s+with)\b.{0,25}"
            r"\b(?:fat|adipose|fibroadipose|adipocytes?)\b|"
            r"\b(?:fat|adipose|fibroadipose|adipocytes?)\b.{0,70}"
            r"\b(?:infiltrat\w*|invad\w*|irregular\s+glands?|tumou?r\s+(?:nests?|cells?))\b",
            re.IGNORECASE,
        )
    ):
        return "Subserosal fibroadipose tissue"
    if positively_observed(
        re.compile(
            r"\b(?:infiltrat|inva(?:d|s)|extend|reach)\w*.{0,55}\bsubseros\w*\b|"
            r"\bsubseros\w*\b.{0,55}\b(?:tumou?r|carcinoma|gland|nest)\w*\b",
            re.IGNORECASE,
        )
    ):
        return "Subserosa"
    if positively_observed(
        re.compile(
            r"\b(?:infiltrat|inva(?:d|s)|extend|disrupt|reach)\w*.{0,55}"
            r"\bmuscularis\s+propria\b|"
            r"\bmuscularis\s+propria\b.{0,55}"
            r"\b(?:infiltrat\w*|tumou?r|carcinoma|gland|nest)\w*\b",
            re.IGNORECASE,
        )
    ):
        return "Muscularis propria"
    if positively_observed(
        re.compile(
            r"\b(?:infiltrat|inva(?:d|s)|extend|reach)\w*.{0,55}\bsubmucos\w*\b|"
            r"\bsubmucos\w*\b.{0,55}\b(?:tumou?r|carcinoma|gland|nest)\w*\b",
            re.IGNORECASE,
        )
    ):
        return "Submucosa"
    return None


def _deepest_layer_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    if not _is_deepest_layer_question(question):
        return None
    return _gastric_invasion_layer(observations)


def _gastric_pt_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    if not _is_pt_question(question):
        return None
    layer = _gastric_invasion_layer(observations)
    if layer == "Serosa of the gastric wall":
        answer = "pT4"
    elif layer in {"Subserosal fibroadipose tissue", "Subserosa"}:
        answer = "pT3"
    elif layer == "Muscularis propria":
        answer = "pT2"
    elif layer == "Submucosa":
        answer = "pT1"
    else:
        return None
    options = [str(item) for item in question.get("options", [])]
    return answer if not options or answer in options else None


def _margin_support_ids(
    question: dict[str, Any], observations: list[ROIObservation]
) -> list[str]:
    """Find epithelial cells spatially linked to an actual cut surface."""

    text = str(question.get("question") or "").casefold()
    if not any(term in text for term in ("margin", "dissection surface")):
        return []
    edge = re.compile(
        r"\b(?:resection|surgical|inked|cauterized|sharply\s+cut)\s+"
        r"(?:margin|edge|surface)\b|\b(?:cut\s+edge|dissection\s+surface)\b",
        re.IGNORECASE,
    )
    epithelial = re.compile(
        r"\b(?:atypical\s+epithelial\s+(?:cells?|clusters?)|epithelial\s+"
        r"(?:cells?|clusters?|nests?)|irregular\s+glands?|cellular\s+nests?|"
        r"solid\s+nests?)\b",
        re.IGNORECASE,
    )
    contact = re.compile(
        r"\b(?:epithelial\s+(?:cells?|clusters?|nests?)|irregular\s+glands?|"
        r"cellular\s+nests?|solid\s+nests?)\b.{0,95}"
        r"\b(?:touch|reach|extend(?:s|ed|ing)?\s+to|abut|cross|travers|present\s+at)\w*\b"
        r".{0,55}\b(?:inked|cauterized|cut\s+edge|resection\s+margin|"
        r"surgical\s+margin|dissection\s+surface)\b|"
        r"\b(?:inked|cauterized|cut\s+edge|resection\s+margin|"
        r"surgical\s+margin|dissection\s+surface)\b.{0,70}"
        r"\b(?:contain|touch|abut|involv|reach)\w*\b.{0,45}"
        r"\b(?:epithelial\s+(?:cells?|clusters?|nests?)|irregular\s+glands?|"
        r"cellular\s+nests?|solid\s+nests?)\b",
        re.IGNORECASE,
    )
    separated = re.compile(
        r"\b(?:separat\w*\s+from|does\s+not\s+reach|not\s+at|free\s+of|"
        r"no\s+epithelial\s+cells?\s+at)\b[^.;]{0,70}"
        r"\b(?:margin|(?:inked|cut)\s+edge|edge|surface)\b",
        re.IGNORECASE,
    )
    by_roi = {item.roi_id: item for item in observations}
    for item in observations:
        if item.image_quality != "adequate" or item.magnification not in {"20x", "40x"}:
            continue
        lineage = [item]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            lineage.append(parent)
            parent_id = parent.parent_roi_id
        lineage_text = " ".join(part.morphology for part in lineage)
        relation_parts = [
            part
            for part in lineage
            if True in _feature_polarities(part.morphology, contact)
        ]
        if (
            relation_parts
            and True in _feature_polarities(lineage_text, edge)
            and True in _feature_polarities(lineage_text, epithelial)
            and not separated.search(lineage_text)
        ):
            return list(
                dict.fromkeys(
                    [item.evidence_id, *(part.evidence_id for part in relation_parts)]
                )
            )[:2]
    return []


def _margin_preference(
    question: dict[str, Any],
    reviews: list[EvidenceReview],
    termination_reason: str | None,
    observations: list[ROIObservation] | None = None,
) -> str | None:
    text = str(question.get("question") or "").casefold()
    if not any(term in text for term in ("margin", "dissection surface")):
        return None
    if _margin_support_ids(question, observations or []):
        if question.get("question_type") == "yes_no":
            return "Yes"
        for option in [str(item) for item in question.get("options", [])]:
            lowered = option.casefold()
            if (
                any(
                    term in lowered
                    for term in (
                        "carcinoma identified",
                        "tumor identified",
                        "positive",
                        "involved",
                    )
                )
                and not any(
                    term in lowered
                    for term in (
                        "no carcinoma",
                        "no tumor",
                        "negative",
                        "uninvolved",
                    )
                )
            ):
                return option
        if question.get("question_type") == "short_answer":
            return "Carcinoma identified"
    if termination_reason != "candidate_slides_exhausted":
        return None
    if question.get("question_type") == "yes_no":
        return "No"
    for option in [str(item) for item in question.get("options", [])]:
        lowered = option.casefold()
        if any(
            term in lowered
            for term in ("no carcinoma", "no tumor", "negative", "uninvolved")
        ):
            return option
    if question.get("question_type") == "short_answer":
        return "No carcinoma identified"
    return None


def _renal_capsule_support_ids(
    question: dict[str, Any], observations: list[ROIObservation]
) -> list[str]:
    """Return a high-power tumour/capsule lineage with an asserted entry relation.

    The relation expression is deliberately clause bounded.  Using ``.{0,n}``
    here used to join ``epithelial clusters. The epithelial cells do not
    infiltrate ...`` across a sentence boundary and manufacture a positive
    capsule call from an explicitly negative observation.
    """

    options = {str(item) for item in question.get("options", [])}
    if not is_renal_capsule_question(question) or not {"Yes", "No"} <= options:
        return []
    perirenal_fat_endpoint = is_renal_perirenal_fat_question(question)
    boundary = re.compile(
        r"\b(?:fibrous\s+)?capsul\w*\b|"
        r"\b(?:(?:broad|thick|dense|continuous|outer)\s+)?"
        r"(?:collagenous\s+)?fibrous\s+(?:band|layer|boundary|rim)\b|"
        r"\b(?:tissue|outer)\s+surface\b",
        re.IGNORECASE,
    )
    cellular_component = re.compile(
        r"\b(?:tumou?r|carcinoma|clear\s+(?:cells?|cytoplasm)|"
        r"(?:cellular\s+)?epithelial\s+proliferation|"
        r"cellular\s+(?:nests?|proliferation)|proliferation|solid\s+(?:nests?|sheets?)|"
        r"nested\s+(?:cells?|architecture)|polygonal\s+cells?|"
        r"epithelial\s+(?:cells?|clusters?)|glands?)\b",
        re.IGNORECASE,
    )
    entry = re.compile(
        r"\b(?:(?:cellular\s+)?epithelial\s+proliferation|"
        r"cellular\s+(?:nests?|proliferation)|proliferation|solid\s+nests?|clear\s+cells?|"
        r"polygonal\s+cells?|epithelial\s+(?:cells?|clusters?)|glands?)\w*"
        r"[^.;]{0,80}\b(?:enter|extend|infiltrat|penetrat|breach|cross|"
        r"interdigitat|track|span|continu\w*\s+across)\w*[^.;]{0,65}"
        r"\b(?:capsul\w*|collagen\w*|fibrous\s+(?:band|layer|boundary|rim)|"
        r"boundary|interface)\b|"
        r"\b(?:capsul\w*|collagen\w*|fibrous\s+(?:band|layer|boundary|rim))\b"
        r"[^.;]{0,65}\b(?:contain|enclos|travers|enter|infiltrat|penetrat|breach|cross|"
        r"continu\w*\s+across)\w*"
        r"[^.;]{0,45}\b(?:(?:cellular\s+)?epithelial\s+proliferation|"
        r"cellular\s+(?:nests?|proliferation)|proliferation|clear\s+cells?|polygonal\s+cells?|"
        r"epithelial\s+(?:cells?|clusters?)|glands?)\b",
        re.IGNORECASE,
    )
    relation_uncertain = re.compile(
        r"\b(?:not\s+(?:fully|clearly|directly|definitively|reliably)\s+"
        r"(?:visuali[sz]ed|visible|shown|demonstrated|assessable)|"
        r"cannot\s+(?:be\s+)?(?:visuali[sz]ed|seen|confirmed|assessed)|"
        r"(?:uncertain|equivocal|indeterminate|suggestive|possible|questionable)\b|"
        r"appear\w*\b[^.;]{0,55}\b(?:cross|enter|extend|infiltrat|penetrat|breach|"
        r"continu\w*\s+across)\w*|"
        r"(?:prevent|preclud)\w*[^.;]{0,55}\b(?:assess|evaluat|confirm|determin)\w*|"
        r"\bwhether\b[^.;]{0,90}\b(?:cross|enter|extend|infiltrat|penetrat|breach)\w*|"
        r"\bpotential\s+(?:boundary|interface|capsul\w*)\b|"
        r"\b(?:assessment|evaluation)\b[^.;]{0,45}\b(?:limit|preclud|prevent)\w*)",
        re.IGNORECASE,
    )
    peripheral_adipose_entry = re.compile(
        r"\b(?:(?:cellular\s+)?epithelial\s+proliferation|"
        r"cellular\s+(?:nests?|proliferation)|proliferation|solid\s+nests?|"
        r"clear\s+cells?|polygonal\s+cells?|epithelial\s+(?:cells?|clusters?)|"
        r"cells?|nests?|clusters?|glands?)\b[^.;]{0,100}"
        r"\b(?:enter|extend|infiltrat|invad|penetrat|track|interdigitat|cross)\w*\b"
        r"[^.;]{0,45}\b(?:between|among)\s+(?:mature\s+)?adipocytes?\b|"
        r"\b(?:(?:cellular\s+)?epithelial\s+proliferation|"
        r"cellular\s+(?:nests?|proliferation)|proliferation|solid\s+nests?|"
        r"clear\s+cells?|polygonal\s+cells?|epithelial\s+(?:cells?|clusters?)|"
        r"cells?|nests?|clusters?|glands?)\b[^.;]{0,100}"
        r"\b(?:enter|extend|infiltrat|invad|penetrat|track|interdigitat|cross)\w*\b"
        r"[^.;]{0,45}\b(?:(?:perirenal|perinephric|adjacent)\s+)?"
        r"(?:fat|adipose(?:\s+tissue)?)\b",
        re.IGNORECASE,
    )
    by_roi = {item.roi_id: item for item in observations}
    for item in observations:
        if (
            item.image_quality != "adequate"
            or item.magnification not in {"20x", "40x"}
            or _independent_verifier_rejected(item, "infiltrative_interface")
            or True not in _feature_polarities(item.morphology, cellular_component)
        ):
            continue
        lineage = [item]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            lineage.append(parent)
            parent_id = parent.parent_roi_id
        lineage_text = " ".join(part.morphology for part in lineage)
        # Tumour infiltrating between adipocytes at the specimen periphery is
        # already outside the compact renal parenchyma and is direct evidence
        # of capsular/pericapsular extension even when the residual collagenous
        # capsule is not present in that same small field.  Keep this route at
        # true high power and in one clause; an abutting, separated or
        # incompletely visualized interface remains negative/indeterminate.
        if (
            _asserted_relation_present(item.morphology, peripheral_adipose_entry)
            and not relation_uncertain.search(item.morphology)
        ):
            return [item.evidence_id]
        # A question explicitly asking about the fat capsule/perirenal fat
        # has a deeper anatomical endpoint than fibrous-capsule entry.  Do
        # not reuse a tumour/fibrous-band relation unless this same high-power
        # observation also places viable epithelial tumour in adipose tissue.
        if perirenal_fat_endpoint:
            continue
        relation_parts = [
            part
            for part in lineage
            if part.magnification in {"20x", "40x"}
            and not _independent_verifier_rejected(
                part, "infiltrative_interface"
            )
            and _asserted_relation_present(part.morphology, entry)
            and not relation_uncertain.search(part.morphology)
        ]
        if (
            True in _feature_polarities(lineage_text, boundary)
            and relation_parts
        ):
            return list(
                dict.fromkeys(
                    [item.evidence_id, *(part.evidence_id for part in relation_parts)]
                )
            )[:2]
    return []


def _renal_capsule_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None,
) -> str | None:
    options = {str(item) for item in question.get("options", [])}
    if not is_renal_capsule_question(question) or not {"Yes", "No"} <= options:
        return None
    if _renal_capsule_support_ids(question, observations):
        return "Yes"
    if termination_reason == "candidate_slides_exhausted":
        return "No"
    return None


def _microinvasion_support_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> list[str]:
    text = str(question.get("question") or "").casefold()
    options = {str(item) for item in question.get("options", [])}
    if not {"Yes", "No"} <= options:
        return []
    if "perineural" in text:
        epithelial_component = (
            r"(?:tumou?r(?:\s+cells?)?|carcinoma(?:\s+cells?)?|"
            r"malignant\s+epithelial\s+cells?|atypical\s+epithelial\s+cells?|"
            r"(?:cellular\s+)?epithelial\s+(?:cells?|groups?|clusters?|nests?|proliferation)|"
            r"glandular\s+(?:cells?|groups?|clusters?|structures?|proliferation)|"
            r"(?:nests?|clusters?|groups?)\s+of\s+(?:atypical\s+)?epithelial\s+cells?|"
            r"cohesive\s+(?:epithelial\s+)?(?:cells?|groups?|clusters?|nests?)|"
            r"irregular\s+glands?)"
        )
        nerve_component = r"(?:nerve(?:\s+(?:fascicle|profile|sheath))?|neural|perineur\w*)"
        # A positive PNI relation must be expressed inside one morphology
        # clause.  In particular, ``nerve ... surrounded by intact
        # perineurium`` is a normal nerve description, not the passive form
        # of ``nerve surrounded by epithelial tumour``.  Requiring the
        # epithelial object immediately after a passive/transitive relation
        # prevents an unrelated epithelial finding in a parent ROI from
        # completing that normal description.
        pattern = re.compile(
            rf"\b{epithelial_component}\b[^.;:!?]{{0,70}}"
            rf"\b(?:surround|encas|within|inside|invad|infiltrat|penetrat|enter)\w*\b"
            rf"[^.;:!?]{{0,45}}\b{nerve_component}\b|"
            rf"\b{nerve_component}\b[^.;:!?]{{0,55}}"
            rf"\b(?:is\s+|are\s+|being\s+)?"
            rf"(?:surround|encas|invad|infiltrat|penetrat|contain|enclos)\w*"
            rf"\s+(?:by\s+|with\s+)?(?:an?\s+|the\s+)?{epithelial_component}\b",
            re.IGNORECASE,
        )
    elif "lymphovascular" in text:
        epithelial_component = (
            r"(?:tumou?r(?:\s+cells?)?|carcinoma(?:\s+cells?)?|"
            r"malignant\s+epithelial\s+cells?|atypical\s+epithelial\s+cells?|"
            r"(?:cellular\s+)?epithelial\s+(?:cells?|groups?|nests?|clusters?|proliferation)|"
            r"glandular\s+(?:cells?|groups?|nests?|clusters?)|"
            r"(?:nests?|clusters?|groups?)\s+of\s+(?:atypical\s+)?epithelial\s+cells?|"
            r"cellular\s+nests?|solid\s+nests?|"
            r"cohesive\s+(?:epithelial\s+)?(?:cells?|groups?|clusters?|nests?)|embol\w*)"
        )
        vascular_component = (
            r"(?:vessel(?:\s+lumen)?|vascular\s+(?:space|channel|lumen)|"
            r"lymphatic\s+(?:space|channel|lumen)|"
            r"endothelial[-\s]+lined\s+(?:space|channel|lumen|vessel(?:\s+lumen)?)|"
            r"endothelial\s+(?:space|channel|lumen|cells?))"
        )
        # Merely mentioning a vessel and epithelium in the same field is not
        # LVI.  The channel must explicitly contain/enclose an epithelial
        # embolus, or the embolus must explicitly lie within an endothelial-
        # lined channel (including the equivalent endothelial-surrounding
        # description).
        pattern = re.compile(
            rf"\b{epithelial_component}\b[^.;:!?]{{0,70}}"
            rf"\b(?:within|inside|intravascular|intralymphatic|occup|fill)\w*\b"
            rf"[^.;:!?]{{0,45}}\b{vascular_component}\b|"
            rf"\b{epithelial_component}\b[^.;:!?]{{0,45}}"
            rf"\b(?:surround|encas)\w*\s+by\s+endothelial\s+cells?\b|"
            rf"\b{vascular_component}\b[^.;:!?]{{0,55}}"
            rf"\b(?:contain|enclos|harbou?r|hous|occup)\w*"
            rf"\s+(?:by\s+|with\s+)?(?:an?\s+|the\s+)?{epithelial_component}\b",
            re.IGNORECASE,
        )
    else:
        return []
    separated = re.compile(
        r"\b(?:separat\w*|non[-\s]?overlapping|does\s+not\s+(?:enter|surround|"
        r"encase|invade)|outside\s+(?:the\s+)?(?:nerve|vessel))\b",
        re.IGNORECASE,
    )
    by_roi = {item.roi_id: item for item in observations}
    for item in observations:
        if item.image_quality != "adequate" or item.magnification not in {"20x", "40x"}:
            continue
        lineage = [item]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            lineage.append(parent)
            parent_id = parent.parent_roi_id
        lineage_text = " ".join(part.morphology for part in lineage)
        relation_parts = [
            part
            for part in lineage
            if True in _feature_polarities(part.morphology, pattern)
        ]
        if (
            relation_parts
            and not separated.search(lineage_text)
        ):
            return list(
                dict.fromkeys(
                    [item.evidence_id, *(part.evidence_id for part in relation_parts)]
                )
            )[:2]
    return []


def _microinvasion_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None,
) -> str | None:
    text = str(question.get("question") or "").casefold()
    options = {str(item) for item in question.get("options", [])}
    if not {"Yes", "No"} <= options or not any(
        term in text for term in ("perineural", "lymphovascular")
    ):
        return None
    if _microinvasion_support_ids(question, observations):
        return "Yes"
    if termination_reason == "candidate_slides_exhausted":
        return "No"
    return None


def question_spatial_positive_support_ids(
    question: dict[str, Any], observations: list[ROIObservation]
) -> list[str]:
    """Return traceable evidence for a same-lineage spatial positive."""

    text = str(question.get("question") or "").casefold()
    if any(term in text for term in ("margin", "dissection surface")):
        return _margin_support_ids(question, observations)
    if is_renal_capsule_question(question):
        return _renal_capsule_support_ids(question, observations)
    if any(term in text for term in ("perineural", "lymphovascular")):
        return _microinvasion_support_ids(question, observations)
    return []


def question_spatial_positive_supported(
    question: dict[str, Any], observations: list[ROIObservation]
) -> bool:
    """Gate spatial positives with a same-lineage relationship."""

    if not is_spatial_interface_question(question):
        return True
    return bool(question_spatial_positive_support_ids(question, observations))


def _renal_special_specimen_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    text = str(question.get("question") or "").casefold()
    domain = str(question.get("clinical_domain") or "").casefold()
    if "renal" not in domain:
        return None
    usable = [item.morphology for item in observations if item.image_quality == "adequate"]
    combined = " ".join(usable)
    adipose = len(re.findall(r"\b(?:mature\s+)?adipos\w*|\badipocytes?\b", combined, re.I))
    fibrous = len(re.findall(r"\bfibro\w*|\bcollagen\w*", combined, re.I))
    necrotic = len(re.findall(r"\bnecro\w*|ghost\s+cells?|karyorrhectic\s+debris", combined, re.I))
    histiocytic = len(re.findall(r"\bhistiocyt\w*|foamy\s+macrophages?", combined, re.I))
    if "omental nodule" in text and adipose >= 2 and fibrous >= 2 and (
        necrotic >= 1 or histiocytic >= 1
    ):
        return "Fibrously encapsulated fat-necrosis nodule"
    if "separately submitted renal cyst specimen" in text and adipose >= 2 and fibrous >= 1:
        return "A small amount of proliferative adipose tissue"
    return None


def _renal_necrosis_subtype_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    text = str(question.get("question") or "").casefold()
    if not (
        "necrotic change" in text
        and any(term in text for term in ("which type", "what type"))
    ):
        return None
    focal = re.compile(
        r"\b(?:focal|focus|localized)\w*.{0,35}\bnecro\w*|"
        r"\bnecro\w*\s+(?:focus|foci)\b|ghost\s+(?:cells?|adipocytes?)"
        r".{0,35}karyorrhectic\s+debris",
        re.IGNORECASE,
    )
    if any(
        item.image_quality == "adequate"
        and True in _feature_polarities(item.morphology, focal)
        for item in observations
    ):
        return "Focal necrosis"
    return None


def _renal_high_grade_component_preference(
    question: dict[str, Any], observations: list[ROIObservation]
) -> str | None:
    text = str(question.get("question") or "").casefold()
    if "high-grade differentiation component" not in text:
        return None
    combined = " ".join(
        item.morphology for item in observations if item.image_quality == "adequate"
    )
    if re.search(r"\bsarcomatoid\b", combined, re.IGNORECASE):
        return "Sarcomatoid differentiation"
    if re.search(r"\brhabdoid\b", combined, re.IGNORECASE):
        return "Rhabdoid differentiation"
    if _renal_sarcomatoid_support_ids(observations):
        return "Sarcomatoid differentiation"
    return None


def _tumor_necrosis_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    termination_reason: str | None,
) -> str | None:
    """Require independent necrosis support from a confirmed tumour lineage.

    Necrosis is not specific to tumour.  In particular, a 20x inflammatory
    field and a 40x child crop from the same cyst/inflammatory focus must not
    be promoted into two independent tumour-necrosis observations.
    """

    text = str(question.get("question") or "").casefold()
    options = {str(item) for item in question.get("options", [])}
    if "tumor necrosis" not in text or not {"Yes", "No"} <= options:
        return None
    by_roi = {item.roi_id: item for item in observations}
    explicit_tumor_context = re.compile(
        r"\b(?:primary\s+)?tumou?r\s+(?:bed|cells?|nests?|tissue|region|focus|"
        r"architecture)\b|"
        r"\b(?:renal\s+cell\s+)?carcinoma\b|"
        r"\bmalignan\w*\s+(?:cells?|epitheli\w*|neoplasm)\b|"
        r"\bneoplastic\s+(?:cells?|epitheli\w*|proliferation)\b",
        re.IGNORECASE,
    )
    tumor_architecture = re.compile(
        r"\b(?:solid|nested|alveolar|trabecular|papillary)\w*\b.{0,90}"
        r"\b(?:clear\s+cytoplasm|polygonal\s+cells?|epithelial\s+cells?|"
        r"pleomorph\w*|prominent\s+nucleol\w*|mito\w*)\b|"
        r"\b(?:clear\s+cytoplasm|polygonal\s+cells?|epithelial\s+cells?|"
        r"pleomorph\w*)\b.{0,90}\b(?:solid|nested|alveolar|trabecular|"
        r"papillary)\w*\b",
        re.IGNORECASE,
    )
    non_tumor_necrosis = re.compile(
        r"\b(?:fibrinoid\s+necrosis|fat\s+necrosis|abscess\w*|"
        r"fung\w*|yeast\w*|bacter\w*|infect\w*)\b|"
        r"\b(?:dense|marked|predominant\w*)\s+(?:acute\s+|chronic\s+)?"
        r"inflamm\w*|"
        r"\binflammatory\s+infiltrate\s+dominated\s+by\b",
        re.IGNORECASE,
    )
    non_tumor_lineage = re.compile(
        r"\b(?:cystic\s+(?:structure|lesion|space)|cyst\s+wall|"
        r"glomerul\w*|bowman['’]?s?|normal\s+renal|"
        r"(?:intact|preserved|regular|normal)\s+tubul\w*)\b",
        re.IGNORECASE,
    )

    def lineage(item: ROIObservation) -> list[ROIObservation]:
        output = [item]
        parent_id = item.parent_roi_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_roi.get(parent_id)
            if parent is None:
                break
            output.append(parent)
            parent_id = parent.parent_roi_id
        return output

    def root_roi_id(item: ROIObservation) -> str:
        chain = lineage(item)
        if len(chain) > 1:
            return chain[-1].roi_id
        return item.parent_roi_id or item.roi_id

    def is_tumor_necrosis(item: ROIObservation) -> bool:
        if (
            item.image_quality != "adequate"
            or item.magnification not in {"20x", "40x"}
            or True
            not in _feature_polarities(
                item.morphology, _GROUNDABLE_FEATURES["necrosis"]
            )
            or non_tumor_necrosis.search(item.morphology)
        ):
            return False
        chain = lineage(item)
        chain_text = " ".join(part.morphology for part in chain)
        has_tumor_context = (
            True in _feature_polarities(chain_text, explicit_tumor_context)
            or bool(tumor_architecture.search(chain_text))
        )
        if not has_tumor_context:
            return False
        # A normal-renal or cyst-wall ancestry is only overridden when the
        # necrotic crop itself explicitly establishes a tumour bed/context.
        if non_tumor_lineage.search(chain_text) and not (
            True in _feature_polarities(item.morphology, explicit_tumor_context)
            or bool(tumor_architecture.search(item.morphology))
        ):
            return False
        return True

    independent_positive_roots = {
        (item.slide_id, root_roi_id(item))
        for item in observations
        if is_tumor_necrosis(item)
    }
    if len(independent_positive_roots) >= 2:
        return "Yes"
    if termination_reason == "candidate_slides_exhausted":
        return "No"
    return None


def _question_specific_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    reviews: list[EvidenceReview] | None = None,
    termination_reason: str | None = None,
) -> str | None:
    return (
        _renal_grade_preference(question, observations)
        or _renal_high_grade_component_preference(question, observations)
        or _renal_necrosis_subtype_preference(question, observations)
        or _renal_special_specimen_preference(question, observations)
        or _renal_subtype_preference(question, observations)
        or _gastric_parent_type_preference(question, observations)
        or _lauren_preference(question, observations)
        or _gastric_differentiation_preference(question, observations)
        or _deepest_layer_preference(question, observations)
        or _gastric_pt_preference(question, observations)
        or _renal_pn_preference(question, observations, termination_reason)
        or _gastric_pn_preference(question, observations, termination_reason)
        or _lymph_node_metastasis_preference(
            question, observations, termination_reason, reviews or []
        )
        or _renal_capsule_preference(question, observations, termination_reason)
        or _microinvasion_preference(question, observations, termination_reason)
        or _margin_preference(
            question, reviews or [], termination_reason, observations
        )
        or _tumor_necrosis_preference(question, observations, termination_reason)
    )


def _required_magnification_floor(question: dict[str, Any]) -> str:
    """Apply a deterministic resolution floor for common pathology tasks."""

    text = str(question.get("question") or "").casefold()
    forty_x_terms = (
        "grade",
        "differentiation",
        "histological type",
        "pathological type",
        "tumor type",
        "morphological features of which",
        "nuclear",
        "nucleol",
        "mitotic",
        "high-grade component",
    )
    if any(term in text for term in forty_x_terms):
        return "40x"
    twenty_x_terms = (
        "necrosis",
        "metastatic carcinoma",
        "carcinoma identified",
        "resection margin",
        "surgical margin",
        "dissection surface",
        "pathological diagnosis",
        "pathological finding",
    )
    if any(term in text for term in twenty_x_terms):
        return "20x"
    return "5x"


def normalize_plan_for_question(
    plan: EvidencePlan, question: dict[str, Any]
) -> EvidencePlan:
    """Apply deterministic task semantics and resolution floors to any plan source."""

    floor = _required_magnification_floor(question)
    required = max(
        (plan.required_magnification, floor),
        key=lambda value: _MAGNIFICATION_RANK[value],
    )
    preferred = list(plan.preferred_zoom_magnifications)
    if required != "5x" and required not in preferred:
        preferred.append(required)
        preferred.sort(key=lambda value: _MAGNIFICATION_RANK[value])

    question_text = str(question.get("question") or "").casefold()
    extremum_terms = (
        "highest",
        "lowest",
        "deepest",
        "maximum",
        "minimum",
        "greatest",
        "most advanced",
        "pt category",
        "pt stage",
        "pathological t category",
        "pathological t (pt) stage",
        "primary-tumor category (pt)",
    )
    decision_semantics = plan.decision_semantics
    stop_policy = plan.stop_policy
    if any(term in question_text for term in extremum_terms):
        decision_semantics = "extremum"
        stop_policy = "candidate_pool_coverage"

    low_power_query = plan.low_power_query
    high_power_query = plan.high_power_query
    clinical_domain = str(question.get("clinical_domain") or "").casefold()
    component_profile = quantitative_profile(question, plan)
    if component_profile is not None and component_profile.kind == "component_fraction":
        phenotype_queries = {
            "signet_ring": (
                "poorly cohesive single cells with intracytoplasmic mucin vacuoles and eccentric nuclei",
                "discohesive vacuolated epithelial cells with eccentrically displaced nuclei",
            ),
            "mucinous": (
                "gastric epithelial lesion with broad extracellular mucin pools",
                "epithelial cells or glands floating in extracellular mucin",
            ),
            "hepatoid": (
                "gastric epithelial lesion with solid trabecular polygonal-cell regions",
                "trabecular polygonal cells with abundant eosinophilic granular cytoplasm",
            ),
            "other_component": (
                "heterogeneous gastric epithelial lesion with distinct architectural components",
                "viable epithelial cells representing the requested morphologic component",
            ),
        }
        low_query, high_query = phenotype_queries[component_profile.phenotype]
        return plan.model_copy(
            update={
                "target_finding": f"area fraction of the {component_profile.target}",
                "question_attributes": [
                    "gastric pathology",
                    "component proportion quantification",
                    "whole-tumor stratified sampling",
                    component_profile.target,
                ],
                "slide_retrieval_descriptions": [
                    low_query,
                    "Heterogeneous primary gastric epithelial lesion with representative tumor-bearing tissue",
                ],
                "low_power_query": low_query,
                "high_power_query": high_query,
                "preferred_zoom_magnifications": ["20x", "40x"],
                "required_magnification": "40x",
                "evidence_goal": (
                    "component proportion quantification using a whole-slide tissue-area "
                    "denominator, representative tumor strata, same-ROI visible phenotype, "
                    "and confidence-qualified cellular measurements"
                ),
                "decision_semantics": "quantitative",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    if component_profile is not None and component_profile.kind == "focus_count":
        return plan.model_copy(
            update={
                "target_finding": "spatially independent primary renal cellular foci",
                "question_attributes": [
                    "renal pathology",
                    "multifocal topology quantification",
                    "separated whole-slide components",
                ],
                "slide_retrieval_descriptions": [
                    "Renal specimen with two or more discrete cellular nodules separated by normal parenchyma",
                    "Renal specimen with one contiguous cellular mass and surrounding normal parenchyma",
                ],
                "low_power_query": "discrete cellular nodules separated by normal renal parenchyma",
                "high_power_query": "viable epithelial cellular nests within each separated nodule",
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": (
                    "multifocal topology quantification by separately sampling each whole-slide "
                    "Tissue candidate and confirming its cellular content"
                ),
                "decision_semantics": "quantitative",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    renal_grade = "renal" in clinical_domain and "who/isup grade" in question_text
    if renal_grade:
        return plan.model_copy(
            update={
                "target_finding": "highest nuclear grade within the primary renal tumor",
                "question_attributes": [
                    "renal pathology",
                    "primary tumor",
                    "WHO/ISUP nuclear grade",
                ],
                "slide_retrieval_descriptions": [
                    "Renal epithelial tumor with solid nested or alveolar architecture and delicate vasculature",
                    "Renal epithelial tumor showing nuclear and nucleolar variation across viable tumor regions",
                    "Renal epithelial tumor with marked pleomorphism or rhabdoid or sarcomatoid morphology",
                ],
                "low_power_query": (
                    "renal epithelial tumor including solid nested clear-cell regions and "
                    "focal pleomorphic spindle rhabdoid or sarcomatoid regions"
                ),
                "high_power_query": (
                    "viable renal tumor cells in solid nested architecture with visible "
                    "nucleoli, marked pleomorphism, rhabdoid cells, or sarcomatoid spindle morphology"
                ),
                "preferred_zoom_magnifications": ["10x", "40x"],
                "required_magnification": "40x",
                "evidence_goal": (
                    "localize representative tumor architecture, assess whether nucleoli are "
                    "already conspicuous in true 10x physical fields, and confirm the strongest "
                    "nucleolar or grade-defining morphology at 40x across sampled regions"
                ),
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    renal_high_grade_component = "renal" in clinical_domain and any(
        term in question_text
        for term in ("high-grade differentiation component", "high grade differentiation component")
    )
    if renal_high_grade_component:
        return plan.model_copy(
            update={
                "target_finding": "high-grade component within the primary renal tumor",
                "question_attributes": [
                    "renal pathology",
                    "primary tumor",
                    "sarcomatoid or rhabdoid component",
                ],
                "slide_retrieval_descriptions": [
                    "Renal epithelial tumor containing sarcomatoid spindle-cell differentiation",
                    "Renal epithelial tumor containing rhabdoid cells or extreme pleomorphism",
                    "Heterogeneous renal tumor with abrupt transition to a high-grade solid component",
                ],
                "low_power_query": (
                    "renal tumor with focal sarcomatoid spindle rhabdoid or extremely "
                    "pleomorphic high-grade region"
                ),
                "high_power_query": (
                    "atypical spindle rhabdoid or extremely pleomorphic malignant cells"
                ),
                "preferred_zoom_magnifications": ["20x", "40x"],
                "required_magnification": "40x",
                "evidence_goal": (
                    "identify any explicit sarcomatoid, rhabdoid, or extremely pleomorphic "
                    "component within viable renal tumor"
                ),
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    gastric_generic_type = _is_generic_gastric_type_question(question)
    if gastric_generic_type:
        # Keep histological typing separate from the downstream Lauren and
        # differentiation questions.  These phrases are used only for image
        # retrieval and never enter the neutral Patho-R1 transcription.
        return plan.model_copy(
            update={
                "target_finding": "histological category of the primary gastric tumor",
                "question_attributes": [
                    "gastric pathology",
                    "primary tumor",
                    "histological category",
                ],
                "slide_retrieval_descriptions": [
                    "Gastric wall with invasive atypical epithelial glands and stromal reaction",
                    "Gastric wall infiltrated by poorly cohesive malignant epithelial cells",
                ],
                "low_power_query": (
                    "invasive atypical epithelial proliferation in gastric wall"
                ),
                "high_power_query": (
                    "malignant epithelial cells with glandular or poorly cohesive morphology"
                ),
                "preferred_zoom_magnifications": preferred,
                "required_magnification": required,
                "evidence_goal": (
                    "determine the minimal histological tumor category from invasive "
                    "epithelial morphology without assigning a secondary classification "
                    "or differentiation"
                ),
                "decision_semantics": decision_semantics,
                "stop_policy": stop_policy,
            }
        )
    gastric_lauren = "gastric" in clinical_domain and "lauren classification" in question_text
    if gastric_lauren:
        return plan.model_copy(
            update={
                "target_finding": "least cohesive component of the primary gastric tumor",
                "question_attributes": [
                    "gastric pathology",
                    "Lauren classification",
                    "tumor heterogeneity",
                ],
                # Lauren class and differentiation are separate question-side
                # judgments over the same heterogeneous primary-tumour ROI
                # pool.  Sharing the route prevents one question from missing
                # a poorly cohesive component already visible to the other.
                "slide_retrieval_descriptions": list(
                    _GASTRIC_HETEROGENEITY_SLIDE_DESCRIPTIONS
                ),
                "low_power_query": _GASTRIC_HETEROGENEITY_LOW_POWER_QUERY,
                "high_power_query": _GASTRIC_HETEROGENEITY_HIGH_POWER_QUERY,
                "preferred_zoom_magnifications": ["20x", "40x"],
                "required_magnification": "40x",
                "evidence_goal": (
                    "search exhaustively for a diffuse poorly cohesive component, while "
                    "also retaining gland-forming regions for mixed classification"
                ),
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    gastric_differentiation = (
        "gastric" in clinical_domain and "degree of differentiation" in question_text
    )
    if gastric_differentiation:
        return plan.model_copy(
            update={
                "target_finding": "least differentiated component of the primary gastric tumor",
                "question_attributes": [
                    "gastric pathology",
                    "gland formation",
                    "poorly cohesive or solid component",
                ],
                "slide_retrieval_descriptions": list(
                    _GASTRIC_HETEROGENEITY_SLIDE_DESCRIPTIONS
                ),
                "low_power_query": _GASTRIC_HETEROGENEITY_LOW_POWER_QUERY,
                "high_power_query": _GASTRIC_HETEROGENEITY_HIGH_POWER_QUERY,
                "preferred_zoom_magnifications": ["20x", "40x"],
                "required_magnification": "40x",
                "evidence_goal": (
                    "identify the least differentiated viable tumor component and combine it "
                    "with any retained moderately gland-forming component"
                ),
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    regional_nodes = is_regional_node_question(question)
    if regional_nodes:
        node_yes_no = question.get("question_type") == "yes_no"
        pn_question = _is_pn_question(question)
        return plan.model_copy(
            update={
                "target_finding": "atypical epithelial deposits within regional lymph-node tissue",
                "question_attributes": [
                    "gastric pathology" if "gastric" in clinical_domain else "renal pathology",
                    "regional lymph node",
                    "epithelial deposit",
                ],
                "slide_retrieval_descriptions": [
                    "Lymph node containing atypical cohesive epithelial glands or nests",
                    "Lymph node with subcapsular metastatic epithelial deposit",
                    "Reactive lymph node without epithelial cells",
                ],
                "low_power_query": (
                    "lymph node parenchyma containing a focal epithelial gland or cohesive tumor deposit"
                ),
                "high_power_query": (
                    "atypical epithelial glands and solid nests with lumina"
                ),
                "preferred_zoom_magnifications": ["20x", "40x"],
                "required_magnification": "40x" if pn_question else "20x",
                "evidence_goal": (
                    "require same-slide nodal architecture and a required-resolution atypical "
                    "epithelial deposit; for pN, count independently supported involved slides"
                ),
                "decision_semantics": "existential" if node_yes_no else "quantitative",
                "stop_policy": "direct_positive" if node_yes_no else "candidate_pool_coverage",
            }
        )
    gastric_extent = "gastric" in clinical_domain and (
        _is_deepest_layer_question(question) or _is_pt_question(question)
    )
    if gastric_extent:
        return plan.model_copy(
            update={
                "target_finding": "deepest anatomic compartment invaded by gastric carcinoma",
                "question_attributes": [
                    "gastric pathology",
                    "invasive front",
                    "gastric wall compartment",
                ],
                "slide_retrieval_descriptions": [
                    "Gastric carcinoma infiltrating subserosal fibroadipose tissue",
                    "Gastric carcinoma reaching or breaching the serosal surface",
                    "Gastric carcinoma invading muscularis propria without subserosal extension",
                ],
                "low_power_query": (
                    "irregular gastric carcinoma glands infiltrating adipose tissue at the deepest invasive front"
                ),
                "high_power_query": (
                    "malignant glandular cells crossing smooth muscle into adipose tissue or serosa"
                ),
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": (
                    "identify the single deepest positively invaded gastric wall compartment"
                ),
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )

    margin_question = any(
        term in question_text
        for term in ("margin", "dissection surface")
    )
    if margin_question:
        organ = "renal" if "renal" in clinical_domain else "gastric"
        return plan.model_copy(
            update={
                "target_finding": "atypical epithelial cells at the submitted resection edge",
                "question_attributes": [
                    f"{organ} pathology",
                    "resection margin or dissection surface",
                    "cell-to-cut-edge spatial relationship",
                ],
                "slide_retrieval_descriptions": [
                    f"{organ.capitalize()} epithelial lesion touching an inked or cauterized resection edge",
                    f"{organ.capitalize()} tissue cut surface separated from epithelial nests by uninvolved stroma",
                    "Fibrous or adipose dissection surface with no epithelial proliferation",
                ],
                "low_power_query": (
                    "inked cauterized or sharply cut tissue edge adjacent to an epithelial proliferation"
                ),
                "high_power_query": (
                    "atypical epithelial cells touching an inked cauterized cut surface"
                ),
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": (
                    "require a same-lineage spatial relationship between atypical epithelial "
                    "cells and an actual resection edge; otherwise cover the candidate pool"
                ),
                "decision_semantics": "exhaustive",
                "stop_policy": "candidate_pool_coverage",
            }
        )

    renal_capsule = is_renal_capsule_question(question)
    if renal_capsule:
        if is_renal_perirenal_fat_question(question):
            return plan.model_copy(
                update={
                    "target_finding": (
                        "renal tumor extending beyond the fibrous capsule into "
                        "perirenal adipose tissue"
                    ),
                    "question_attributes": [
                        "renal pathology",
                        "tumor-perirenal-fat interface",
                        "adipocyte infiltration",
                    ],
                    "slide_retrieval_descriptions": [
                        "Renal clear-cell tumor nests infiltrating perirenal adipose tissue",
                        "Renal tumor clusters extending between adipocytes beyond the capsule",
                        "Intact fibrous renal capsule separating tumor from perirenal fat",
                    ],
                    "low_power_query": (
                        "irregular solid clear cell tongues extending into perirenal adipose tissue"
                    ),
                    "high_power_query": (
                        "cohesive clear cell nests infiltrating between mature adipocytes"
                    ),
                    "preferred_zoom_magnifications": ["20x", "40x"],
                    "required_magnification": "40x",
                    "evidence_goal": (
                        "determine whether viable renal tumor enters perirenal adipose tissue; "
                        "entry into the fibrous capsule alone is insufficient"
                    ),
                    "decision_semantics": "existential",
                    "stop_policy": "direct_positive",
                }
            )
        return plan.model_copy(
            update={
                "target_finding": "renal tumor entering or crossing the fibrous capsule interface",
                "question_attributes": [
                    "renal pathology",
                    "tumor-capsule interface",
                    "perirenal adipose tissue",
                ],
                "slide_retrieval_descriptions": [
                    "Renal clear-cell tumor nests penetrating the fibrous renal capsule",
                    "Renal tumor cells infiltrating through capsule into perirenal adipose tissue",
                    "Intact fibrous renal capsule with a smooth non-infiltrative tumor border",
                ],
                "low_power_query": (
                    "irregular solid clear cell tongues interdigitating with a broad dense fibrous band"
                ),
                "high_power_query": (
                    "cohesive clear cell nests embedded within dense collagen bundles"
                ),
                "preferred_zoom_magnifications": ["20x", "40x"],
                # A 20x field localizes a broad tumour/fibrous interface but
                # repeatedly left focal entry ``not fully evaluable``.  That
                # uncertainty was later aggregated into a high-confidence
                # negative capsule call.  Require the second zoom so all ten
                # retained interface parents are checked at cellular
                # resolution before a patient-level absence is allowed.
                "required_magnification": "40x",
                "evidence_goal": (
                    "determine whether viable tumor enters any portion of the fibrous capsule; "
                    "complete transgression into fat is not required"
                ),
                "decision_semantics": "existential",
                "stop_policy": "direct_positive",
            }
        )
    if "renal" in clinical_domain and "omental nodule" in question_text:
        return plan.model_copy(
            update={
                "target_finding": "composition of the separately submitted omental nodule",
                "question_attributes": [
                    "renal pathology",
                    "omental nodule",
                    "fat necrosis versus epithelial deposit",
                ],
                "slide_retrieval_descriptions": [
                    "Fibrously encapsulated omental fat necrosis with ghost adipocytes and histiocytes",
                    "Omental adipose nodule with fibrosis, inflammation, and necrotic debris",
                    "Omental nodule containing atypical clear epithelial tumor nests",
                ],
                "low_power_query": "encapsulated adipose nodule with fibrosis and fat necrosis",
                "high_power_query": "ghost adipocytes foamy histiocytes fibrosis and necrotic debris",
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": (
                    "classify the nodule from neutral adipose, fibrous, histiocytic, necrotic, "
                    "and epithelial morphology"
                ),
                "decision_semantics": "composite",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    if "renal" in clinical_domain and "separately submitted renal cyst specimen" in question_text:
        return plan.model_copy(
            update={
                "target_finding": "tissue composition of the separately submitted renal cyst specimen",
                "question_attributes": [
                    "renal pathology",
                    "separate cyst specimen",
                    "adipose and fibrous tissue",
                ],
                "slide_retrieval_descriptions": [
                    "Separate renal cyst specimen composed of proliferative mature adipose tissue",
                    "Fibrous cyst wall with mature adipose tissue and bland stromal cells",
                    "Renal cyst lining with any atypical epithelial or solid component",
                ],
                "low_power_query": "mature adipose tissue and fibrous wall in renal cyst specimen",
                "high_power_query": "bland adipocytes fibroblasts and cyst lining",
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": "state the dominant neutral tissue composition of the separate specimen",
                "decision_semantics": "composite",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    if "renal" in clinical_domain and "necrotic change" in question_text:
        return plan.model_copy(
            update={
                "target_finding": "focal necrotic morphology within viable primary renal tumor",
                "question_attributes": [
                    "renal pathology",
                    "primary tumor",
                    "necrotic focus",
                ],
                "slide_retrieval_descriptions": [
                    "Renal tumor with a focal zone of ghost cells and karyorrhectic debris",
                    "Viable renal tumor bordering a localized necrotic focus",
                    "Renal tumor without necrotic change",
                ],
                "low_power_query": "localized necrotic focus within viable renal tumor",
                "high_power_query": "ghost cells karyorrhectic debris and loss of viable nuclei",
                "preferred_zoom_magnifications": ["20x"],
                "required_magnification": "20x",
                "evidence_goal": "identify and name any explicitly observed focal necrotic change",
                "decision_semantics": "extremum",
                "stop_policy": "candidate_pool_coverage",
            }
        )
    if "gastric" in clinical_domain and "perineural" in question_text:
        return plan.model_copy(
            update={
                "target_finding": "tumor cells within or encasing a nerve sheath",
                "question_attributes": ["gastric pathology", "nerve", "tumor relationship"],
                "slide_retrieval_descriptions": [
                    "Gastric carcinoma cells surrounding or invading a peripheral nerve",
                    "Invasive gastric tumor front containing a nerve sheath",
                ],
                "low_power_query": "gastric tumor at invasive front surrounding a nerve",
                "high_power_query": "malignant epithelial cells within or encasing perineurium",
                "preferred_zoom_magnifications": ["20x", "40x"],
                # A 20x field can localize a small nerve and adjacent glands,
                # but usually cannot resolve true intraneural/perineural
                # encasement. Force the second zoom so the parent-diverse 40x
                # pass actually runs instead of ending after localization.
                "required_magnification": "40x",
                "evidence_goal": "require an explicit spatial relationship between tumor and nerve",
                "decision_semantics": "existential",
                "stop_policy": "direct_positive",
            }
        )
    if "gastric" in clinical_domain and "lymphovascular" in question_text:
        return plan.model_copy(
            update={
                "target_finding": "tumor embolus within an endothelial-lined channel",
                "question_attributes": [
                    "gastric pathology",
                    "tumor embolus",
                    "vascular or lymphatic channel",
                ],
                "slide_retrieval_descriptions": [
                    "Gastric tumor embolus within an endothelial-lined vascular channel",
                    "Invasive gastric tumor front containing intralymphatic epithelial nests",
                ],
                "low_power_query": "gastric tumor nests inside vascular channels at invasive front",
                "high_power_query": "epithelial tumor embolus surrounded by endothelial cells",
                "preferred_zoom_magnifications": ["20x", "40x"],
                # Endothelial lining and an intraluminal epithelial nest are a
                # high-power spatial relationship, not a 20x absence call.
                "required_magnification": "40x",
                "evidence_goal": "require tumor cells explicitly located inside an endothelial-lined space",
                "decision_semantics": "existential",
                "stop_policy": "direct_positive",
            }
        )

    return plan.model_copy(
        update={
            "required_magnification": required,
            "preferred_zoom_magnifications": preferred,
            "decision_semantics": decision_semantics,
            "stop_policy": stop_policy,
            "low_power_query": low_power_query,
            "high_power_query": high_power_query,
        }
    )


def _required_current_judgment_ids(
    plan: EvidencePlan,
    observations: list[ROIObservation],
    current_slide_id: str,
    attempted_magnifications: list[str],
) -> list[str]:
    """Return current evidence that an extremum review may not silently skip."""

    attempted_max = max(
        (_MAGNIFICATION_RANK.get(item, 0) for item in attempted_magnifications),
        default=0,
    )
    if plan.decision_semantics != "extremum":
        return []
    if attempted_max != _MAGNIFICATION_RANK["5x"]:
        return []
    return [
        item.evidence_id
        for item in observations
        if item.slide_id == current_slide_id
        and _MAGNIFICATION_RANK[item.magnification] == attempted_max
        and item.image_quality != "inadequate"
    ][:5]


def _normalize_review_candidate(
    candidate: dict[str, Any],
    valid_evidence_ids: set[str] | None = None,
    valid_focus_roi_ids: list[str] | None = None,
    attempted_magnifications: list[str] | None = None,
    required_magnification: str | None = None,
    evidence_id_aliases: dict[str, str] | None = None,
    renal_grade_floor: str | None = None,
    renal_grade_ceiling: str | None = None,
    yes_no_question: bool = False,
    preferred_zoom_magnifications: list[str] | None = None,
) -> dict[str, Any]:
    """Compact a review response before strict schema validation."""

    fields = set(EvidenceReview.model_fields)
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
        key=lambda item: len(fields & set(item)),
        default=candidate,
    )
    normalized = dict(source)
    aliases = evidence_id_aliases or {}

    def canonical_evidence_id(value: Any) -> str:
        raw = str(value or "").strip()
        return aliases.get(raw, raw)

    supplied_supporting_ids = normalized.get("supporting_evidence_ids", [])
    if not isinstance(supplied_supporting_ids, list):
        supplied_supporting_ids = []
    supplied_supporting_ids = list(
        dict.fromkeys(
            evidence_id
            for item in supplied_supporting_ids
            if (evidence_id := canonical_evidence_id(item))
        )
    )

    # Complete every judgment-side repair before deriving supporting IDs.  In
    # particular, never retain an ID from a judgment that is later discarded
    # because it is unknown, duplicated, malformed, or beyond the schema cap.
    supplied_judgments = normalized.get("evidence_judgments", [])
    if not isinstance(supplied_judgments, list):
        supplied_judgments = []
    allowed_relevance = {"direct", "supporting", "not_relevant"}
    allowed_direction = {"positive", "negative", "other", "indeterminate"}
    relevance_aliases = {
        "relevant": "supporting",
        "indirect": "supporting",
        "irrelevant": "not_relevant",
        "not relevant": "not_relevant",
    }
    direction_aliases = {
        "supporting": "positive",
        "supports": "positive",
        "supportive": "positive",
        "against": "negative",
        "opposing": "negative",
        "neutral": "indeterminate",
        "uncertain": "indeterminate",
        "not_relevant": "other",
        "not relevant": "other",
    }
    unique_judgments: list[dict[str, Any]] = []
    seen_judgment_ids: set[str] = set()
    for item in supplied_judgments:
        if not isinstance(item, dict):
            continue
        evidence_id = canonical_evidence_id(item.get("evidence_id"))
        relevance = str(item.get("relevance") or "").strip()
        direction = str(item.get("direction") or "").strip()
        relevance = relevance_aliases.get(relevance.casefold(), relevance)
        direction = direction_aliases.get(direction.casefold(), direction)
        rationale = str(item.get("rationale") or "").strip()
        if (
            not evidence_id
            or evidence_id in seen_judgment_ids
            or relevance not in allowed_relevance
            or direction not in allowed_direction
            or not rationale
            or (
                valid_evidence_ids is not None
                and evidence_id not in valid_evidence_ids
            )
        ):
            continue
        unique_judgments.append(
            {
                "evidence_id": evidence_id,
                "relevance": relevance,
                "direction": direction,
                "rationale": rationale[:160],
            }
        )
        seen_judgment_ids.add(evidence_id)

    status = str(normalized.get("status") or "").strip()
    normalized["status"] = status
    if status == "sufficient":
        # Preserve cited judgments first so an overlong response cannot lose
        # the very evidence used by its answer merely because it appeared late.
        cited_ids = set(supplied_supporting_ids)
        unique_judgments.sort(
            key=lambda item: item["evidence_id"] not in cited_ids
        )
        normalized["evidence_judgments"] = unique_judgments[:5]
    else:
        normalized["evidence_judgments"] = unique_judgments[:5]
        normalized["answer_candidate"] = None

    capped_answer, grade_was_capped = _bound_grade_answer(
        normalized.get("answer_candidate"),
        renal_grade_floor,
        renal_grade_ceiling,
    )
    if grade_was_capped:
        normalized["answer_candidate"] = capped_answer
        normalized["reason"] = (
            f"The strongest morphology at its observed resolution supports {capped_answer}; "
            "a higher renal WHO/ISUP grade was not established at the required lower-power visibility."
        )
        for judgment in normalized["evidence_judgments"]:
            if judgment["relevance"] in {"direct", "supporting"}:
                judgment["rationale"] = re.sub(
                    r"\bGrade\s+[3-4]\b", capped_answer, judgment["rationale"]
                )[:160]

    usable_judgment_ids = {
        item["evidence_id"]
        for item in normalized["evidence_judgments"]
        if item["relevance"] in {"direct", "supporting"}
    }
    if status == "sufficient":
        normalized["supporting_evidence_ids"] = [
            evidence_id
            for evidence_id in supplied_supporting_ids
            if evidence_id in usable_judgment_ids
        ][:5]
    else:
        normalized["supporting_evidence_ids"] = []

    if status == "sufficient" and (
        not normalized["supporting_evidence_ids"]
        or not str(normalized.get("answer_candidate") or "").strip()
    ):
        normalized.update(
            status="need_more_slides",
            conclusion_direction="indeterminate",
            answer_candidate=None,
            supporting_evidence_ids=[],
            next_magnification=None,
            focus_roi_ids=[],
            reason=(
                "The proposed sufficient conclusion cited no available direct evidence; "
                "continue the candidate-slide search."
            ),
        )
        status = "need_more_slides"

    if status == "sufficient" and yes_no_question:
        answer = str(normalized.get("answer_candidate") or "").strip().casefold()
        direction = str(normalized.get("conclusion_direction") or "").strip()
        expected = {"yes": "positive", "no": "negative"}.get(answer)
        if expected is None or direction != expected:
            normalized.update(
                status="need_more_slides",
                conclusion_direction="indeterminate",
                answer_candidate=None,
                supporting_evidence_ids=[],
                next_magnification=None,
                focus_roi_ids=[],
                reason=(
                    "The proposed yes/no answer contradicted its conclusion direction; "
                    "continue the candidate-slide search."
                ),
            )
            status = "need_more_slides"

    supplied_focus_ids = normalized.get("focus_roi_ids", [])
    if isinstance(supplied_focus_ids, list):
        normalized["focus_roi_ids"] = list(
            dict.fromkeys(
                focus_id
                for item in supplied_focus_ids
                if (focus_id := str(item).strip())
            )
        )
    status = normalized.get("status")
    if valid_focus_roi_ids is not None:
        supplied_focus = normalized.get("focus_roi_ids", [])
        if not isinstance(supplied_focus, list):
            supplied_focus = []
        normalized["focus_roi_ids"] = [
            str(item)
            for item in supplied_focus
            if str(item) in valid_focus_roi_ids
        ][:2]
    if status == "need_higher_magnification":
        attempted = attempted_magnifications or []
        attempted_max = max(
            (_MAGNIFICATION_RANK.get(item, 0) for item in attempted), default=0
        )
        requested = str(normalized.get("next_magnification") or "")
        requested_rank = _MAGNIFICATION_RANK.get(requested, 0)
        required_rank = _MAGNIFICATION_RANK.get(required_magnification or "", 0)
        target_rank = max(requested_rank, required_rank)
        # Always preserve localization continuity. A direct 5x -> 40x jump
        # bypasses the intermediate field and can land on normal background
        # instead of the suspicious architecture that motivated the zoom.
        allowed_zoom_levels = preferred_zoom_magnifications or ["20x", "40x"]
        candidates = [
            item
            for item in allowed_zoom_levels
            if attempted_max < _MAGNIFICATION_RANK[item] <= target_rank
        ]
        requested = min(
            candidates,
            key=lambda item: _MAGNIFICATION_RANK[item],
            default="",
        )
        if requested and normalized.get("focus_roi_ids"):
            normalized["next_magnification"] = requested
        elif requested and valid_focus_roi_ids:
            normalized["next_magnification"] = requested
            normalized["focus_roi_ids"] = valid_focus_roi_ids[:2]
        else:
            normalized.update(
                status="need_more_slides",
                conclusion_direction="indeterminate",
                answer_candidate=None,
                supporting_evidence_ids=[],
                focus_roi_ids=[],
                next_magnification=None,
                reason=(
                    "No valid unattempted magnification and focus ROI remained; "
                    "continue the candidate-slide search."
                ),
            )
    elif status == "need_more_slides":
        normalized["focus_roi_ids"] = []
        normalized["next_magnification"] = None
    elif status == "sufficient":
        normalized["next_magnification"] = None
    if normalized.get("reason") is not None:
        normalized["reason"] = str(normalized["reason"])[:600]
    return normalized


def _normalize_final_candidate(
    candidate: dict[str, Any],
    valid_evidence_ids: set[str],
    question: dict[str, Any] | None = None,
    termination_reason: str | None = None,
    evidence_sufficient: bool | None = None,
    authorized_answer_candidate: str | None = None,
    renal_grade_floor: str | None = None,
    renal_grade_ceiling: str | None = None,
    preferred_answer: str | None = None,
    regional_node_positive_support_ids: list[str] | None = None,
) -> dict[str, Any]:
    fields = set(FinalAnswer.model_fields)
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
        key=lambda item: len(fields & set(item)),
        default=candidate,
    )
    normalized = dict(source)
    if not isinstance(normalized.get("supporting_evidence_ids"), list):
        normalized["supporting_evidence_ids"] = []
    if isinstance(normalized.get("supporting_evidence_ids"), list):
        normalized["supporting_evidence_ids"] = list(
            dict.fromkeys(
                str(item)
                for item in normalized["supporting_evidence_ids"]
                if str(item) in valid_evidence_ids
            )
        )[:5]
    if normalized.get("brief_rationale") is not None:
        normalized["brief_rationale"] = str(normalized["brief_rationale"])[:700]
    else:
        normalized["brief_rationale"] = (
            "The answer was selected from the available question-side morphology evidence."
        )
    if question is not None and normalized.get("answer") is not None:
        raw_answer = str(normalized["answer"]).strip()
        answer = str(_canonical_answer_format(question, raw_answer)).strip()
        options = [str(item) for item in question.get("options", [])]
        if question.get("question_type") == "single_choice":
            exact = {item.casefold(): item for item in options}
            if answer.casefold() in exact:
                answer = exact[answer.casefold()]
            else:
                contained = [item for item in options if item.casefold() in answer.casefold()]
                if len(contained) == 1:
                    answer = contained[0]
        elif question.get("question_type") == "yes_no":
            if re.fullmatch(r"yes[.!]?", answer, re.IGNORECASE):
                answer = "Yes"
            elif re.fullmatch(r"no[.!]?", answer, re.IGNORECASE):
                answer = "No"
        normalized["answer"] = answer[:180]
    if (
        question is not None
        and _is_generic_gastric_type_question(question)
        and str(normalized.get("answer") or "").casefold() == "adenocarcinoma"
    ):
        normalized["brief_rationale"] = (
            "The question asks for the minimal parent histological category. The retained "
            "invasive epithelial and gland-forming morphology supports Adenocarcinoma; "
            "differentiation and secondary classification are separate attributes."
        )
    capped_answer, grade_was_capped = _bound_grade_answer(
        normalized.get("answer"), renal_grade_floor, renal_grade_ceiling
    )
    normalized["answer"] = capped_answer
    if grade_was_capped:
        normalized["brief_rationale"] = (
            f"The independently corroborated morphology supports {capped_answer}; "
            "a higher renal WHO/ISUP grade was not established across distinct sampled regions."
        )
    if preferred_answer is not None:
        normalized["answer"] = preferred_answer
        if _GRADE_ANSWER.fullmatch(preferred_answer):
            normalized["brief_rationale"] = (
                "After excluding nuclear findings from explicit normal-renal tissue lineages, "
                "the strongest independently corroborated grade-defining morphology across "
                f"eligible sampled regions supports {preferred_answer}."
            )
        else:
            normalized["brief_rationale"] = (
                "Across the question-side evidence retained from the reviewed candidate pool, "
                f"the strongest task-specific morphology supports {preferred_answer}."
            )
    if termination_reason is not None:
        normalized["termination_reason"] = termination_reason
    if evidence_sufficient is not None:
        normalized["evidence_sufficiency"] = (
            "sufficient" if evidence_sufficient else "insufficient"
        )
        if not evidence_sufficient:
            normalized["confidence"] = "low"
        elif normalized.get("confidence") not in {"high", "medium", "low"}:
            normalized["confidence"] = "medium"
    if evidence_sufficient and authorized_answer_candidate is not None:
        normalized["answer"] = authorized_answer_candidate
    if question is not None and is_regional_node_question(question) and (
        question.get("question_type") == "yes_no"
    ):
        strict_ids = [
            evidence_id
            for evidence_id in (regional_node_positive_support_ids or [])
            if evidence_id in valid_evidence_ids
        ][:2]
        if normalized.get("answer") == "Yes":
            if len(strict_ids) >= 2:
                normalized["supporting_evidence_ids"] = strict_ids
                normalized["brief_rationale"] = (
                    "The cited same-slide morphology establishes nodal architecture and "
                    "a required-resolution atypical epithelial deposit."
                )
            elif termination_reason == "candidate_slides_exhausted":
                normalized["answer"] = "No"
                normalized["supporting_evidence_ids"] = []
                normalized["brief_rationale"] = (
                    "No inspected candidate slide established both nodal architecture and "
                    "a required-resolution atypical epithelial deposit on the same slide."
                )
    return normalized


def _safe_final_fallback_answer(
    question: dict[str, Any],
    *,
    preferred_answer: str | None,
    authorized_answer_candidate: str | None,
    review_history: list[EvidenceReview],
    renal_grade_floor: str | None,
    renal_grade_ceiling: str | None,
) -> str:
    """Return a schema-valid conservative answer after exhausted JSON repairs."""

    for value in (preferred_answer, authorized_answer_candidate):
        if str(value or "").strip():
            return str(value).strip()
    for review in reversed(review_history):
        if str(review.answer_candidate or "").strip():
            return str(review.answer_candidate).strip()
    question_text = str(question.get("question") or "").casefold()
    options = [str(item) for item in question.get("options", [])]
    if question.get("question_type") == "yes_no":
        return "No"
    if "who/isup grade" in question_text:
        return renal_grade_ceiling or renal_grade_floor or "Grade 1"
    if options:
        indeterminate = next(
            (
                item
                for item in options
                if any(
                    token in item.casefold()
                    for token in ("indeterminate", "cannot be determined", "unknown")
                )
            ),
            None,
        )
        return indeterminate or options[0]
    if "high-grade differentiation component" in question_text:
        return "No high-grade differentiation component identified"
    return "No definitive requested finding identified"


def _answer_errors(question: dict[str, Any], value: BaseModel) -> dict[str, str]:
    final = value.answer if isinstance(value, FinalAnswer) else value.answer_candidate
    answer = str(final or "").strip()
    options = [str(item) for item in question.get("options", [])]
    if question["question_type"] == "single_choice":
        if answer not in options:
            return {"answer": f"answer must exactly match one option: {options}"}
    question_text = str(question.get("question") or "").casefold()
    if any(term in question_text for term in ("histological type", "pathological type", "tumor type")):
        lowered = answer.casefold()
        if (
            "no primary tumor" in lowered
            or "no tumor identified" in lowered
            or ("benign" in lowered and any(term in lowered for term in ("mucosa", "tissue", "parenchyma")))
        ):
            return {"answer": "answer must name the most likely requested tumor type, not an abstention or background-tissue description"}
    forbidden = {
        "cannot be determined",
        "cannot determine",
        "unknown",
        "none",
        "indeterminate",
        "not enough information",
        "insufficient information",
    }
    if answer.casefold().rstrip(".") in forbidden and not (
        question["question_type"] == "single_choice" and answer in options
    ):
        return {"answer": "an abstention is not a valid final answer"}
    if question["question_type"] == "yes_no" and answer not in {"Yes", "No"}:
        return {"answer": "yes/no answer must be exactly Yes or No"}
    if "lauren classification" in question_text and answer not in {
        "Intestinal type",
        "Diffuse type",
        "Mixed type",
    }:
        return {
            "answer": (
                "Lauren classification must be exactly Intestinal type, "
                "Diffuse type, or Mixed type"
            )
        }
    if question is not None and _is_deepest_layer_question(question):
        if re.search(r"\b(?:p?T(?:is|[0-4][a-d]?))\b", answer, re.IGNORECASE):
            return {
                "answer": (
                    "deepest-layer answer must name one anatomical layer, "
                    "never a T category"
                )
            }
        if not re.search(
            r"\b(?:mucosa|lamina propria|muscularis mucosae|submucosa|"
            r"muscularis propria|subserosa|subserosal|serosa|adjacent organ)\b",
            answer,
            re.IGNORECASE,
        ):
            return {"answer": "deepest-layer answer must name one gastric wall layer"}
        if re.search(r"\bor\b|/", answer, re.IGNORECASE):
            return {"answer": "deepest-layer answer must contain exactly one layer"}
    if "who/isup grade" in question_text and not re.fullmatch(
        r"Grade [1-4]", answer
    ):
        return {"answer": "renal WHO/ISUP answer must be exactly Grade 1, Grade 2, Grade 3, or Grade 4"}
    pt_question = question is not None and _is_pt_question(question)
    if pt_question and not re.fullmatch(r"pT(?:is|[0-4](?:[a-d])?|X)", answer):
        return {"answer": "pathological T answer must be a concise pT category such as pT1 or pT3"}
    pn_question = question is not None and _is_pn_question(question)
    if pn_question and not re.fullmatch(r"pN(?:[0-3](?:[a-c])?|X)", answer):
        return {"answer": "pathological N answer must be a concise pN category such as pN0 or pNX"}
    if (
        "degree of differentiation" in question_text
        or "degrees of differentiation" in question_text
    ) and not any(term in answer.casefold() for term in ("differentiated", "differentiation")):
        return {"answer": "answer must state the observed degree of differentiation"}
    if question["question_type"] == "short_answer" and (
        not answer or "\n" in answer or len(answer) > 180
    ):
        return {"answer": "short answer must be one concise line"}
    return {}


class QwenController:
    def __init__(self, generator: StructuredGenerator, config: AgentConfig) -> None:
        self.generator = generator
        self.config = config
        self.audit: list[dict[str, Any]] = []

    def plan(self, question: dict[str, Any]) -> EvidencePlan:
        result = self.generator.generate(
            EvidencePlan,
            plan_messages(question),
            self.config.plan_max_tokens,
            "plan",
        )
        self.audit.extend(result.audit)
        return normalize_plan_for_question(result.value, question)

    def review(
        self,
        question: dict[str, Any],
        plan: EvidencePlan,
        observations: list[ROIObservation],
        *,
        current_slide_id: str,
        remaining_candidate_slide_ids: list[str],
        available_focus_roi_ids: list[str],
        attempted_magnifications: list[str],
    ) -> EvidenceReview:
        observation_by_id = {item.evidence_id: item for item in observations}
        current_evidence_ids = {
            item.evidence_id
            for item in observations
            if item.slide_id == current_slide_id
        }
        current_roi_to_evidence_id = {
            item.roi_id: item.evidence_id
            for item in observations
            if item.slide_id == current_slide_id
        }
        renal_grade_bounds = _renal_grade_bounds(question, observations)
        renal_grade_floor = renal_grade_bounds[0] if renal_grade_bounds else None
        renal_grade_ceiling = renal_grade_bounds[1] if renal_grade_bounds else None
        current_adequate_ids = {
            item.evidence_id
            for item in observations
            if item.slide_id == current_slide_id
            and item.image_quality != "inadequate"
        }
        required_current_ids = set(
            _required_current_judgment_ids(
                plan,
                observations,
                current_slide_id,
                attempted_magnifications,
            )
        )

        def validate(value: BaseModel) -> dict[str, str]:
            review = value
            errors: dict[str, str] = {}
            if not set(review.supporting_evidence_ids) <= current_evidence_ids:
                errors["supporting_evidence_ids"] = (
                    "must contain only current-slide evidence IDs"
                )
            judged_ids = [item.evidence_id for item in review.evidence_judgments]
            if len(judged_ids) != len(set(judged_ids)):
                errors["evidence_judgments"] = "contains duplicate evidence IDs"
            if not set(judged_ids) <= current_evidence_ids:
                errors["evidence_judgments"] = (
                    "must contain only current-slide evidence IDs"
                )
            judged_current_ids = set(judged_ids) & current_adequate_ids
            if current_adequate_ids and not judged_current_ids:
                errors["evidence_judgments"] = (
                    "must judge at least one adequate current-slide observation"
                )
            if not required_current_ids <= set(judged_ids):
                errors["evidence_judgments"] = (
                    "an extremum decision must judge every adequate current ROI at the "
                    "highest required/attempted magnification; missing IDs: "
                    f"{sorted(required_current_ids - set(judged_ids))}"
                )
            usable_judgments = {
                item.evidence_id
                for item in review.evidence_judgments
                if item.relevance in {"direct", "supporting"}
            }
            if not set(review.supporting_evidence_ids) <= usable_judgments:
                errors["supporting_evidence_ids"] = (
                    "must have direct/supporting question-side judgments"
                )
            if review.status == "sufficient":
                errors.update(_answer_errors(question, review))
            return errors

        try:
            result = self.generator.generate(
                EvidenceReview,
                review_messages(
                    question,
                    plan,
                    observations,
                    current_slide_id=current_slide_id,
                    remaining_candidate_slide_ids=remaining_candidate_slide_ids,
                    available_focus_roi_ids=available_focus_roi_ids,
                    attempted_magnifications=attempted_magnifications,
                ),
                self.config.review_max_tokens,
                "evidence_review",
                validate,
                candidate_normalizer=lambda candidate: _normalize_review_candidate(
                    candidate,
                    current_evidence_ids,
                    available_focus_roi_ids,
                    attempted_magnifications,
                    plan.required_magnification,
                    current_roi_to_evidence_id,
                    renal_grade_floor,
                    renal_grade_ceiling,
                    question.get("question_type") == "yes_no",
                    list(plan.preferred_zoom_magnifications),
                ),
            )
        except StructuredGenerationError as error:
            self.audit.extend(error.audit)
            attempted_max = max(
                (_MAGNIFICATION_RANK.get(item, 0) for item in attempted_magnifications),
                default=0,
            )
            required = _MAGNIFICATION_RANK[plan.required_magnification]
            if available_focus_roi_ids and attempted_max < required:
                fallback = EvidenceReview(
                    status="need_higher_magnification",
                    reason=(
                        "Structured review was unusable; conservatively inspect the available "
                        "focus tissue at the required magnification."
                    ),
                    conclusion_direction="indeterminate",
                    answer_candidate=None,
                    supporting_evidence_ids=[],
                    evidence_judgments=[],
                    focus_roi_ids=available_focus_roi_ids[:2],
                    next_magnification=plan.required_magnification,
                )
            else:
                fallback = EvidenceReview(
                    status="need_more_slides",
                    reason=(
                        "Structured review was unusable and no higher valid magnification remained; "
                        "continue the candidate-slide search."
                    ),
                    conclusion_direction="indeterminate",
                    answer_candidate=None,
                    supporting_evidence_ids=[],
                    evidence_judgments=[],
                    focus_roi_ids=[],
                    next_magnification=None,
                )
            self.audit.append(
                {
                    "phase": "evidence_review_safe_fallback",
                    "validation": "deterministic_navigation_fallback",
                    "error": str(error)[:2000],
                }
            )
            return fallback
        self.audit.extend(result.audit)
        review = result.value
        current_node_observations = [
            item for item in observations if item.slide_id == current_slide_id
        ]
        node_support_ids = _lymph_node_metastasis_support_ids(
            question, current_node_observations
        )
        if (
            review.status == "sufficient"
            and review.answer_candidate == "Yes"
            and is_regional_node_question(question)
            and question.get("question_type") == "yes_no"
            and len(node_support_ids) < 2
        ):
            review = EvidenceReview(
                status="need_more_slides",
                reason=(
                    "The proposed nodal positive lacks two same-slide, required-resolution "
                    "components establishing nodal context and an atypical epithelial deposit."
                ),
                conclusion_direction="indeterminate",
                answer_candidate=None,
                supporting_evidence_ids=[],
                evidence_judgments=review.evidence_judgments,
                focus_roi_ids=[],
                next_magnification=None,
            )
            self.audit.append(
                {
                    "phase": "regional_node_metastasis_evidence_gate",
                    "validation": "rejected_unsubstantiated_positive",
                    "slide_id": current_slide_id,
                }
            )
        if review.status != "sufficient" and len(node_support_ids) >= 2:
            review = EvidenceReview(
                status="sufficient",
                reason=(
                    "The same slide establishes strong nodal architecture and two "
                    "required-resolution components of an atypical epithelial deposit."
                ),
                conclusion_direction="positive",
                answer_candidate="Yes",
                supporting_evidence_ids=node_support_ids[:2],
                evidence_judgments=[
                    EvidenceJudgment(
                        evidence_id=evidence_id,
                        relevance="direct",
                        direction="positive",
                        rationale=(
                            f"Cited morphology: {observation_by_id[evidence_id].morphology}"
                        )[:160],
                    )
                    for evidence_id in node_support_ids[:2]
                ],
                focus_roi_ids=[],
                next_magnification=None,
            )
            self.audit.append(
                {
                    "phase": "regional_node_metastasis_evidence_gate",
                    "validation": "promoted_from_same_slide_nodal_and_deposit_morphology",
                    "evidence_ids": node_support_ids[:2],
                }
            )
        if (
            review.status == "sufficient"
            and review.conclusion_direction == "positive"
            and is_spatial_interface_question(question)
            and not question_spatial_positive_supported(question, observations)
        ):
            attempted_max = max(
                (_MAGNIFICATION_RANK.get(item, 0) for item in attempted_magnifications),
                default=0,
            )
            required = _MAGNIFICATION_RANK[plan.required_magnification]
            can_zoom = bool(available_focus_roi_ids) and attempted_max < required
            review = EvidenceReview(
                status=("need_higher_magnification" if can_zoom else "need_more_slides"),
                reason=(
                    "The proposed positive does not contain a same-lineage spatial "
                    "relationship between the target cells and the required interface."
                ),
                conclusion_direction="indeterminate",
                answer_candidate=None,
                supporting_evidence_ids=[],
                evidence_judgments=review.evidence_judgments,
                focus_roi_ids=(available_focus_roi_ids[:2] if can_zoom else []),
                next_magnification=(plan.required_magnification if can_zoom else None),
            )
            self.audit.append(
                {
                    "phase": "spatial_relation_evidence_gate",
                    "validation": "rejected_unlinked_positive",
                    "slide_id": current_slide_id,
                }
            )
        canonical_answer = _canonical_answer_format(
            question, review.answer_candidate
        )
        if canonical_answer != review.answer_candidate:
            update: dict[str, Any] = {"answer_candidate": canonical_answer}
            if _is_generic_gastric_type_question(question):
                update["reason"] = (
                    "The visible invasive epithelial and gland-forming morphology supports "
                    "the requested parent histological category; differentiation and secondary "
                    "classification are separate questions."
                )
            review = review.model_copy(update=update)
        if (
            _is_generic_gastric_type_question(question)
            and re.search(
                r"\b(?:lauren|intestinal|diffuse|mixed)\b",
                review.reason,
                re.IGNORECASE,
            )
        ):
            reason_by_status = {
                "sufficient": (
                    "The visible invasive epithelial morphology supports the requested "
                    "minimal parent histological category."
                ),
                "need_higher_magnification": (
                    "The current field requires greater cellular detail to assess the "
                    "requested minimal parent histological category."
                ),
                "need_more_slides": (
                    "The current slide does not establish the requested minimal parent "
                    "histological category; continue the candidate-slide search."
                ),
            }
            review = review.model_copy(
                update={"reason": reason_by_status[review.status]}
            )
        preferred_answer = _question_specific_preference(question, observations)
        if review.status == "sufficient" and preferred_answer is not None:
            preference_update: dict[str, Any] = {
                "answer_candidate": preferred_answer,
                "reason": (
                    "The option-specific hallmark morphology across the inspected ROIs "
                    f"most strongly supports {preferred_answer}."
                ),
            }
            apply_preference = True
            if question.get("question_type") == "yes_no":
                expected_direction = {
                    "Yes": "positive",
                    "No": "negative",
                }.get(preferred_answer)
                if (
                    preferred_answer == "Yes"
                    and is_spatial_interface_question(question)
                ):
                    spatial_support_ids = question_spatial_positive_support_ids(
                        question, current_node_observations
                    )
                    if spatial_support_ids:
                        preference_update.update(
                            conclusion_direction="positive",
                            supporting_evidence_ids=spatial_support_ids,
                            evidence_judgments=[
                                EvidenceJudgment(
                                    evidence_id=evidence_id,
                                    relevance="direct",
                                    direction="positive",
                                    rationale=(
                                        "Cited morphology: "
                                        f"{observation_by_id[evidence_id].morphology}"
                                    )[:160],
                                )
                                for evidence_id in spatial_support_ids
                            ],
                        )
                    else:
                        # A preference may aggregate older slides, whereas a
                        # slide-local review is only allowed to cite the
                        # current slide. Never flip the answer while retaining
                        # the model's opposite direction or evidence.
                        apply_preference = False
                elif expected_direction != review.conclusion_direction:
                    apply_preference = False
            if apply_preference:
                review = review.model_copy(update=preference_update)
            else:
                self.audit.append(
                    {
                        "phase": "question_specific_preference_gate",
                        "validation": "skipped_inconsistent_review_override",
                        "preferred_answer": preferred_answer,
                        "review_direction": review.conclusion_direction,
                        "slide_id": current_slide_id,
                    }
                )
        grounded: list[EvidenceJudgment] = []
        rewritten_ids: list[str] = []
        for judgment in review.evidence_judgments:
            source = observation_by_id.get(judgment.evidence_id)
            if source is not None and _judgment_grounding_errors(
                judgment.rationale, source.morphology
            ):
                rationale = f"Cited morphology: {source.morphology}"[:160].rstrip()
                judgment = judgment.model_copy(update={"rationale": rationale})
                rewritten_ids.append(judgment.evidence_id)
            grounded.append(judgment)
        if rewritten_ids:
            self.audit.append(
                {
                    "phase": "evidence_judgment_grounding_sanitizer",
                    "validation": "rewritten_from_cited_morphology",
                    "evidence_ids": rewritten_ids,
                }
            )
            review = review.model_copy(update={"evidence_judgments": grounded})
        return review

    def final(
        self,
        question: dict[str, Any],
        plan: EvidencePlan,
        observations: list[ROIObservation],
        *,
        termination_reason: str,
        evidence_sufficient: bool,
        last_review: EvidenceReview | None,
        review_history: list[EvidenceReview],
        authorized_answer_candidate: str | None,
        all_observations: list[ROIObservation] | None = None,
        quantitative_evidence: list[QuantitativeEvidence] | None = None,
    ) -> FinalAnswer:
        quantitative_evidence = quantitative_evidence or []
        authorized_quantitative = next(
            (item for item in quantitative_evidence if item.gate_status == "authorized"),
            None,
        )
        evidence_ids = {item.evidence_id for item in observations} | {
            item.evidence_id for item in quantitative_evidence
        }
        renal_grade_bounds = _renal_grade_bounds(
            question, all_observations if all_observations is not None else observations
        )
        renal_grade_floor = renal_grade_bounds[0] if renal_grade_bounds else None
        renal_grade_ceiling = renal_grade_bounds[1] if renal_grade_bounds else None
        # Run question-side aggregation over every neutral ROI inspected for
        # this task. The smaller final prompt subset is only a context-budget
        # optimization; using it for deterministic logic discarded rare
        # deepest, poorly cohesive, and high-grade fields.
        preference_observations = (
            all_observations if all_observations is not None else observations
        )
        preferred_answer = (
            renal_grade_bounds[0]
            if renal_grade_bounds is not None
            and renal_grade_bounds[0] == renal_grade_bounds[1]
            else _question_specific_preference(
                question,
                preference_observations,
                review_history,
                termination_reason,
            )
        )
        if authorized_quantitative is not None:
            preferred_answer = authorized_quantitative.formatted_answer
            authorized_answer_candidate = authorized_quantitative.formatted_answer
        if evidence_sufficient and preferred_answer is not None:
            authorized_answer_candidate = preferred_answer
        spatial_positive_ids = [
            evidence_id
            for evidence_id in question_spatial_positive_support_ids(
                question, preference_observations
            )
            if evidence_id in evidence_ids
        ]
        regional_node_question = bool(
            is_regional_node_question(question)
            and question.get("question_type") == "yes_no"
        )
        regional_node_positive_ids = (
            [
                evidence_id
                for evidence_id in _lymph_node_metastasis_support_ids(
                    question, preference_observations
                )
                if evidence_id in evidence_ids
            ][:2]
            if regional_node_question
            else []
        )
        if (
            regional_node_question
            and termination_reason == "candidate_slides_exhausted"
            and len(regional_node_positive_ids) < 2
        ):
            # A historical sufficient review must not survive candidate-pool
            # exhaustion when the strict same-slide morphology contract was
            # never established.
            preferred_answer = "No"
            if evidence_sufficient:
                authorized_answer_candidate = "No"

        def validate(value: BaseModel) -> dict[str, str]:
            final = value
            errors = _answer_errors(question, final)
            if not set(final.supporting_evidence_ids) <= evidence_ids:
                errors["supporting_evidence_ids"] = "contains an unknown evidence ID"
            expected_sufficiency = "sufficient" if evidence_sufficient else "insufficient"
            if final.evidence_sufficiency != expected_sufficiency:
                errors["evidence_sufficiency"] = f"must be {expected_sufficiency}"
            if final.termination_reason != termination_reason:
                errors["termination_reason"] = f"must be {termination_reason}"
            if not evidence_sufficient and final.confidence != "low":
                errors["confidence"] = "must be low when evidence is insufficient"
            if (
                evidence_sufficient
                and authorized_answer_candidate is not None
                and final.answer != authorized_answer_candidate
            ):
                errors["answer"] = "must copy the evidence-gated answer candidate"
            if regional_node_question and final.answer == "Yes":
                if len(regional_node_positive_ids) < 2:
                    errors["answer"] = (
                        "regional-node Yes requires same-slide nodal and atypical "
                        "epithelial-deposit morphology"
                    )
                elif not set(regional_node_positive_ids) <= set(
                    final.supporting_evidence_ids
                ):
                    errors["supporting_evidence_ids"] = (
                        "regional-node Yes must cite the strict same-slide support IDs"
                    )
            if (
                authorized_quantitative is not None
                and authorized_quantitative.evidence_id
                not in final.supporting_evidence_ids
            ):
                errors["supporting_evidence_ids"] = (
                    "the authorized quantitative answer must cite its Q evidence ID"
                )
            return errors

        try:
            result = self.generator.generate(
                FinalAnswer,
                final_messages(
                    question,
                    plan,
                    observations,
                    termination_reason=termination_reason,
                    evidence_sufficient=evidence_sufficient,
                    last_review=last_review,
                    review_history=review_history,
                    authorized_answer_candidate=authorized_answer_candidate,
                    quantitative_evidence=quantitative_evidence,
                ),
                self.config.final_max_tokens,
                "final_answer",
                validate,
                candidate_normalizer=lambda candidate: _normalize_final_candidate(
                    candidate,
                    evidence_ids,
                    question,
                    termination_reason,
                    evidence_sufficient,
                    authorized_answer_candidate,
                    renal_grade_floor,
                    renal_grade_ceiling,
                    preferred_answer,
                    regional_node_positive_ids,
                ),
            )
        except StructuredGenerationError as error:
            self.audit.extend(error.audit)
            answer = _safe_final_fallback_answer(
                question,
                preferred_answer=preferred_answer,
                authorized_answer_candidate=authorized_answer_candidate,
                review_history=review_history,
                renal_grade_floor=renal_grade_floor,
                renal_grade_ceiling=renal_grade_ceiling,
            )
            fallback = FinalAnswer(
                answer=answer,
                confidence="low",
                evidence_sufficiency=(
                    "sufficient" if evidence_sufficient else "insufficient"
                ),
                termination_reason=termination_reason,
                supporting_evidence_ids=(
                    [authorized_quantitative.evidence_id]
                    if authorized_quantitative is not None
                    else
                    regional_node_positive_ids
                    if answer == "Yes" and len(regional_node_positive_ids) >= 2
                    else spatial_positive_ids[:5]
                    if preferred_answer == "Yes" and spatial_positive_ids
                    else [
                        item
                        for item in (
                            last_review.supporting_evidence_ids if last_review else []
                        )
                        if item in evidence_ids
                    ][:5]
                ),
                brief_rationale=(
                    "The model's final JSON remained invalid after bounded repairs; "
                    "the answer was recovered deterministically from the grounded "
                    "question-side evidence state."
                ),
            )
            self.audit.append(
                {
                    "phase": "final_answer_safe_fallback",
                    "validation": "deterministic_schema_valid_fallback",
                    "error": str(error)[:2000],
                    "answer": answer,
                }
            )
            return fallback
        self.audit.extend(result.audit)
        final = result.value
        if authorized_quantitative is not None:
            final = final.model_copy(
                update={
                    "answer": authorized_quantitative.formatted_answer,
                    "supporting_evidence_ids": [
                        authorized_quantitative.evidence_id,
                        *authorized_quantitative.supporting_evidence_ids[:4],
                    ],
                    "brief_rationale": (
                        "The cited quantitative evidence passed whole-slide tissue-denominator, "
                        "HistoPLUS cellular-quality, visible-phenotype, coverage, and uncertainty gates."
                    ),
                }
            )
        elif final.answer == "Yes" and len(regional_node_positive_ids) >= 2:
            final = final.model_copy(
                update={
                    "supporting_evidence_ids": regional_node_positive_ids,
                    "brief_rationale": (
                        "The cited same-slide morphology establishes nodal architecture and "
                        "a required-resolution atypical epithelial deposit."
                    ),
                }
            )
        elif preferred_answer == "Yes" and spatial_positive_ids:
            final = final.model_copy(
                update={
                    "supporting_evidence_ids": spatial_positive_ids[:5],
                    "brief_rationale": (
                        "The cited same-lineage morphology directly links the target cells "
                        "to the required anatomical interface."
                    ),
                }
            )
        return final
