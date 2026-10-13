"""r25 evidence consistency guards layered over r23_5 aggregation."""

from __future__ import annotations

import re
from typing import Any

from planpath.core.controller import (
    _canonical_answer_format as r23_5_canonical_answer_format,
    _gastric_invasion_layer as r23_5_gastric_invasion_layer,
    _question_specific_preference as r23_5_question_specific_preference,
    _renal_grade_bounds as r23_5_grade_bounds,
    question_specific_focus_roi_ids as r23_5_focus_roi_ids,
)
from planpath.core.schemas import ROIObservation


_GRADE_FOUR = re.compile(
    r"\b(?:rhabdoid|sarcomatoid|multinucleated\s+giant|"
    r"extreme(?:ly)?\s+pleomorph)\w*\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|without|lack\w*|absen\w*|negative\s+for|free\s+of)\b",
    re.IGNORECASE,
)
_SPINDLE_COMPONENT = re.compile(
    r"\b(?:spindle(?:[-\s]+shaped)?[-\s]+cells?|spindle[-\s]+cell|fascicles?)\b",
    re.IGNORECASE,
)
_HIGH_GRADE_SPINDLE = re.compile(
    r"\b(?:atyp\w*|pleomorph\w*|hyperchrom\w*|mitotic|necros\w*|"
    r"destructive|disorganiz\w*)\b",
    re.IGNORECASE,
)
_EPITHELIAL = re.compile(
    r"\b(?:epithelial|gland\w*|carcinoma|tumou?r|cells?|cellular\s+nests?)\b",
    re.IGNORECASE,
)
_DEEP_RELATION = re.compile(
    r"\b(?:infiltrat|inva(?:d|s)|extend|reach|breach|cross|travers|"
    r"penetrat|directly\s+abut)\w*\b",
    re.IGNORECASE,
)
_GASTRIC_DEEP_STRUCTURE = {
    "Serosa of the gastric wall": re.compile(
        r"\b(?:serosa|serosal|mesothelial|outer\s+(?:free\s+)?surface)\b",
        re.IGNORECASE,
    ),
    "Subserosal fibroadipose tissue": re.compile(
        r"\b(?:subseros\w*\s+)?(?:fat|adipose|fibroadipose|adipocytes?)\b",
        re.IGNORECASE,
    ),
    "Subserosa": re.compile(r"\bsubseros\w*\b", re.IGNORECASE),
}


def _positive_grade_four(text: str) -> bool:
    """Require a non-negated grade-four feature in one local clause."""

    for clause in re.split(r"[.;:\n]", text):
        if _GRADE_FOUR.search(clause) and not _NEGATION.search(clause):
            return True
    return False


def _remove_negated_grade_four_clauses(text: str) -> str:
    clauses = re.split(r"([.;:\n])", text)
    output: list[str] = []
    for clause in clauses:
        if _GRADE_FOUR.search(clause) and _NEGATION.search(clause):
            clause = _GRADE_FOUR.sub("grade-four feature", clause)
        output.append(clause)
    return "".join(output)


def _joint_spindle_confirmation(observations: list[ROIObservation]) -> bool:
    """Require cross-scale cellular/tissue qualification for inferred sarcomatoid foci."""

    for item in observations:
        analysis = item.cellular_analysis
        if (
            item.magnification not in {"20x", "40x"}
            or analysis is None
            or analysis.quality != "adequate"
            or analysis.instance_count < 10
            or not any(
                "Tissue Map context check" in value
                for value in analysis.limitations
            )
        ):
            continue
        clauses = re.split(r"[.;:\n]", item.morphology)
        if any(
            _SPINDLE_COMPONENT.search(clause)
            and _HIGH_GRADE_SPINDLE.search(clause)
            and not _NEGATION.search(clause)
            for clause in clauses
        ):
            return True
    return False


def renal_grade_bounds(
    question: dict[str, Any], observations: list[ROIObservation]
) -> tuple[str, str] | None:
    """Keep r23_5 grading but prevent negated features from proving Grade 4."""

    bounds = r23_5_grade_bounds(question, observations)
    if bounds != ("Grade 4", "Grade 4"):
        return bounds
    if any(
        item.magnification in {"20x", "40x"}
        and _positive_grade_four(item.morphology)
        for item in observations
    ):
        return bounds
    joint_spindle = _joint_spindle_confirmation(observations)
    sanitized = [
        item.model_copy(
            update={
                "morphology": (
                    _remove_negated_grade_four_clauses(item.morphology)
                    if joint_spindle
                    else _SPINDLE_COMPONENT.sub(
                        "mesenchymal cells",
                        _remove_negated_grade_four_clauses(item.morphology),
                    )
                )
            }
        )
        for item in observations
    ]
    return r23_5_grade_bounds(question, sanitized)


def _joint_qualified(item: ROIObservation) -> bool:
    analysis = item.cellular_analysis
    return bool(
        analysis is not None
        and analysis.quality == "adequate"
        and analysis.instance_count >= 10
        and any(
            "Tissue Map context check" in value for value in analysis.limitations
        )
    )


def _direct_deep_relation(item: ROIObservation, layer: str) -> bool:
    structure = _GASTRIC_DEEP_STRUCTURE[layer]
    for clause in re.split(r"[.;:\n]", item.morphology):
        if (
            _EPITHELIAL.search(clause)
            and structure.search(clause)
            and _DEEP_RELATION.search(clause)
            and not _NEGATION.search(clause)
        ):
            return True
    return False


def gastric_invasion_layer(
    observations: list[ROIObservation],
) -> str | None:
    """Reject an isolated unqualified deep-field claim.

    A single direct Patho-R1 relation remains sufficient when both Tissue Map
    and HistoPLUS independently qualify that same field.  Without that joint
    qualification, two independent ROI roots are required.  This is an
    evidence reliability rule: auxiliary negatives never prove absence and
    no component output is translated into a stage.
    """

    active = list(observations)
    by_roi = {item.roi_id: item for item in observations}
    joint_measurement_available = any(_joint_qualified(item) for item in observations)

    def root_id(item: ROIObservation) -> str:
        current = item
        seen: set[str] = set()
        while current.parent_roi_id and current.parent_roi_id not in seen:
            seen.add(current.parent_roi_id)
            parent = by_roi.get(current.parent_roi_id)
            if parent is None:
                return current.parent_roi_id
            current = parent
        return current.roi_id

    while True:
        layer = r23_5_gastric_invasion_layer(active)
        if layer not in _GASTRIC_DEEP_STRUCTURE:
            return layer
        direct = [
            item
            for item in active
            if item.image_quality == "adequate"
            and item.magnification in {"20x", "40x"}
            and _direct_deep_relation(item, layer)
        ]
        if any(_joint_qualified(item) for item in direct):
            return layer
        if (
            not joint_measurement_available
            and len({root_id(item) for item in direct}) >= 2
        ):
            return layer
        if not direct:
            return None
        rejected_ids = {item.evidence_id for item in direct}
        active = [item for item in active if item.evidence_id not in rejected_ids]


_FOCUS_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "special_component": (
        re.compile(
            r"\b(?:signet[-\s]?ring|intracytoplasmic\s+mucin|mucin\s+vacuol|"
            r"extracellular\s+mucin|mucin\s+(?:pool|lake)|hepatoid|squamous|"
            r"sarcomatoid|rhabdoid|spindle[-\s]+cell)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "diffuse_tumor": (
        re.compile(
            r"\b(?:diffuse|discohesive|poorly\s+cohesive|non[-\s]?cohesive|"
            r"isolated\s+(?:tumou?r\s+)?cells?|single[-\s]?cell\s+infiltrat|"
            r"without\s+gland\s+formation)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "gland_tumor": (
        re.compile(
            r"\b(?:irregular|crowded|angulated|complex|branching|haphazard)\w*"
            r".{0,55}\b(?:gland|tubul)\w*\b|"
            r"\b(?:gland|tubul)\w*.{0,55}"
            r"\b(?:irregular|crowded|angulated|complex|branching|haphazard)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "viable_tumor": (
        re.compile(
            r"\b(?:carcinoma|neoplastic|tumou?r|atypical\s+epithelial|"
            r"solid\s+nests?|alveolar|papillary)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "treatment_bed": (
        re.compile(
            r"\b(?:fibros|hyaliniz|granulation|treatment\s+bed|regression|"
            r"therapy[-\s]+related|foamy\s+histiocyte)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "immune": (
        re.compile(
            r"\b(?:lymphocyt|plasma\s+cell|macrophage|histiocyte|"
            r"inflammator|immune)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "secondary_change": (
        re.compile(
            r"\b(?:necro|hemorrhag|haemorrhag|calcif|mineraliz|cystic|debris)\w*\b",
            re.IGNORECASE,
        ),
    ),
    "collecting_system": (
        re.compile(
            r"\b(?:renal\s+pelvi|pelvicalyceal|urothel|collecting\s+system|"
            r"renal\s+sinus)\w*\b",
            re.IGNORECASE,
        ),
    ),
}


def _focus_score(item: ROIObservation, pattern: re.Pattern[str]) -> float:
    """Rank a field for another zoom without converting it into an answer."""

    if not pattern.search(item.morphology):
        return float("-inf")
    score = 2.0 if item.image_quality == "adequate" else 0.0
    score += {"5x": 0.0, "10x": 0.5, "20x": 1.0, "40x": 1.5}.get(
        item.magnification, 0.0
    )
    analysis = item.cellular_analysis
    if analysis is not None and analysis.quality != "unavailable":
        grouped = analysis.grouped_counts
        score += min(
            3.0,
            float(grouped.get("neoplastic_like", 0)) / 20.0
            + float(grouped.get("epithelial", 0)) / 30.0,
        )
        if any("Tissue Map context check" in value for value in analysis.limitations):
            score += 0.5
    return score


def _representativeness_score(item: ROIObservation) -> float:
    score = 2.0 if item.image_quality == "adequate" else 0.0
    analysis = item.cellular_analysis
    if analysis is None or analysis.quality == "unavailable":
        return score
    grouped = analysis.grouped_counts
    score += min(
        3.0,
        float(grouped.get("neoplastic_like", 0)) / 20.0
        + float(grouped.get("epithelial", 0)) / 30.0,
    )
    if any("Tissue Map context check" in value for value in analysis.limitations):
        score += 0.5
    return score


def _stratum_priority(question: dict[str, Any], route: str) -> tuple[str, ...]:
    text = str(question.get("question") or "").casefold()
    domain = str(question.get("clinical_domain") or "").casefold()
    if route == "tumor_phenotype" and "gastric" in domain:
        if "special" in text or "component" in text:
            return (
                "mucin_pool",
                "diffuse_or_poorly_cohesive_tumor",
                "gland_forming_tumor",
            )
        if "lauren" in text or "differentiation" in text:
            return (
                "diffuse_or_poorly_cohesive_tumor",
                "gland_forming_tumor",
                "mucin_pool",
            )
        return (
            "gland_forming_tumor",
            "diffuse_or_poorly_cohesive_tumor",
            "mucin_pool",
        )
    if route == "tumor_phenotype":
        return ("viable_renal_tumor", "interstitium_or_fibrosis")
    if route == "tumor_bed_response":
        return (
            "viable_renal_tumor"
            if "renal" in domain
            else "gland_forming_tumor",
            "interstitium_or_fibrosis"
            if "renal" in domain
            else "desmoplastic_stroma_or_tumor_bed",
            "immune_or_lymphoid",
        )
    if route == "tissue_composition":
        return (
            "necrosis_or_debris"
            if "renal" in domain
            else "necrosis_or_ulcer_debris",
            "viable_renal_tumor"
            if "renal" in domain
            else "gland_forming_tumor",
        )
    if route == "tumor_distribution":
        return (
            "viable_renal_tumor"
            if "renal" in domain
            else "gland_forming_tumor",
        )
    return ()


def question_specific_focus_roi_ids(
    question: dict[str, Any],
    observations: list[ROIObservation],
    current_slide_id: str,
    available_roi_ids: list[str],
) -> list[str]:
    """Extend r23_5 focus selection to all r25 component-bearing routes.

    Tissue Map makes heterogeneous hypotheses available at 5x.  This helper
    carries morphology from at most one field per hypothesis into the next
    zoom.  At 20x, HistoPLUS counts break ties only between already relevant
    morphology fields; they never establish a diagnosis or an absence.
    """

    base = r23_5_focus_roi_ids(
        question,
        observations,
        current_slide_id,
        available_roi_ids,
    )
    from .context import roi_stratum
    from .routing import component_route

    route = component_route()
    strata_by_route = {
        "tumor_phenotype": (
            "special_component",
            "diffuse_tumor",
            "gland_tumor",
            "viable_tumor",
        ),
        "tumor_bed_response": ("viable_tumor", "treatment_bed", "immune"),
        "tissue_composition": ("secondary_change", "viable_tumor"),
        "tumor_distribution": ("viable_tumor",),
        "collecting_system_interface": ("collecting_system", "viable_tumor"),
    }
    strata = strata_by_route.get(route)
    if strata is None:
        return base
    available = set(available_roi_ids)
    candidates = [
        item
        for item in observations
        if item.slide_id == current_slide_id and item.roi_id in available
    ]
    selected: list[str] = []

    # Carry one ROI from each question-relevant Tissue Map hypothesis through
    # the unchanged two-parent zoom contract. HistoPLUS measurements, when
    # already available at 20x, select the most representative member within
    # a stratum; they do not alter the stratum's diagnostic meaning.
    for stratum in _stratum_priority(question, route):
        ranked = sorted(
            (
                (_representativeness_score(item), -index, item.roi_id)
                for index, item in enumerate(candidates)
                if roi_stratum(item.roi_id) == stratum
            ),
            reverse=True,
        )
        if ranked and ranked[0][2] not in selected:
            selected.append(ranked[0][2])

    selected.extend(roi_id for roi_id in base if roi_id not in selected)
    overall: list[tuple[float, int, str]] = []
    for stratum in strata:
        patterns = _FOCUS_PATTERNS[stratum]
        ranked: list[tuple[float, int, str]] = []
        for index, item in enumerate(candidates):
            score = max(_focus_score(item, pattern) for pattern in patterns)
            if score == float("-inf"):
                continue
            value = (score, -index, item.roi_id)
            ranked.append(value)
            overall.append(value)
        ranked.sort(reverse=True)
        if ranked and ranked[0][2] not in selected:
            selected.append(ranked[0][2])
    overall.sort(reverse=True)
    selected.extend(
        roi_id
        for _, _, roi_id in overall
        if roi_id not in selected
    )
    return selected


def canonical_answer_format(question: dict[str, Any], value: Any) -> Any:
    """Canonicalize special-component nomenclature, without changing polarity."""

    answer = r23_5_canonical_answer_format(question, value)
    text = str(question.get("question") or "").casefold()
    if "special histological component" not in text:
        return answer
    raw = str(answer or "").strip()
    lowered = raw.casefold()
    components: list[tuple[int, str]] = []
    signet = re.search(r"\bsignet[-\s]?ring\b", lowered)
    if signet is not None:
        components.append((signet.start(), "Signet-ring cell carcinoma"))
    mucinous = re.search(r"\bmucinous\b", lowered)
    if mucinous is not None:
        label = (
            "Poorly differentiated mucinous adenocarcinoma"
            if re.search(r"\bpoorly\s+differentiated\b", lowered)
            else "Mucinous adenocarcinoma"
        )
        components.append((mucinous.start(), label))
    hepatoid = re.search(r"\bhepatoid\b", lowered)
    if hepatoid is not None:
        components.append((hepatoid.start(), "Hepatoid adenocarcinoma"))
    if not components:
        return answer
    components.sort()
    return "; ".join(dict.fromkeys(label for _, label in components))


def _gastric_special_component_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
) -> str | None:
    """Name a signet-ring component only from its same-field morphology.

    The Tissue Map lineage is a localization qualification, not a diagnosis:
    the linked 40x Patho-R1 transcription must independently describe an
    isolated epithelial cell with a dominant cytoplasmic vacuole and an
    eccentrically displaced nucleus. This removes final-LLM wording variance
    without turning a map class or HistoPLUS cell label into an answer.
    """

    text = str(question.get("question") or "").casefold()
    domain = str(question.get("clinical_domain") or "").casefold()
    if "gastric" not in domain or "special histological component" not in text:
        return None
    from .context import roi_stratum

    epithelial = re.compile(r"\b(?:epithelial|signet[-\s]?ring)\s+cell\b", re.I)
    vacuole = re.compile(
        r"\b(?:large|dominant|prominent|marked\w*)\b.{0,45}"
        r"\b(?:cytoplasmic\s+)?vacuol\w*\b|"
        r"\b(?:cytoplasmic\s+)?vacuol\w*\b.{0,45}"
        r"\b(?:occup\w*\s+most|dominant|large|marked\w*)\b",
        re.I,
    )
    eccentric = re.compile(
        r"\b(?:eccentric\w*|peripherally)\s+(?:displac\w*|located)\b",
        re.I,
    )
    isolated = re.compile(
        r"\b(?:single|isolated|discohesive|poorly\s+cohesive)\b",
        re.I,
    )
    for item in observations:
        if (
            item.image_quality != "adequate"
            or item.magnification != "40x"
            or roi_stratum(item.roi_id)
            not in {"mucin_pool", "diffuse_or_poorly_cohesive_tumor"}
        ):
            continue
        morphology = item.morphology
        if (
            epithelial.search(morphology)
            and vacuole.search(morphology)
            and eccentric.search(morphology)
            and isolated.search(morphology)
            and not re.search(
                r"\b(?:no|without|absen\w*|lack\w*)\b.{0,35}\bvacuol\w*\b",
                morphology,
                re.IGNORECASE,
            )
        ):
            return "Signet-ring cell carcinoma"
    return None


def question_specific_preference(
    question: dict[str, Any],
    observations: list[ROIObservation],
    reviews: list[Any] | None = None,
    termination_reason: str | None = None,
) -> str | None:
    return _gastric_special_component_preference(question, observations) or (
        r23_5_question_specific_preference(
            question,
            observations,
            reviews,
            termination_reason,
        )
    )
