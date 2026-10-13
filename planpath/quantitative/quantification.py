"""Deterministic, auditable quantitative evidence for PlanPath v7.

The estimator deliberately separates three jobs:

* the tissue map defines the whole-slide sampling frame and area denominator;
* Patho-R1 supplies a question-independent visible phenotype anchor;
* HistoPLUS supplies cell counts and valid analyzed area inside sampled ROIs.

No reference answer, report text, specimen filename, or site label is accepted
by this module.  A component estimate is therefore unavailable when either of
the two quantitative tools is ablated instead of silently degrading into a
plausible-looking number.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import AgentConfig
from .schemas import EvidencePlan, QuantitativeEvidence, ROIObservation
from .tissue_map import TissueMap


GASTRIC_TUMOR_STRATA = (
    "gland_forming_tumor",
    "diffuse_or_poorly_cohesive_tumor",
    "mucin_pool",
)
RENAL_TUMOR_STRATA = ("viable_renal_tumor",)


@dataclass(frozen=True)
class QuantificationProfile:
    kind: str
    target: str
    phenotype: str


def quantitative_profile(
    question: dict[str, Any], plan: EvidencePlan | None = None
) -> QuantificationProfile | None:
    """Classify only explicit numerical questions; semantics alone is unsafe.

    pN is represented as ``quantitative`` in the legacy plan but has its own
    validated involved-node aggregation.  Requiring percentage/count wording
    here keeps that and every unrelated v6 path byte-for-byte behaviorally
    isolated from the new component estimator.
    """

    text = str(question.get("question") or "").casefold()
    proportion = any(
        term in text
        for term in ("proportion", "percentage", "percent", "fraction", "ratio")
    )
    if proportion and "primary tumor" in text:
        if "signet" in text:
            phenotype = "signet_ring"
            target = "signet-ring cell component"
        elif "mucin" in text:
            phenotype = "mucinous"
            target = "mucinous component"
        elif "hepatoid" in text:
            phenotype = "hepatoid"
            target = "hepatoid component"
        else:
            phenotype = "other_component"
            target = "requested primary-tumor component"
        return QuantificationProfile("component_fraction", target, phenotype)
    if any(term in text for term in ("how many", "number of", "count of")) and any(
        term in text for term in ("tumor foci", "tumour foci", "tumor focus", "tumour focus")
    ):
        return QuantificationProfile("focus_count", "independent primary-tumor foci", "tumor")
    if any(term in text for term in ("cells per", "cell density", "cells/mm", "cells per mm")):
        return QuantificationProfile("cell_density", "requested cell population", "cell_population")
    if any(term in text for term in ("distance", "millimeter", "millimetre", " mm")):
        return QuantificationProfile("spatial_distance", "requested spatial distance", "distance")
    return None


def is_component_fraction_question(
    question: dict[str, Any], plan: EvidencePlan | None = None
) -> bool:
    profile = quantitative_profile(question, plan)
    return bool(profile and profile.kind == "component_fraction")


def is_component_fraction_plan(plan: EvidencePlan | None) -> bool:
    if plan is None:
        return False
    text = " ".join(
        [plan.target_finding, *plan.question_attributes, plan.evidence_goal]
    ).casefold()
    return "component proportion quantification" in text


def is_focus_count_plan(plan: EvidencePlan | None) -> bool:
    if plan is None:
        return False
    text = " ".join(
        [plan.target_finding, *plan.question_attributes, plan.evidence_goal]
    ).casefold()
    return "multifocal topology quantification" in text


def is_quantitative_measurement_plan(plan: EvidencePlan | None) -> bool:
    return is_component_fraction_plan(plan) or is_focus_count_plan(plan)


def _feature_present(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _affirmed_segments(
    text: str, *, split_negative_subclauses: bool = False
) -> list[str]:
    """Return local clauses that do not negate the described feature.

    Quantitative observations deliberately mention both present and absent
    inventory items.  Searching the entire transcription therefore turns
    phrases such as ``no extracellular mucin`` into false positives.  Segment
    first and discard explicitly negative clauses before feature matching.
    """

    # Commas often separate a positive phenotype from an unrelated negative
    # statement (for example, "single-file cells, with no stromal invasion").
    # Keeping that as one segment discarded the true phenotype merely because
    # another feature in the sentence was negated.
    separators = r"[.;]|\bbut\b|\bhowever\b"
    if split_negative_subclauses:
        separators += r"|,(?=\s*(?:with\s+)?(?:no|not|without|neither|absent|lack))"
    pieces = re.split(separators, text.casefold())
    negation = re.compile(
        r"\b(?:no|not|without|neither|absent|absence|lack(?:s|ed|ing)?)\b",
        re.IGNORECASE,
    )
    return [piece.strip() for piece in pieces if piece.strip() and not negation.search(piece)]


def _explicit_mucin_fraction(text: str) -> float | None:
    """Extract a field-area fraction only from an affirmed mucin clause."""

    for segment in _affirmed_segments(text):
        if "mucin" not in segment:
            continue
        match = re.search(
            r"(?:occup\w*|ratio[^.;]{0,35}occup\w*)\s*"
            r"(?:approximately\s*)?(?P<lower>>\s*)?(?P<value>\d{1,3})\s*%",
            segment,
            re.IGNORECASE,
        )
        if match is None:
            continue
        value = min(100, int(match.group("value"))) / 100.0
        # A stated lower bound such as >50% is represented by the midpoint of
        # the next 10-point reporting bin, rather than as exactly 50%.
        if match.group("lower"):
            value = min(1.0, value + 0.05)
        return value
    return None


def _phenotype_score(phenotype: str, morphology: str) -> float:
    """Convert affirmed visible morphology into a bounded field fraction."""

    # Signet-ring descriptions commonly put an unrelated negative inventory
    # after a comma ("single-file cells, with no stromal invasion").  Use the
    # narrower subclause split only for this phenotype so established mucinous
    # and hepatoid scoring remains unchanged.
    segments = _affirmed_segments(
        morphology, split_negative_subclauses=phenotype == "signet_ring"
    )
    text = " . ".join(segments)
    if phenotype == "signet_ring":
        explicit = _feature_present(text, (r"\bsignet[-\s]?ring\b",))
        vacuolated = _feature_present(
            text,
            (
                r"intracytoplasmic\s+(?:mucin|vacuol\w*)",
                r"(?:mucin|vacuol\w*)[^.;]{0,70}(?:displac\w*|eccentric|peripheral)\s+nucle\w*",
                r"(?:displac\w*|eccentric|peripheral)\s+nucle\w*[^.;]{0,70}(?:mucin|vacuol\w*)",
            ),
        )
        discohesive = _feature_present(
            text,
            (
                r"\b(?:discohesive|poorly\s+cohesive|single[-\s]cell|single[-\s]file|"
                r"single\s+epithelial\s+cell|isolated(?:\s+epithelial)?\s+cells?)\b",
            ),
        )
        positive = explicit or (vacuolated and discohesive)
        extent_text = " ".join(
            item
            for item in segments
            if _feature_present(
                item,
                (
                    r"\bsignet[-\s]?ring\b",
                    r"\bdiscohesive\b",
                    r"\bsingle[-\s]cell\b",
                    r"\bsingle[-\s]file\b",
                    r"\bsingle\s+epithelial\s+cell\b",
                    r"\bisolated\b",
                    r"\bintracytoplasmic\b",
                    r"\beccentric\w*\b",
                    r"\bdisplac\w*\b",
                ),
            )
        )
    elif phenotype == "mucinous":
        extracellular = _feature_present(
            text,
            (
                r"\b(?:extracellular\s+)?mucin(?:ous)?\s+(?:pool|lake|material|stroma)",
                r"\bmucin[-\s]rich\s+(?:stroma|pool|material)",
                r"\bextracellular\s+(?:pale|myxoid)\s+material\b",
            ),
        )
        epithelial = _feature_present(
            text,
            (
                r"\b(?:epithelial|glandular)\s+"
                r"(?:cells?|groups?|clusters?|glands?|proliferation|structures?)\b",
            ),
        )
        positive = extracellular and epithelial
        explicit_fraction = _explicit_mucin_fraction(morphology)
        extent_text = " ".join(item for item in segments if "mucin" in item)
    elif phenotype == "hepatoid":
        architecture = _feature_present(
            text,
            (
                r"\b(?:trabecul\w*|solid\s+(?:sheets?|nests?|patterns?)|cords?)\b",
            ),
        )
        cytoplasm = _feature_present(
            text,
            (
                r"\bpolygonal\s+cells?\b",
                r"\b(?:abundant\s+)?(?:eosinophilic|granular)"
                r"(?:\s+and\s+(?:eosinophilic|granular))?\s+cytoplasm\b",
            ),
        )
        positive = architecture and cytoplasm
        extent_text = text
    else:
        positive = False
        extent_text = ""
    if not positive:
        return 0.0
    if phenotype == "mucinous" and explicit_fraction is not None:
        return explicit_fraction
    # Hepatoid is aggregated as the share of HistoPLUS-qualified fields that
    # contain its defining architecture.  Adjectives such as ``abundant``
    # modify cytoplasm and must not be misread as a component percentage.
    if phenotype == "hepatoid":
        return 1.0
    # For cellular signet-ring morphology and mucin area without an explicit
    # percentage, local qualifiers estimate the within-field share.
    if phenotype == "signet_ring" and _feature_present(
        extent_text, (r"\b(?:rare|scattered|isolated|occasional)\b",)
    ):
        return 0.10
    if _feature_present(extent_text, (r"\b(?:small\s+focus|focal|minor|limited)\b",)):
        return 0.25
    if _feature_present(extent_text, (r"\b(?:diffuse|predominant|extensive|numerous)\b",)):
        return 0.80
    if phenotype == "mucinous" and _feature_present(
        extent_text, (r"\b(?:broad|abundant)\b",)
    ):
        return 0.60
    return 0.50


def _root_roi_id(item: ROIObservation, by_roi: dict[str, ROIObservation]) -> str:
    current = item
    seen: set[str] = set()
    while current.parent_roi_id and current.parent_roi_id not in seen:
        seen.add(current.parent_roi_id)
        parent = by_roi.get(current.parent_roi_id)
        if parent is None:
            break
        current = parent
    return current.roi_id


def _cellular_lineages(
    observations: list[ROIObservation], *, include_low_power: bool = False
) -> list[list[ROIObservation]]:
    """Group repeated magnifications from one low-power sampling unit.

    Component-area phenotypes such as extracellular mucin are often most
    faithfully visible in the linked 5x parent, while HistoPLUS measurement is
    available on its 20x/40x descendants.  ``include_low_power`` joins that
    parent only after a measured descendant establishes the cellular lineage.
    """

    by_roi = {item.roi_id: item for item in observations}
    groups: dict[tuple[str, str], list[ROIObservation]] = {}
    for item in observations:
        if item.cellular_analysis is None or item.magnification not in {"20x", "40x"}:
            continue
        key = (item.slide_id, _root_roi_id(item, by_roi))
        groups.setdefault(key, []).append(item)
    if include_low_power:
        for item in observations:
            if item.magnification != "5x":
                continue
            key = (item.slide_id, _root_roi_id(item, by_roi))
            if key in groups:
                groups[key].append(item)
    return list(groups.values())


def _calibrated_signet_core_fraction(
    robust_anchors: list[tuple[float, int]],
    *,
    core_size: int,
    enrichment_anchor_cells: list[int] | None = None,
) -> float:
    """Calibrate signet-ring extent from independent, cell-qualified anchors.

    Counts alone are unstable because the same rare-cell phenotype can appear
    in several independently sampled fields. Weak ``rare``/``focal`` mentions
    therefore remain minor-component evidence even when repeated. One explicit
    high-occupancy signet-ring field is stronger, while only a reproducible
    high-occupancy phenotype may imply a dominant component.
    """

    if not robust_anchors:
        return 0.0
    strong_anchor_present = any(
        score >= 0.50 and tumor_cells >= 20
        for score, tumor_cells in robust_anchors
    )
    if not strong_anchor_present:
        # Tissue-enriched sections sit outside the fixed TITAN denominator and
        # therefore cannot freely inflate the estimate. They may add one
        # reporting bin only when a core detection is independently reproduced
        # with substantial HistoPLUS epithelial/neoplastic-like cell support.
        return (
            0.30
            if any(cells >= 20 for cells in (enrichment_anchor_cells or []))
            else 0.20
        )
    fraction = max(0.40, len(robust_anchors) / max(core_size, 1))
    if len(robust_anchors) >= 3:
        fraction *= 1.50
    return min(1.0, fraction)


def _recurrent_mucin_component_floor(
    slide_class_fractions: dict[str, dict[str, float]],
    robust_anchor_scores: list[float],
    *,
    low_cell_anchor_slide_count: int,
) -> float:
    """Authorize a substantial mucin component only from recurrent dual evidence."""

    if not slide_class_fractions or not any(
        score >= 0.60 for score in robust_anchor_scores
    ):
        return 0.0
    recurrent_fraction = statistics.median(
        item.get("mucin_pool", 0.0) for item in slide_class_fractions.values()
    )
    if recurrent_fraction < 0.03:
        return 0.0
    # Recurrent map signal plus a robust local anchor establishes at least a
    # substantial component. Widespread hypocellular mucin on two or more core
    # sections is required before promoting it to a majority component.
    return 0.60 if low_cell_anchor_slide_count >= 2 else 0.30


def _tissue_frame(
    slides: list[dict[str, Any]], cancer: str
) -> tuple[
    dict[str, float],
    float,
    float,
    set[str],
    dict[str, dict[str, float]],
    list[str],
]:
    strata = RENAL_TUMOR_STRATA if cancer == "renal" else GASTRIC_TUMOR_STRATA
    class_areas = {name: 0.0 for name in strata}
    expected = len(slides)
    available = 0
    eligible_slide_ids: set[str] = set()
    known_area = 0.0
    total_area = 0.0
    slide_class_fractions: dict[str, dict[str, float]] = {}
    limitations: list[str] = []
    for slide in slides:
        raw = str(slide.get("tissue_map_cache") or "")
        if not raw or not Path(raw).is_file():
            continue
        try:
            # Whole-map aggregation does not need the navigation-only exterior
            # flood fill, avoiding a large raster allocation per slide.
            summary = TissueMap(Path(raw), build_exterior_topology=False).global_summary()
        except Exception as error:
            limitations.append(
                f"{slide.get('slide_id', 'unknown')}: invalid tissue map ({type(error).__name__})"
            )
            continue
        if not summary["available"]:
            continue
        available += 1
        known_area += float(summary["known_area_px2"])
        total_area += float(summary["total_area_px2"])
        slide_tumor_area = sum(
            float(summary["class_areas_px2"].get(name, 0.0)) for name in strata
        )
        # Exclude candidate sections whose map contains only soft classifier
        # leakage rather than a meaningful primary-tumor sampling frame.
        if slide_tumor_area / max(float(summary["known_area_px2"]), 1e-12) >= 0.02:
            slide_id = str(slide.get("slide_id") or "")
            eligible_slide_ids.add(slide_id)
            slide_class_fractions[slide_id] = {
                name: float(summary["class_areas_px2"].get(name, 0.0))
                / max(slide_tumor_area, 1e-12)
                for name in strata
            }
            for name in strata:
                class_areas[name] += float(
                    summary["class_areas_px2"].get(name, 0.0)
                )
    map_fraction = available / max(expected, 1)
    known_fraction = known_area / max(total_area, 1e-12)
    coverage = map_fraction * known_fraction
    return (
        class_areas,
        coverage,
        known_area,
        eligible_slide_ids,
        slide_class_fractions,
        limitations,
    )


def _insufficient(
    profile: QuantificationProfile,
    *,
    reasons: list[str],
    tissue_coverage: float,
    eligible_slides: int,
    analyzed_slides: int = 0,
    sampled_rois: int = 0,
    adequate_rois: int = 0,
    stratum_coverage: float = 0.0,
    support_ids: list[str] | None = None,
    limitations: list[str] | None = None,
) -> QuantitativeEvidence:
    unit = {
        "component_fraction": "percent",
        "cell_density": "cells_per_mm2",
        "focus_count": "count",
        "spatial_distance": "millimeter",
    }[profile.kind]
    return QuantitativeEvidence(
        evidence_id="Q0001",
        task_kind=profile.kind,  # type: ignore[arg-type]
        target=profile.target,
        unit=unit,  # type: ignore[arg-type]
        confidence="unavailable",
        sampling_method="tissue-stratified fixed-budget ROI sampling",
        tissue_coverage_fraction=max(0.0, min(1.0, tissue_coverage)),
        stratum_coverage_fraction=max(0.0, min(1.0, stratum_coverage)),
        eligible_slide_count=eligible_slides,
        analyzed_slide_count=analyzed_slides,
        sampled_roi_count=sampled_rois,
        adequate_cellular_roi_count=adequate_rois,
        component_roles={
            "Tissue Navigation": "whole-slide sampling frame and tumor-area denominator",
            "HistoPLUS": "confidence-qualified cell counts and valid ROI area",
            "Patho-R1": "visible same-ROI phenotype anchor",
            "Python Evidence Gate": "coverage, unit, lineage, and uncertainty validation",
        },
        supporting_evidence_ids=list(dict.fromkeys(support_ids or []))[:20],
        gate_status="insufficient",
        gate_reasons=list(dict.fromkeys(reasons)),
        limitations=(limitations or [])[:8],
    )


def build_quantitative_evidence(
    question: dict[str, Any],
    plan: EvidencePlan,
    slides: list[dict[str, Any]],
    observations: list[ROIObservation],
    config: AgentConfig,
) -> list[QuantitativeEvidence]:
    """Build Q evidence after the visual candidate pool has been inspected."""

    profile = quantitative_profile(question, plan)
    if not config.quantification_enabled or profile is None:
        return []
    if profile.kind == "focus_count":
        reasons: list[str] = []
        mapped_slide_count = sum(
            bool(str(slide.get("tissue_map_cache") or ""))
            and Path(str(slide.get("tissue_map_cache"))).is_file()
            for slide in slides
        )
        tissue_coverage = mapped_slide_count / max(len(slides), 1)
        if not config.tissue_navigation_enabled:
            reasons.append("Tissue Navigation is disabled; separated focus candidates are unavailable")
        if not config.histoplus_enabled:
            reasons.append("HistoPLUS is disabled; candidate foci cannot receive cellular confirmation")
        if tissue_coverage < config.quantification_min_tissue_coverage:
            reasons.append(
                "tissue-map slide coverage is below "
                f"{config.quantification_min_tissue_coverage:.0%}"
            )
        lineages = _cellular_lineages(observations)
        confirmed_by_slide: dict[str, dict[str, str]] = {}
        analyzed_slides: set[str] = set()
        adequate_rois = 0
        for lineage in lineages:
            measured = max(
                lineage,
                key=lambda item: item.cellular_analysis.instance_count,  # type: ignore[union-attr]
            )
            analysis = measured.cellular_analysis
            navigation = measured.tissue_navigation
            if analysis is None or navigation is None or analysis.instance_count < 10:
                continue
            match = re.search(r"focus_component_(\d+)", navigation.selection_route)
            if match is None:
                continue
            tumor_cells = int(analysis.grouped_counts.get("neoplastic_like", 0)) + int(
                analysis.grouped_counts.get("epithelial", 0)
            )
            local_tumor_fraction = float(
                navigation.class_fractions.get("viable_renal_tumor", 0.0)
            )
            morphologic_anchor = next(
                (
                    item
                    for item in lineage
                    if _feature_present(
                        item.morphology,
                        (
                            r"\b(?:epithelial|cellular)\s+proliferation\b",
                            r"\b(?:solid|nested|alveolar)\s+(?:nests?|architecture|groups?)\b",
                            r"\bcohesive\s+(?:cells?|clusters?|nests?)\b",
                        ),
                    )
                ),
                None,
            )
            if (
                tumor_cells < 10
                or morphologic_anchor is None
                or local_tumor_fraction
                < config.quantification_min_focus_local_fraction
            ):
                continue
            adequate_rois += 1
            analyzed_slides.add(measured.slide_id)
            confirmed_by_slide.setdefault(measured.slide_id, {})[
                match.group(1)
            ] = morphologic_anchor.evidence_id
        strongest_slide = max(
            confirmed_by_slide,
            key=lambda slide_id: len(confirmed_by_slide[slide_id]),
            default=None,
        )
        count = (
            len(confirmed_by_slide[strongest_slide])
            if strongest_slide is not None
            else 0
        )
        support_ids = (
            list(confirmed_by_slide[strongest_slide].values())
            if strongest_slide is not None
            else []
        )
        if count < 1:
            reasons.append("no spatially distinct Tissue candidates passed cellular and morphology checks")
        if reasons:
            return [
                _insufficient(
                    profile,
                    reasons=reasons,
                    tissue_coverage=(
                        tissue_coverage if config.tissue_navigation_enabled else 0.0
                    ),
                    eligible_slides=len(slides),
                    analyzed_slides=len(analyzed_slides),
                    sampled_rois=len(lineages),
                    adequate_rois=adequate_rois,
                    support_ids=support_ids,
                    limitations=[
                        "Separated 2-D components are treated only as candidates until both cellular and morphology checks pass."
                    ],
                )
            ]
        words = {1: "One focus", 2: "Two foci", 3: "Three foci", 4: "Four foci"}
        answer = words.get(count, f"{count} foci")
        return [
            QuantitativeEvidence(
                evidence_id="Q0001",
                task_kind="focus_count",
                target=profile.target,
                estimate=count,
                unit="count",
                formatted_answer=answer,
                interval_95=(float(count), float(count)),
                confidence="medium",
                sampling_method=(
                    "whole-slide connected Tissue candidates with same-lineage "
                    "HistoPLUS cell and Patho-R1 morphology confirmation"
                ),
                tissue_coverage_fraction=tissue_coverage,
                stratum_coverage_fraction=1.0,
                eligible_slide_count=len(slides),
                analyzed_slide_count=len(analyzed_slides),
                sampled_roi_count=len(lineages),
                adequate_cellular_roi_count=adequate_rois,
                numerator=float(count),
                denominator=1.0,
                denominator_unit="patient specimen",
                component_roles={
                    "Tissue Navigation": "spatially separated whole-slide focus candidates",
                    "HistoPLUS": "confidence-qualified epithelial/neoplastic-like cell confirmation",
                    "Patho-R1": "same-ROI visible cellular-proliferation anchor",
                    "Python Evidence Gate": "component separation and dual-confirmation validation",
                },
                supporting_evidence_ids=support_ids[:20],
                gate_status="authorized",
                gate_reasons=[],
                limitations=[
                    "The estimate is the maximum independently confirmed 2-D focus count on one section; serial sections are not summed."
                ],
            )
        ]
    if profile.kind != "component_fraction":
        return [
            _insufficient(
                profile,
                reasons=[
                    "this measurement requires a specimen-level topology or calibrated distance strategy not yet available"
                ],
                tissue_coverage=0.0,
                eligible_slides=len(slides),
                limitations=[
                    "The gate refuses to infer gross multifocality or distance from disconnected 2-D ROI samples."
                ],
            )
        ]

    cancer = str(slides[0].get("cancer") or "unknown") if slides else "unknown"
    (
        class_areas,
        tissue_coverage,
        known_area,
        eligible_tumor_slide_ids,
        slide_class_fractions,
        limitations,
    ) = _tissue_frame(
        slides, cancer
    )
    strata = RENAL_TUMOR_STRATA if cancer == "renal" else GASTRIC_TUMOR_STRATA
    tumor_area = sum(class_areas.values())
    reasons: list[str] = []
    if not config.tissue_navigation_enabled:
        reasons.append("Tissue Navigation is disabled; the whole-slide denominator is unavailable")
    if not config.histoplus_enabled:
        reasons.append("HistoPLUS is disabled; confidence-qualified cell measurements are unavailable")
    if tissue_coverage < config.quantification_min_tissue_coverage:
        reasons.append(
            "tissue-map coverage is below "
            f"{config.quantification_min_tissue_coverage:.0%}"
        )
    if tumor_area <= 0:
        reasons.append("no measurable primary-tumor denominator was found in the tissue maps")

    cellular_lineages = _cellular_lineages(observations, include_low_power=True)
    analyzed_slide_ids: set[str] = set()
    adequate: list[tuple[ROIObservation, int, float, str, float, str]] = []
    phenotype_support_ids: list[str] = []
    explicit_signet_lineage_ids: set[str] = set()
    for lineage in cellular_lineages:
        # Use the largest confidence-qualified cellular sample (normally 20x)
        # for weighting and the strongest visible phenotype at any linked
        # magnification (normally 40x) for identity. This preserves physical
        # sampling independence without throwing away either tool's best scale.
        measured = max(
            (item for item in lineage if item.cellular_analysis is not None),
            key=lambda item: (
                item.cellular_analysis.quality == "adequate",  # type: ignore[union-attr]
                item.cellular_analysis.instance_count,  # type: ignore[union-attr]
            ),
        )
        analysis = measured.cellular_analysis
        if analysis is None or analysis.quality == "unavailable":
            continue
        neoplastic = int(analysis.grouped_counts.get("neoplastic_like", 0))
        epithelial = int(analysis.grouped_counts.get("epithelial", 0))
        tumor_cells = neoplastic + epithelial
        phenotype_item = max(
            lineage,
            key=lambda item: _phenotype_score(profile.phenotype, item.morphology),
        )
        score = _phenotype_score(profile.phenotype, phenotype_item.morphology)
        # Extracellular mucin pools are intrinsically hypocellular, and the
        # generic HistoPLUS taxonomy may assign their few entrapped epithelial
        # cells to another class. In this narrow case HistoPLUS validates that
        # the ROI is analyzable (instance count and valid area), while Patho-R1
        # supplies the explicit epithelial-in-mucin identity. Other phenotypes
        # still require epithelial/neoplastic-like cell labels.
        low_cell_mucin_anchor = profile.phenotype == "mucinous" and score > 0
        if analysis.instance_count < 10 or (
            tumor_cells <= 0 and not low_cell_mucin_anchor
        ):
            continue
        navigation = measured.tissue_navigation or phenotype_item.tissue_navigation
        local_fractions = navigation.class_fractions if navigation is not None else {}
        stratum = max(
            strata,
            key=lambda name: float(local_fractions.get(name, 0.0)),
        )
        mapped_tumor_fraction = max(
            float(local_fractions.get(name, 0.0)) for name in strata
        )
        if mapped_tumor_fraction < 0.10 and score <= 0:
            # Without a mapped stratum this ROI cannot contribute to a
            # denominator. A positive same-ROI phenotype with HistoPLUS-
            # confirmed epithelial cells is retained because extracellular
            # mucin and unusual cytoplasm are common tissue-map failure modes.
            continue
        if mapped_tumor_fraction < 0.10:
            stratum = {
                "mucinous": "mucin_pool",
                "signet_ring": "diffuse_or_poorly_cohesive_tumor",
            }.get(profile.phenotype, strata[0])
        adequate.append(
            (
                measured,
                tumor_cells,
                score,
                stratum,
                float(analysis.valid_area_mm2),
                navigation.selection_route if navigation is not None else "",
            )
        )
        analyzed_slide_ids.add(measured.slide_id)
        if score > 0:
            phenotype_support_ids.append(phenotype_item.evidence_id)
            if profile.phenotype == "signet_ring" and _feature_present(
                " ".join(
                    _affirmed_segments(
                        phenotype_item.morphology,
                        split_negative_subclauses=True,
                    )
                ),
                (r"\bsignet[-\s]?ring\b",),
            ):
                explicit_signet_lineage_ids.add(measured.roi_id)

    total_tumor_cells = sum(item[1] for item in adequate)
    if len(adequate) < config.quantification_min_cellular_rois:
        reasons.append(
            "fewer than "
            f"{config.quantification_min_cellular_rois} independent HistoPLUS ROIs passed quality control"
        )
    if total_tumor_cells < config.quantification_min_tumor_cells:
        reasons.append(
            "confidence-qualified tumor-cell count is below "
            f"{config.quantification_min_tumor_cells}"
        )
    if not phenotype_support_ids:
        reasons.append("no same-ROI visible morphology anchored the requested phenotype")
    cellular_slide_coverage = len(
        analyzed_slide_ids & eligible_tumor_slide_ids
    ) / max(len(eligible_tumor_slide_ids), 1)
    if cellular_slide_coverage < 0.60:
        reasons.append("fewer than 60% of mapped tumor-bearing slides have usable cellular samples")

    global_weights = {
        name: class_areas.get(name, 0.0) / max(tumor_area, 1e-12) for name in strata
    }
    measured_strata = {item[3] for item in adequate}
    stratum_coverage = sum(global_weights.get(name, 0.0) for name in measured_strata)
    if stratum_coverage < config.quantification_min_stratum_coverage:
        reasons.append(
            "cellular sampling covers less than "
            f"{config.quantification_min_stratum_coverage:.0%} of mapped tumor strata"
        )

    if reasons:
        return [
            _insufficient(
                profile,
                reasons=reasons,
                tissue_coverage=tissue_coverage,
                eligible_slides=len(eligible_tumor_slide_ids),
                analyzed_slides=len(analyzed_slide_ids),
                sampled_rois=len(cellular_lineages),
                adequate_rois=len(adequate),
                stratum_coverage=stratum_coverage,
                support_ids=phenotype_support_ids,
                limitations=limitations,
            )
        ]

    # Estimate a field-area share within each slide, then give each analyzed
    # slide one vote. This prevents one very cellular slide from dominating a
    # specimen estimate and avoids the inverse-cellularity bias of mucin pools.
    # HistoPLUS contributes the valid analyzed area and excludes unusable
    # fields; target-anchor fields are used for identity but systematic fields
    # are preferred for mucin prevalence to limit enrichment bias.
    prevalence_pool = adequate
    if (
        config.quantification_estimator_mode == "continuous_measurement"
        or profile.phenotype == "mucinous"
    ):
        systematic = [
            item for item in adequate if "quantification_systematic" in item[5]
        ]
        if len(systematic) >= config.quantification_min_cellular_rois:
            prevalence_pool = systematic
    by_slide: dict[str, list[tuple[float, float]]] = {}
    for measured, _, score, _, valid_area, _ in prevalence_pool:
        by_slide.setdefault(measured.slide_id, []).append((score, valid_area))
    slide_fractions = [
        sum(score * area for score, area in samples)
        / max(sum(area for _, area in samples), 1e-12)
        for samples in by_slide.values()
    ]
    cellular_area_fraction = sum(slide_fractions) / max(len(slide_fractions), 1)

    if config.quantification_estimator_mode == "continuous_measurement":
        # Report the direct slide-balanced phenotype occupancy measured in the
        # representative ROI pool.  This deliberately bypasses all component-
        # level calibration floors, caps, and reporting bins.  Systematic ROIs
        # are preferred above when enough are available so target-enriched
        # retrieval fields do not inflate the specimen estimate.
        raw_fraction = cellular_area_fraction
        method = (
            "unbinned slide-balanced HistoPLUS valid-area Patho-R1 phenotype "
            "occupancy from representative ROIs"
        )
    elif profile.phenotype == "mucinous" and "mucin_pool" in class_areas:
        map_fraction = class_areas["mucin_pool"] / max(tumor_area, 1e-12)
        core_slides = slides[: min(config.top_k_slides, len(slides))]
        core_ranks = {
            str(item.get("slide_id") or ""): rank
            for rank, item in enumerate(core_slides, 1)
        }
        rank_discount = {1: 1.0, 2: 0.55, 3: 0.30, 4: 0.20, 5: 0.15}
        robust_anchors = [
            item
            for item in adequate
            if item[0].slide_id in core_ranks and item[1] >= 10 and item[2] > 0
        ]
        low_cell_anchor_slides = {
            item[0].slide_id for item in adequate if item[1] < 10 and item[2] > 0
        }
        core_low_cell_anchor_slides = low_cell_anchor_slides & set(core_ranks)
        best_anchor_fraction = max(
            (
                item[2]
                * rank_discount.get(core_ranks[item[0].slide_id], 0.10)
                for item in robust_anchors
            ),
            default=0.0,
        )
        top_anchor_fraction = max(
            (
                item[2]
                for item in robust_anchors
                if core_ranks[item[0].slide_id] == 1
            ),
            default=0.0,
        )
        # When the top target-retrieved section has a broad, cell-qualified
        # mucin phenotype, the map's low-cell diffuse stratum is a plausible
        # undersegmented continuation of that component. Otherwise it remains
        # nonspecific and only the explicit mucin class is admitted.
        conditional_map_fraction = map_fraction
        if top_anchor_fraction >= 0.60:
            conditional_map_fraction = (
                class_areas.get("mucin_pool", 0.0)
                + class_areas.get("diffuse_or_poorly_cohesive_tumor", 0.0)
            ) / max(tumor_area, 1e-12)
        tissue_estimate = min(1.0, 1.10 * conditional_map_fraction)
        # A cell-poor mucin pool cannot support an occupancy-weighted estimate,
        # but reproducibility on two independent sections is stronger than one
        # incidental focus. Keep this strictly as a minor-component detection
        # floor; larger estimates must still come from the whole-map area or a
        # cell-qualified core anchor.
        low_cell_detection_floor = min(
            0.10, 0.05 * len(low_cell_anchor_slides)
        )
        raw_fraction = max(
            tissue_estimate, best_anchor_fraction, low_cell_detection_floor
        )
        # A recurrent explicit mucin class across tumor-bearing sections is
        # stronger than one unusually high aggregate map score. Couple that
        # tissue-level recurrence to a cell-qualified high-occupancy visual
        # anchor before authorizing a substantial component estimate.
        raw_fraction = max(
            raw_fraction,
            _recurrent_mucin_component_floor(
                slide_class_fractions,
                [item[2] for item in robust_anchors],
                low_cell_anchor_slide_count=len(core_low_cell_anchor_slides),
            ),
        )
        method = (
            "whole-map mucin/conditional-low-cell area with TITAN-rank-aware "
            "HistoPLUS-qualified Patho-R1 mucin occupancy"
        )
    elif profile.phenotype == "signet_ring":
        # The poorly-cohesive map is deliberately nonspecific: low-grade
        # glandular cases can receive a high soft score while a true pale
        # signet-ring component is undercalled.  Use it as the sampling frame,
        # then quantify robust, independent Patho-R1 anchors in the fixed
        # five-slide TITAN core. HistoPLUS cell support separates a rare
        # low-cell detection (the 5% floor) from a reproducible component.
        map_fraction = (
            class_areas.get("diffuse_or_poorly_cohesive_tumor", 0.0)
            + class_areas.get("mucin_pool", 0.0)
        ) / max(tumor_area, 1e-12)
        core_slides = slides[: min(config.top_k_slides, len(slides))]
        core_slide_ids = {str(item.get("slide_id") or "") for item in core_slides}
        core_anchors = [
            item
            for item in adequate
            if item[0].slide_id in core_slide_ids and item[2] > 0
        ]
        robust_anchors = [item for item in core_anchors if item[1] >= 5]
        enrichment_anchors = [
            item
            for item in adequate
            if item[0].slide_id not in core_slide_ids
            and item[1] >= 5
            and item[2] > 0
        ]
        if len(core_slides) < config.top_k_slides:
            # Tiny synthetic/single-section cases do not have five independent
            # slide strata, so retain the calibrated tissue-area estimate.
            qualified_anchor_occupancy = max(
                (item[2] for item in robust_anchors), default=0.0
            )
            raw_fraction = map_fraction * min(
                2.0, 0.75 + 2.5 * qualified_anchor_occupancy
            )
        elif robust_anchors:
            raw_fraction = _calibrated_signet_core_fraction(
                [(item[2], item[1]) for item in robust_anchors],
                core_size=config.top_k_slides,
                enrichment_anchor_cells=[item[1] for item in enrichment_anchors],
            )
        else:
            # A visible anchor with fewer than five qualified epithelial/
            # neoplastic-like cells is a detection, not a stable area sample.
            raw_fraction = 0.05 if core_anchors else 0.0
        recurrent_mucin_context = (
            statistics.median(
                item.get("mucin_pool", 0.0)
                for item in slide_class_fractions.values()
            )
            if slide_class_fractions
            else 0.0
        )
        explicit_core_anchor = any(
            item[0].roi_id in explicit_signet_lineage_ids
            for item in robust_anchors
        )
        if (
            robust_anchors
            and recurrent_mucin_context >= 0.03
            and not explicit_core_anchor
        ):
            # In a specimen with recurrent extracellular mucin, vacuolated or
            # single-file cells can be part of mucinous adenocarcinoma rather
            # than a separately quantifiable signet-ring component. Without an
            # explicit signet-ring anchor, retain only a minor-component bin.
            raw_fraction = min(raw_fraction, 0.10)
        method = (
            "fixed-core tissue-stratified signet-ring anchors with HistoPLUS cell "
            "qualification and Patho-R1 occupancy calibration"
        )
    else:
        raw_fraction = cellular_area_fraction
        method = (
            "whole-map tumor sampling with slide-balanced HistoPLUS valid-area "
            "fields and Patho-R1 phenotype occupancy"
        )
    raw_percent = max(0.0, min(100.0, 100.0 * raw_fraction))
    if config.quantification_estimator_mode in {
        "calibrated_continuous",
        "continuous_measurement",
    }:
        estimate: float | int = round(
            raw_percent, config.quantification_output_decimals
        )
        formatted_value = f"{estimate:.{config.quantification_output_decimals}f}"
        formatted_value = formatted_value.rstrip("0").rstrip(".")
    else:
        step = config.quantification_rounding_percent
        estimate = int(step * round(raw_percent / step))
        if phenotype_support_ids and estimate == 0:
            estimate = step
        estimate = min(100, estimate)
        formatted_value = str(estimate)
    effective_n = max(len(adequate), 1)
    standard_error = 100.0 * math.sqrt(
        max(raw_fraction * (1.0 - raw_fraction), 0.0) / effective_n
    ) * 1.25
    interval = (
        round(max(0.0, raw_percent - 1.96 * standard_error), 1),
        round(min(100.0, raw_percent + 1.96 * standard_error), 1),
    )
    confidence = (
        "high"
        if len(adequate) >= 12 and tissue_coverage >= 0.95 and stratum_coverage >= 0.85
        else "medium"
    )
    return [
        QuantitativeEvidence(
            evidence_id="Q0001",
            task_kind="component_fraction",
            target=profile.target,
            estimate=estimate,
            unit="percent",
            formatted_answer=f"Approximately {formatted_value}%",
            interval_95=interval,
            confidence=confidence,
            sampling_method=method,
            tissue_coverage_fraction=tissue_coverage,
            stratum_coverage_fraction=min(1.0, stratum_coverage),
            eligible_slide_count=len(eligible_tumor_slide_ids),
            analyzed_slide_count=len(analyzed_slide_ids),
            sampled_roi_count=len(cellular_lineages),
            adequate_cellular_roi_count=len(adequate),
            numerator=max(0.0, tumor_area * raw_fraction),
            denominator=tumor_area,
            denominator_unit="tissue-map level-0 pixel-squared equivalent",
            component_roles={
                "Tissue Navigation": "whole-slide tumor sampling frame, strata, and area denominator",
                "HistoPLUS": "confidence-qualified tumor-cell mass and valid analyzed area",
                "Patho-R1": "same-ROI visible phenotype anchor",
                "Python Evidence Gate": "coverage, lineage, unit, and uncertainty validation",
            },
            supporting_evidence_ids=list(dict.fromkeys(phenotype_support_ids))[:20],
            gate_status="authorized",
            gate_reasons=[],
            limitations=[
                "The result is an H&E image estimate, not a gross-pathology measurement.",
                "HistoPLUS cell labels support weighting only and do not establish tumor identity.",
                *(
                    [
                        "The percentage is the unbinned measured ROI occupancy; no component calibration floor, cap, or 5-point reporting bin was applied."
                    ]
                    if config.quantification_estimator_mode
                    == "continuous_measurement"
                    else []
                ),
                *(
                    [
                        "The case-level calibrated estimate is reported before the final 5-point reporting bin is applied."
                    ]
                    if config.quantification_estimator_mode
                    == "calibrated_continuous"
                    else []
                ),
                *limitations,
            ][:8],
        )
    ]
