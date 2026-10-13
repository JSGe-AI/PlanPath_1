"""Small, strict contracts used by the v7 state machine."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Magnification = Literal["5x", "10x", "20x", "40x"]
EvidenceStatus = Literal[
    "sufficient", "need_higher_magnification", "need_more_slides"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, protected_namespaces=()
    )


class EvidencePlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    target_finding: Annotated[str, Field(min_length=1, max_length=300)]
    question_attributes: Annotated[list[str], Field(min_length=1, max_length=6)]
    decision_semantics: Literal[
        "existential", "exhaustive", "extremum", "quantitative", "composite"
    ]
    slide_retrieval_descriptions: Annotated[
        list[str], Field(min_length=1, max_length=3)
    ]
    low_power_query: Annotated[str, Field(min_length=1, max_length=300)]
    high_power_query: Annotated[str, Field(min_length=1, max_length=300)]
    preferred_zoom_magnifications: Annotated[
        list[Literal["10x", "20x", "40x"]], Field(max_length=3)
    ]
    required_magnification: Magnification
    evidence_goal: Annotated[str, Field(min_length=1, max_length=600)]
    stop_policy: Literal["direct_positive", "candidate_pool_coverage"]

    @field_validator(
        "question_attributes",
        "slide_retrieval_descriptions",
        "preferred_zoom_magnifications",
    )
    @classmethod
    def unique_items(cls, values: list[Any]) -> list[Any]:
        if any(not str(value).strip() for value in values):
            raise ValueError("list items must be non-empty")
        # Small local models occasionally repeat one otherwise valid member
        # verbatim. Rejecting that deterministic formatting defect caused
        # every temperature-zero repair (and then every task retry) to return
        # the same invalid plan. Preserve the first occurrence instead.
        output: list[Any] = []
        seen: set[str] = set()
        for value in values:
            key = str(value).casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(value)
        return output

    @model_validator(mode="after")
    def valid_policy(self) -> "EvidencePlan":
        if self.decision_semantics != "existential" and self.stop_policy == "direct_positive":
            raise ValueError("only existential questions may use direct_positive stopping")
        ordered = sorted(
            self.preferred_zoom_magnifications,
            key=lambda value: int(value.rstrip("x")),
        )
        if self.preferred_zoom_magnifications != ordered:
            raise ValueError("preferred zoom magnifications must be ordered")
        return self


class SlideScore(StrictModel):
    slide_id: Annotated[str, Field(min_length=1)]
    rank: Annotated[int, Field(ge=1)]
    score: float
    matched_description: Annotated[str, Field(min_length=1)]
    per_description_scores: list[float]


class TissueNavigation(StrictModel):
    """Question-conditioned navigation metadata; never diagnostic evidence."""

    map_available: bool
    target_score: float = 0.0
    interface_score: float = 0.0
    heterogeneity: float = 0.0
    map_confidence: float = 0.0
    unknown_fraction: float = 0.0
    fusion_score: float = 0.0
    adipose_morphology_score: float = 0.0
    selection_route: str = "plip_only"
    target_labels: list[str] = Field(default_factory=list)
    class_fractions: dict[str, float] = Field(default_factory=dict)


class ROI(StrictModel):
    roi_id: Annotated[str, Field(min_length=1)]
    slide_id: Annotated[str, Field(min_length=1)]
    coordinates: tuple[int, int, int, int]
    magnification: Magnification
    retrieval_score: float
    matched_query: Annotated[str, Field(min_length=1)]
    parent_roi_id: str | None = None
    tissue_navigation: TissueNavigation | None = None

    @model_validator(mode="after")
    def valid_box(self) -> "ROI":
        x1, y1, x2, y2 = self.coordinates
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid ROI box: {self.coordinates}")
        return self


class ObservationContent(StrictModel):
    """Question-independent transcription of one ROI.

    Patho-R1 is deliberately not asked whether a target is present.  Mapping
    neutral morphology to a VQA conclusion belongs to the Qwen controller.
    """

    image_quality: Literal["adequate", "limited", "inadequate"]
    morphology: Annotated[str, Field(min_length=1, max_length=900)]
    limitations: Annotated[list[str], Field(default_factory=list, max_length=4)]


class OuterSurfaceAssessment(StrictModel):
    """Question-independent visual check of a tissue-to-glass boundary."""

    boundary_identity: Literal[
        "anatomic_free_surface",
        "cut_resection_or_artifact_edge",
        "indeterminate",
    ]
    epithelial_relation: Literal[
        "not_present_or_not_assessable",
        "separated_from_surface",
        "reaches_surface",
        "traverses_surface",
    ]
    rationale: Annotated[str, Field(min_length=1, max_length=500)]


class CellularAnalysis(StrictModel):
    """Question-independent HistoPLUS measurements linked to one H&E ROI."""

    source: Literal["HistoPLUS"] = "HistoPLUS"
    model_name: str
    model_signature: str
    input_mpp: float
    quality: Literal["adequate", "limited", "unavailable"]
    instance_count: Annotated[int, Field(ge=0)]
    valid_area_mm2: Annotated[float, Field(ge=0)]
    class_counts: dict[str, int] = Field(default_factory=dict)
    class_densities_per_mm2: dict[str, float] = Field(default_factory=dict)
    grouped_counts: dict[str, int] = Field(default_factory=dict)
    grouped_densities_per_mm2: dict[str, float] = Field(default_factory=dict)
    mean_type_confidence: float = 0.0
    nuclear_area_um2_quantiles: dict[str, float] = Field(default_factory=dict)
    cancer_immune_contact_fraction: float | None = None
    cancer_endothelial_contact_fraction: float | None = None
    annotation_path: str | None = None
    overlay_path: str | None = None
    limitations: Annotated[list[str], Field(default_factory=list, max_length=4)]


class QuantitativeEvidence(StrictModel):
    """Auditable Python-derived measurement admitted by the evidence gate.

    ``Q`` evidence is deliberately separate from morphology (``E`` evidence).
    A numerical result is authorized only when its tissue denominator and its
    cellular measurement are both available and at least one linked ROI
    visibly anchors the requested phenotype.
    """

    schema_version: Literal["1.0"] = "1.0"
    evidence_id: Annotated[str, Field(pattern=r"^Q\d{4}$")]
    task_kind: Literal[
        "component_fraction", "cell_density", "focus_count", "spatial_distance"
    ]
    target: Annotated[str, Field(min_length=1, max_length=180)]
    estimate: float | int | None = None
    unit: Literal["percent", "cells_per_mm2", "count", "millimeter"]
    formatted_answer: Annotated[str | None, Field(max_length=180)] = None
    interval_95: tuple[float, float] | None = None
    confidence: Literal["high", "medium", "low", "unavailable"]
    sampling_method: Annotated[str, Field(min_length=1, max_length=240)]
    tissue_coverage_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    stratum_coverage_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    eligible_slide_count: Annotated[int, Field(ge=0)] = 0
    analyzed_slide_count: Annotated[int, Field(ge=0)] = 0
    sampled_roi_count: Annotated[int, Field(ge=0)] = 0
    adequate_cellular_roi_count: Annotated[int, Field(ge=0)] = 0
    numerator: Annotated[float | None, Field(ge=0.0)] = None
    denominator: Annotated[float | None, Field(gt=0.0)] = None
    denominator_unit: str | None = None
    component_roles: dict[str, str] = Field(default_factory=dict)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    gate_status: Literal["authorized", "insufficient"]
    gate_reasons: list[str] = Field(default_factory=list)
    limitations: Annotated[list[str], Field(default_factory=list, max_length=8)]

    @model_validator(mode="after")
    def valid_quantitative_gate(self) -> "QuantitativeEvidence":
        if self.interval_95 is not None:
            low, high = self.interval_95
            if low > high:
                raise ValueError("interval_95 must be ordered")
        if self.gate_status == "authorized":
            if self.estimate is None or not self.formatted_answer:
                raise ValueError("authorized quantitative evidence needs an answer")
            if self.gate_reasons:
                raise ValueError("authorized quantitative evidence cannot have gate reasons")
            if not self.supporting_evidence_ids:
                raise ValueError("authorized quantitative evidence needs linked E evidence")
        return self


class ROIObservation(ObservationContent):
    evidence_id: Annotated[str, Field(min_length=1)]
    slide_id: Annotated[str, Field(min_length=1)]
    roi_id: Annotated[str, Field(min_length=1)]
    magnification: Magnification
    parent_roi_id: str | None = None
    # Carried for the deterministic quantifier only. Prompt serialization
    # always removes this navigation metadata before any reasoning call.
    tissue_navigation: TissueNavigation | None = None
    cellular_analysis: CellularAnalysis | None = None


class EvidenceJudgment(StrictModel):
    """Question-side interpretation of one neutral ROI description."""

    evidence_id: Annotated[str, Field(min_length=1)]
    relevance: Literal["direct", "supporting", "not_relevant"]
    direction: Literal["positive", "negative", "other", "indeterminate"]
    rationale: Annotated[str, Field(min_length=1, max_length=160)]


class EvidenceReview(StrictModel):
    status: EvidenceStatus
    reason: Annotated[
        str, Field(min_length=1, max_length=600)
    ] = "No additional rationale was returned."
    conclusion_direction: Literal["positive", "negative", "other", "indeterminate"]
    answer_candidate: str | None
    supporting_evidence_ids: list[str]
    evidence_judgments: Annotated[
        list[EvidenceJudgment], Field(default_factory=list, max_length=5)
    ]
    focus_roi_ids: Annotated[list[str], Field(max_length=2)]
    next_magnification: Literal["10x", "20x", "40x"] | None

    @model_validator(mode="after")
    def fields_match_status(self) -> "EvidenceReview":
        if self.status == "sufficient":
            if not self.answer_candidate or not self.supporting_evidence_ids:
                raise ValueError("sufficient review needs an answer and supporting evidence")
            if self.next_magnification is not None:
                raise ValueError("sufficient review cannot request magnification")
        elif self.status == "need_higher_magnification":
            if self.next_magnification is None or not self.focus_roi_ids:
                raise ValueError("magnification request needs a level and focus ROIs")
            if self.answer_candidate is not None:
                raise ValueError("an insufficient review cannot propose a final answer")
        else:
            if self.next_magnification is not None:
                raise ValueError("need_more_slides cannot request magnification")
            if self.answer_candidate is not None:
                raise ValueError("an insufficient review cannot propose a final answer")
        return self


class FinalAnswer(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    answer: Annotated[str, Field(min_length=1, max_length=180)]
    confidence: Literal["high", "medium", "low"]
    evidence_sufficiency: Literal["sufficient", "insufficient"]
    termination_reason: Literal[
        "evidence_sufficient", "candidate_slides_exhausted", "budget_exhausted"
    ]
    supporting_evidence_ids: list[str]
    brief_rationale: Annotated[str, Field(min_length=1, max_length=700)]


class WorkflowState(StrictModel):
    candidate_slide_ids: list[str]
    visited_slide_ids: list[str]
    current_slide_id: str | None
    observations: list[ROIObservation]
    reviews: list[EvidenceReview]
    termination_reason: Literal[
        "evidence_sufficient", "candidate_slides_exhausted", "budget_exhausted"
    ] | None = None
