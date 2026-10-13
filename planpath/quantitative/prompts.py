"""Prompts for planning, evidence-gap review, patch description, and answering."""

from __future__ import annotations

import json
from typing import Any

from .schemas import (
    EvidencePlan,
    EvidenceReview,
    FinalAnswer,
    ObservationContent,
    OuterSurfaceAssessment,
    QuantitativeEvidence,
    ROIObservation,
)


PLAN_PROMPT_VERSION = "v6.2-spatial-recall-plan-r1-20260905"
OBSERVATION_PROMPT_VERSION = "v6.2-neutral-outer-surface-r2-20260906"
QUANTITATIVE_OBSERVATION_PROMPT_VERSION = (
    "v7.0-neutral-component-inventory-r1-20260911"
)
OBSERVATION_VERIFICATION_PROMPT_VERSION = (
    "v6.2-neutral-adipose-interface-verifier-r3-20260907"
)
# The r3 verifier changes only selective high-risk retranscription. Retain
# the r2 cache-address family so existing question-independent primary Patho
# transcriptions can be reused and migrated in place instead of regenerating
# every ordinary ROI. Per-record consensus and verifier versions decide
# which high-risk entries must be rechecked.
OBSERVATION_VERIFICATION_CACHE_FAMILY_VERSION = (
    "v6.2-neutral-outer-surface-verifier-r2-20260906"
)

CORE_RULES = """You are the reasoning controller for patient-level pathology VQA.
Rules:
1. Similarity scores, retrieval rank, and CONCH prototype tissue-map probabilities only decide what to inspect; they are never diagnostic evidence.
2. Direct diagnostic evidence must come from the question-independent ROI morphology transcriptions. A linked HistoPLUS cellular_analysis is a model-derived measurement that may support localization, cellular quantification, and focus selection, but its predicted cell types cannot alone establish tumor identity, grade, stage, invasion, or absence. General knowledge may map visible morphology to an answer but cannot invent a missing feature.
3. Do not assume that a retrieved WSI or ROI contains the target. Not observed is not proven absent.
4. A direct, adequate positive finding may support early stopping for an existential question. Negative, exhaustive, extremum, quantitative, and composite conclusions require inspection of the whole supplied candidate-slide pool.
5. Request higher magnification only when the current ROI likely contains relevant tissue but its resolution is inadequate. Request another slide when tissue sampling, slide coverage, or missing target tissue is the limiting factor.
6. Never use specimen-site labels or infer evidence from slide IDs or filenames.
7. Preserve the organ context explicitly named in the question and never substitute a different organ's grading or staging system. WHO/ISUP grading of a renal tumor is not prostate Gleason grading."""


QUESTION_SIDE_MEDICAL_RULES = """
Question-side interpretation rules:
- For renal WHO/ISUP nucleolar grading, the observation magnification matters. Conspicuous,
  eosinophilic nucleoli seen only at 40x objective-level ROI resolution support Grade 2, not Grade 3.
  Grade 3 requires nucleoli already conspicuous at approximately 10x objective resolution; do not
  invent that lower-power visibility from a 40x adjective such as prominent. Grade 4 requires extreme
  pleomorphism, multinucleated giant cells, or rhabdoid/sarcomatoid differentiation. A single isolated
  nucleus or cell without representative tissue context cannot establish the tumor's WHO/ISUP grade.
  Repeated representative 40x tumor fields from independent sampled slides can establish Grade 2 even
  when other regions have inconspicuous nucleoli; do not average away a corroborated higher-grade focus.
- For a gastric pT question, map only a directly identified deepest wall layer: lamina propria,
  muscularis mucosae, or submucosa is pT1; muscularis propria is pT2; subserosal connective tissue
  without serosal penetration is pT3; serosal penetration is pT4a; adjacent-organ invasion is pT4b.
  An isolated mucosal ROI does not prove pT1 when deeper interfaces were not represented.
- If the question asks for the deepest invaded layer, answer with one anatomical layer and never a T
  category. If it asks for pT, answer with one pT category and never two alternatives.
- For renal subtype, papillary renal cell carcinoma requires directly described papillary structures,
  not generic glands or eosinophilic cytoplasm. Clear-cell morphology is supported by nested/alveolar
  architecture with delicate vascular septa and/or clear cytoplasm. Chromophobe morphology requires
  features such as perinuclear halos and wrinkled nuclei; oncocytoma requires granular eosinophilic
  cytoplasm with a compatible nested/tubular pattern. Never diagnose a subtype merely by excluding
  alternatives when its own hallmark was not observed.
- Lauren intestinal type is supported by cohesive gland-forming or tubular architecture; diffuse type
  requires poorly cohesive, discohesive, signet-ring, or single-cell infiltrative morphology; mixed
  type requires both components. Do not use an indeterminate label when asked for the most likely class.
- Gastric differentiation is primarily the degree of gland formation in the tumor's dominant
  architecture. Nuclear pleomorphism, prominent nucleoli, or focal atypia alone do not establish poor
  differentiation when irregular but substantial glands remain. Do not choose the worst focal nuclear
  appearance as though differentiation were an extremum task.
- For a generic gastric histological-type question, answer the minimal parent tumor category supported
  by the epithelial morphology. Intestinal type, diffuse type, and mixed type are Lauren classifications
  and must not replace the requested histological category; differentiation is likewise a separate
  attribute.
- For regional lymph-node metastasis, a positive call requires strong nodal context (such as a lymph-node
  capsule or subcapsular sinus) and a cohesive atypical epithelial or glandular deposit on the same slide.
  The deposit and residual nodal structure may occupy different ROI lineages when tumor has replaced part
  of the node. A nonspecific lymphocytic infiltrate around primary-tumor epithelium is not nodal context.
  A negative call or pN assessment requires coverage of the supplied patient-level candidate pool.
- For renal-capsule involvement, cohesive renal-tumor cells entering or interdigitating within any
  portion of a broad fibrous capsule support involvement; do not require complete traversal into
  perirenal fat. Mere contact with a smooth intact band, intratumoral delicate septa, or a ragged
  tissue-to-glass edge without an identifiable fibrous capsule do not establish capsule involvement.
- When the question explicitly asks about the fat capsule or perirenal fat, require viable renal-tumor
  cells extending into adipose tissue or between adipocytes. Entry into a fibrous capsule alone does
  not establish this deeper adipose-tissue endpoint.
"""


def _schema(model: type[Any]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))


def _question(question: dict[str, Any]) -> dict[str, Any]:
    value = {
        "question_type": question["question_type"],
        "question": question["question"],
        "options": question.get("options", []),
    }
    if question.get("clinical_domain"):
        value["clinical_domain"] = str(question["clinical_domain"])
    return value


def _observation_payload(
    item: ROIObservation, *, include_cellular_analysis: bool = True
) -> dict[str, Any]:
    """Create a bounded, portable observation payload for Qwen.

    HistoPLUS output is useful during slide-local review and for deterministic
    Python aggregation, but its repeated class tables are neither direct
    diagnostic evidence nor needed by the final answer call.  Omitting those
    tables from the cross-slide final prompt prevents large extremum cases
    from exhausting the Qwen context window while preserving the neutral
    morphology and every question-side evidence judgment.
    """

    value = item.model_dump(mode="json")
    # Tissue probabilities define sampling strata but remain hidden from both
    # review and final LLM calls. They are consumed only by the Python
    # quantification/evidence gate.
    value.pop("tissue_navigation", None)
    if not include_cellular_analysis:
        value.pop("cellular_analysis", None)
        return value
    cellular = value.get("cellular_analysis")
    if isinstance(cellular, dict):
        cellular.pop("annotation_path", None)
        cellular.pop("overlay_path", None)
    return value


def plan_messages(question: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        CORE_RULES
        + "\n\nCreate a concise, option-balanced search plan without answering the question. "
        "clinical_domain, when supplied, is fixed cohort-level context and must be reflected in "
        "the retrieval descriptions; it is not a slide label or observed evidence. "
        "slide_retrieval_descriptions must contain 1-3 short English, caption-like descriptions "
        "of the tissue/specimen appearance needed to answer the question; these will be encoded by "
        "TITAN and compared directly with cached whole-slide embeddings. Do not privilege one answer "
        "option or phrase the assessment scope as if a positive finding were already present. For a "
        "yes/no or positive/negative question, include both the target-bearing appearance and its "
        "assessable counterstate in the plan. Do not mention a supplied specimen label because none is "
        "available. low_power_query and high_power_query are sent to a pathology image-text encoder: "
        "write each as a direct visual noun phrase, never as an instruction beginning with Retrieve, "
        "never mention a view or magnification, and never assert that the target is present. The low-power "
        "phrase should describe architecture or interfaces; the high-power phrase should describe the "
        "cellular detail needed for the decision. "
        "Use only 10x, 20x, and/or 40x in preferred_zoom_magnifications. Reserve 10x for "
        "renal WHO/ISUP nucleolar grading, and include 40x only for nuclear, "
        "nucleolar, grading, tumor typing, or unresolved cellular detail. required_magnification is the "
        "lowest level at which a patient-level conclusion is defensible: grading, differentiation, "
        "nuclear detail, mitoses, and tumor typing require 40x; cellular confirmation usually requires "
        "20x; purely architectural extent may use 5x. Non-existential semantics must use "
        "candidate_pool_coverage. target_finding and evidence_goal must describe a neutral assessment, "
        "not a presumed positive result. Return exactly one JSON object.\nJSON Schema: "
        + _schema(EvidencePlan)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(_question(question), ensure_ascii=False)},
    ]


def observation_messages(
    image_url: str,
    magnification: str,
    *,
    quantitative_composition: bool = False,
) -> list[dict[str, Any]]:
    if magnification == "5x":
        visual_checklist = (
            "At 5x, describe the dominant tissue organization, spatial distribution, stromal "
            "reaction, and visible interfaces. Use a specific term such as glandular or papillary "
            "only when that structure is unambiguous in the pixels; otherwise state that the "
            "architecture is not reliably resolved. Do not force every possible feature into the "
            "description and do not turn empty clefts into glands or papillae. "
            "When different generic tissue compartments are visible, describe components such as "
            "smooth muscle, fibrous stroma, and adipose tissue. If cellular nests or glands visibly "
            "cross their interface, report that local spatial relationship. If crossing is not "
            "directly visible, omit the claim rather than declaring the whole compartment or specimen "
            "uninvolved. This is a local spatial description, not a specimen-level invasion conclusion. "
            "When a cellular epithelial proliferation approaches an outer or free tissue surface, "
            "describe whether a mesothelial lining is actually visible and whether the proliferation "
            "is separated from, reaches, or crosses that surface. Distinguish this from an inked, "
            "cauterized, sharply cut, resection, or dissection surface whenever visible. "
            "Do not describe nuclei, chromatin, nucleoli, individual mitoses, or nuclear grade. "
        )
    elif magnification == "10x":
        visual_checklist = (
            "At 10x, describe the dominant cellular architecture and only nuclear features that "
            "are genuinely conspicuous across this full physical field. For renal nucleolar "
            "assessment, explicitly state whether discrete eosinophilic nucleoli are already "
            "clearly visible in multiple intact cells at this resolution; do not magnify mentally, "
            "infer visibility from dark nuclear dots, or use prominent/conspicuous when the feature "
            "would require a closer crop. Distinguish a representative regional finding from one "
            "isolated nucleus. Describe visible tissue interfaces, but do not convert a local "
            "relationship into a diagnosis or specimen-level conclusion. "
        )
    else:
        visual_checklist = (
            "At 20x/40x, describe only clearly visible cell arrangement, cytoplasm, nuclei, vessels, "
            "stroma, necrosis, or inflammation; do not mechanically fill a checklist with presumed "
            "negative findings. Distinguish pale material inside nucleated cell borders from "
            "extracellular myxoid material, adipocyte spaces, processing clefts, and blank glass. "
            "Call cytoplasm clear or vacuolated only when the pale space is intracellular. State "
            "not assessable when too few intact nucleated cells are present. Call nucleoli prominent "
            "or conspicuous only when discrete nucleolar bodies are sharply visible inside multiple "
            "intact nuclei; if nuclei are too small, blurred, or equivocal, explicitly say nucleoli "
            "are not assessable and do not infer prominence from dark staining. Use necrosis only "
            "for a tissue zone showing loss of viable cellular outlines with ghost cells and/or "
            "karyorrhectic debris; do not call lumina, clear cytoplasm, folds, or dark staining necrosis. "
            "At 5x and 20x, when a cellular epithelial proliferation approaches a fibrous band or "
            "an adipose boundary, trace whether the same proliferation is visibly continuous across "
            "that boundary. If crossing is not directly visible in this ROI, say that crossing is not "
            "shown here; do not convert limited local visibility into a specimen-level negative. "
            "When epithelial groups coexist with a nerve profile, endothelial-lined channel, fibrous "
            "boundary, adipose interface, or inked/cauterized tissue edge, explicitly describe whether "
            "they are separated, abutting, surrounding, inside, or crossing that structure. Do not turn "
            "this local geometry into a specimen-level diagnosis. "
            "If an epithelial proliferation is near an outer/free tissue surface, state whether a "
            "mesothelial lining is visible and whether the cells are separated from, reach, or cross "
            "it; do not equate an inked, cauterized, sharply cut, resection, or dissection surface "
            "with a free mesothelial surface. "
        )
    quantitative_checklist = (
        " For this composition-sampling ROI, also record neutral features needed for area/cell "
        "quantification. At low power, state whether extracellular mucin-like pools are absent, "
        "focal, broad, or predominant in this field. At 20x/40x, explicitly distinguish: "
        "(a) extracellular mucin containing epithelial cells or glands; (b) discohesive single "
        "epithelial cells with an intracytoplasmic mucin-like vacuole and an eccentrically displaced "
        "nucleus; and (c) trabecular/solid polygonal cells with abundant eosinophilic granular "
        "cytoplasm. Describe only a pattern actually visible, and qualify it as rare/scattered, "
        "focal, extensive, or predominant when supportable. These are visual attributes, not disease "
        "names or diagnoses."
        if quantitative_composition
        else ""
    )
    system = (
        "You are a question-independent pathology image transcriber observing exactly one ROI. You "
        "are intentionally not given a clinical question, answer options, specimen label, retrieval "
        "query, or diagnostic target. Report only morphology directly visible in the pixels. Do not "
        "name a disease or lesion class (including benign/malignant/tumor), assign a grade or stage, "
        "state a specimen-level margin/invasion conclusion, infer an organ/site, invent clinical "
        "history, "
        "or claim that an unseen feature is absent. At 5x describe tissue composition, architecture, "
        "interfaces, and geographic changes only; do not claim nuclear detail, nucleoli, or mitotic "
        "counts. At 10x, report only nuclear details genuinely conspicuous across the full field. "
        "At 20x/40x, describe visible cellular and nuclear morphology without converting it "
        "to a diagnosis. Inspect the actual pixels instead of completing a memorized report template. "
        + visual_checklist
        + quantitative_checklist
        + "If image quality prevents description, say so in limitations. Return exactly "
        "one JSON object with only image_quality, morphology, and limitations. "
        "JSON Schema: "
        + _schema(ObservationContent)
    )
    payload = {
        "magnification": magnification,
        "instruction": "Transcribe this ROI neutrally and independently of any downstream question.",
    }
    if quantitative_composition:
        payload["quantitative_composition_inventory"] = True
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ],
        },
    ]


def observation_verification_messages(
    image_url: str,
    magnification: str,
) -> list[dict[str, Any]]:
    """Independent, standardized visual inventory for high-risk claims."""

    system = (
        "You are an independent pathology image transcription quality controller. You are not "
        "given a clinical question, answer choices, specimen label, retrieval query, first-pass "
        "description, or diagnostic target. Inspect this one ROI from scratch and report only "
        "features directly visible in its pixels. Do not name a disease, lesion class, organ/site, "
        "grade, stage, margin status, or patient-level invasion status. Do not guess a missing "
        "feature. At 5x report only unambiguous architecture, generic tissue compartments, and local "
        "interfaces. Call a papillary structure only when true fronds with visible fibrovascular "
        "cores are present. At 10x, call nucleoli conspicuous only when discrete bodies are already "
        "clearly visible in multiple intact cells across the unzoomed physical field. At 20x/40x, "
        "call nucleoli prominent only when discrete nucleolar bodies "
        "are sharply visible in multiple intact nuclei; otherwise say not assessable. Call necrosis "
        "only for a tissue zone with loss of viable cell outlines plus ghost cells or karyorrhectic "
        "debris. Distinguish intracellular clear cytoplasm from extracellular pale material, fat, "
        "lumina, clefts, and blank glass. Report a crossing relationship only when cellular nests or "
        "glands visibly traverse a smooth-muscle, fibrous-band, stromal, or adipose interface. If an "
        "approached boundary is visible but direct crossing is not, report only that crossing is not "
        "shown in this ROI and do not state a specimen-level negative. At 20x/40x, whenever epithelial "
        "cells or glands and a nerve profile or endothelial-lined channel coexist in the ROI, describe "
        "their visible relationship explicitly as separated, abutting, surrounding/encasing, or inside "
        "the structure. Use surrounding/encasing or inside only when that geometry is directly visible; "
        "do not infer it from proximity, a clear space, or the mere presence of both structures. "
        "At 20x/40x, whenever epithelial cells or glands and mature adipocytes coexist, first distinguish "
        "true adipocytes with large optically empty cytoplasmic vacuoles and peripheral nuclei from "
        "intracellular clear cytoplasm, glands, vessels, clefts, and glass. Then explicitly describe the "
        "epithelial-to-adipocyte relationship as separated, abutting, or entering between adipocytes. "
        "Use entering between adipocytes only when intact nucleated epithelial cells are directly visible "
        "among multiple unequivocal mature adipocytes; proximity or co-occurrence is insufficient. "
        "When an epithelial proliferation approaches an outer/free tissue surface, report whether a "
        "mesothelial lining is visible and whether the cells are separated from, reach, or traverse "
        "that surface. Explicitly distinguish any inked, cauterized, sharply cut, resection, or "
        "dissection surface; these are not interchangeable with a mesothelial free surface. Return "
        "exactly one "
        "JSON object with only image_quality, morphology, and limitations. JSON Schema: "
        + _schema(ObservationContent)
    )
    payload = {
        "magnification": magnification,
        "instruction": "Independently transcribe this ROI for visual-consensus checking.",
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ],
        },
    ]


def outer_surface_verification_messages(
    image_url: str,
    magnification: str,
) -> list[dict[str, Any]]:
    """Resolve a candidate tissue-to-glass boundary without clinical context."""

    system = (
        "You independently inspect one pathology ROI and classify only its visible tissue-to-glass "
        "boundary. You are not given a clinical question, answer, specimen label, retrieval query, "
        "organ, or first-pass description. Do not diagnose, grade, stage, or infer an unseen gross "
        "surface. An anatomic free surface requires visible mesothelial-type lining or a convincing "
        "smooth external tissue contour with coherent underlying tissue organization. Blank glass "
        "alone is insufficient. Torn, folded, fragmented, sharply cut, inked, cauterized, or otherwise "
        "artifactual edges are cut_resection_or_artifact_edge; use indeterminate whenever the image "
        "cannot distinguish them from a true free surface. Separately classify the directly visible "
        "relationship of epithelial cells or glands to that same boundary. 'Reaches' means cells are "
        "directly exposed at the boundary with no intervening mesothelium or stroma; proximity or "
        "contact with an intact mesothelial layer is separated_from_surface. 'Traverses' requires "
        "cells visibly crossing beyond the boundary. Keep the rationale under 300 characters. "
        "Return exactly one JSON object. JSON Schema: "
        + _schema(OuterSurfaceAssessment)
    )
    payload = {
        "magnification": magnification,
        "instruction": "Classify the visible boundary and epithelial-to-boundary relation conservatively.",
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ],
        },
    ]


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
    system = (
        CORE_RULES
        + QUESTION_SIDE_MEDICAL_RULES
        + "\n\nJudge the current evidence gap and select exactly one status: sufficient, "
        "need_higher_magnification, or need_more_slides. For higher magnification, choose only an "
        "unattempted 10x/20x/40x level allowed by the plan, and copy only the 1-2 most "
        "question-relevant focus_roi_ids from "
        "available_focus_roi_ids; never copy the whole ROI list. "
        "Interpret the neutral morphology on the question side. Keep the output compact. For every "
        "status, record evidence_judgments for question-relevant adequate observations from "
        "current_slide_observations; use an empty list only if every current observation is "
        "inadequate. Normally include 1-3 judgments. For an extremum decision at every review, compare "
        "every current observation at the highest attempted "
        "magnification and include one judgment for each adequate ROI (up to five); a focal "
        "higher-severity finding must not be outvoted by several bland ROIs. For sufficient, include "
        "only 2-5 decisive judgments. Each "
        "judgment rationale must be one short sentence. Never add a feature absent from an ROI's "
        "morphology text. A direct judgment must cite a description that visibly establishes the "
        "claimed direction at an adequate magnification. A low-power ROI with potentially relevant "
        "target architecture but unresolved cellular detail is supporting, not not_relevant. When "
        "requesting higher magnification, every focus_roi_id must be the roi_id of a current "
        "direct/supporting judgment; prefer potentially relevant architecture over preserved normal "
        "background. evidence_judgments and supporting_evidence_ids require evidence_id values (E####), "
        "whereas focus_roi_ids require roi_id values. "
        "For sufficient, answer_candidate must obey "
        "the question format and supporting evidence IDs must be copied from direct/supporting "
        "judgments. For either insufficient status answer_candidate must "
        "be null. Do not use WSI rank, retrieval similarity, ROI selection query, or identifiers as "
        "evidence. Never use tissue_navigation fields as evidence. HistoPLUS cellular_analysis may "
        "prioritize a focus ROI or corroborate a supporting judgment, but a direct judgment and any "
        "sufficient conclusion must remain visibly grounded in that ROI's morphology text. A "
        "HistoPLUS negative result does not prove absence. If the plan requires candidate_pool_coverage and candidate slides remain, do not "
        "declare sufficient. Do not use 5x observations for nuclear grade, nucleoli, mitotic activity, "
        "or other details below their resolution. If required_magnification has not been attempted on "
        "the current potentially relevant tissue, request it before moving to another slide. This is a "
        "strictly local current-slide review: prior slides are consolidated later by the final-answer "
        "call, so never cite an evidence ID absent from current_slide_observations. Return "
        "exactly one JSON object.\nJSON Schema: "
        + _schema(EvidenceReview)
    )
    payload = {
        "question": _question(question),
        "plan": plan.model_dump(mode="json"),
        "current_slide_id": current_slide_id,
        "remaining_candidate_slide_ids": remaining_candidate_slide_ids,
        "available_focus_roi_ids": available_focus_roi_ids,
        "attempted_magnifications_on_current_slide": attempted_magnifications,
        "current_slide_observations": [
            _observation_payload(item)
            for item in observations
            if item.slide_id == current_slide_id
        ],
        "prior_observation_count_reserved_for_final": sum(
            item.slide_id != current_slide_id for item in observations
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


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
    quantitative_evidence: list[QuantitativeEvidence] | None = None,
) -> list[dict[str, str]]:
    quantitative_evidence = quantitative_evidence or []
    quantitative_rules = (
        "An authorized Q#### item is a Python-gated measurement, not a raw tissue class or cell "
        "label. It may support the requested numeric answer because its tissue denominator, "
        "cellular weighting, visible E#### phenotype anchors, coverage, and uncertainty have "
        "already passed the gate. An insufficient Q#### item must never be converted into a "
        "number. supporting_evidence_ids may refer to supplied observations or quantitative_evidence. "
        if quantitative_evidence
        else ""
    )
    supporting_rule = (
        "supporting_evidence_ids must refer to supplied observations or quantitative_evidence "
        "and include no more than five IDs. "
        if quantitative_evidence
        else "supporting_evidence_ids must refer to supplied observations and include no more than five IDs. "
    )
    if evidence_sufficient:
        binding = (
            "The evidence gate authorized a sufficient answer. Copy authorized_answer_candidate "
            "exactly as answer. Set evidence_sufficiency to sufficient and termination_reason to "
            "evidence_sufficient."
        )
    else:
        binding = (
            "The candidate slide pool or compute budget was exhausted without sufficient evidence. "
            "You must still give the most likely answer supported by the observations; set confidence "
            "to low and evidence_sufficiency to insufficient. For free-text answers, do not answer "
            "Unknown, None, or another non-answer. For single-choice, every explicitly supplied option "
            "is valid, including an indeterminate option when that best reflects the evidence. Copy the supplied "
            f"termination reason exactly: {termination_reason}."
        )
    system = (
        CORE_RULES
        + QUESTION_SIDE_MEDICAL_RULES
        + "\n\nProduce the final answer by interpreting the question-independent ROI morphology. "
        "Do not treat the search plan, retrieval terms, tissue_navigation, or tissue-map class "
        "probabilities as observed evidence. Linked HistoPLUS measurements are auxiliary and cannot "
        "replace a same-ROI visible morphology finding. Yes/no must "
        "be exactly Yes or No; single-choice must exactly copy one option; short-answer must be one "
        "concise conclusion. Answer the requested attribute, never a description of background tissue "
        "and never an abstention such as no primary tumor identified. For subtype decisions, a directly "
        "described hallmark architecture or cytoplasm pattern outweighs nonspecific pleomorphism or "
        "mitotic activity; severity alone never identifies a subtype. For highest-grade decisions, do "
        "not infer a high grade from generic atypia when the defining grade feature was not described. "
        "For a generic histological-type question, return the minimal histological category and do not "
        "append a separate classification or differentiation qualifier that was not requested. "
        "Read the complete question_side_review_history before deciding: an earlier slide's direct "
        "finding remains evidence and must not be forgotten merely because the final slide was weak. "
        "Resolve conflicts by preferring adequate required-magnification hallmark morphology over "
        "lower-magnification or nonspecific architecture. For extremum questions such as highest "
        "grade, decide from the strongest focal qualifying feature across the complete review history, "
        "not from the majority or average appearance. "
        "The observations payload has already been selected from direct/supporting question-side "
        "judgments and includes linked low-power parents where available; do not reinterpret omitted "
        "not-relevant observations as evidence. "
        + quantitative_rules
        + supporting_rule
        + binding
        + " Return exactly one JSON object.\nJSON Schema: "
        + _schema(FinalAnswer)
    )
    payload = {
        "question": _question(question),
        "plan": plan.model_dump(mode="json"),
        "observations": [
            _observation_payload(item, include_cellular_analysis=False)
            for item in observations
        ],
        "last_question_side_review": (
            last_review.model_dump(mode="json") if last_review is not None else None
        ),
        "question_side_review_history": [
            {
                "status": item.status,
                "reason": item.reason,
                "conclusion_direction": item.conclusion_direction,
                "evidence_judgments": [
                    judgment.model_dump(mode="json")
                    for judgment in item.evidence_judgments
                ],
            }
            for item in review_history
        ],
        "authorized_answer_candidate": authorized_answer_candidate,
        "authorized_termination_reason": termination_reason,
        "authorized_evidence_sufficiency": (
            "sufficient" if evidence_sufficient else "insufficient"
        ),
    }
    if quantitative_evidence:
        payload["quantitative_evidence"] = [
            item.model_dump(mode="json") for item in quantitative_evidence
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
