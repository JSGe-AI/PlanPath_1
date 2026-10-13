"""Question-anchored Tissue Map and HistoPLUS routing for r25.

r23_5 derives routing from the complete generated search plan.  Incidental
words in a high-power query (for example ``nucleoli`` in a hilar-vessel task)
can consequently activate a grade-oriented HistoPLUS route.  r25 classifies
the clinical endpoint from the original question and uses the plan only as a
fallback when no task context is installed.
"""

from __future__ import annotations

import re

from planpath.core.schemas import EvidencePlan
from planpath.core.tissue_routing import (
    TissueRoutingPolicy,
    routing_policy as r23_5_routing_policy,
)

from .context import current_question


def _question_text() -> tuple[str, str]:
    question = current_question() or {}
    return (
        str(question.get("question") or "").casefold(),
        str(question.get("clinical_domain") or "").casefold(),
    )


def component_route() -> str:
    """Return the evidence role supported by the original question."""

    text, domain = _question_text()
    if not text:
        return "fallback"
    if any(
        term in text
        for term in (
            "separately submitted adrenal",
            "additional benign cystic lesion",
            "renal hilar arterial",
            "renal hilar venous",
            "hilar arterial or venous",
        )
    ):
        return "semantic_only"
    if any(term in text for term in ("margin", "dissection surface", "cut edge")):
        return "margin_interface"
    if "capsule" in text or any(
        term in text
        for term in (
            "perirenal fat",
            "perinephric fat",
            "perirenal adipose",
            "perirenal soft tissue",
            "renal sinus fat",
            "renal sinus tissue",
        )
    ):
        return "renal_boundary"
    if any(
        term in text
        for term in (
            "renal sinus",
        )
    ):
        # Renal sinus involvement can be localized as a tumour-to-fat/stroma
        # transition. Renal-pelvis questions remain semantic-only because the
        # map has no urothelial/pelvicalyceal class; activating it there would
        # add noise rather than information.
        return "collecting_system_interface"
    if any(
        term in text
        for term in (
            "lymphovascular",
            "vascular invasion",
            "tumor embol",
            "tumour embol",
            "tumor thrombus",
            "tumour thrombus",
        )
    ):
        return "vascular_interface"
    if any(term in text for term in ("perineural", "neural invasion")):
        return "neural_interface"
    if re.search(r"\blymph\s+nodes?\b|\bregional[-\s]+nodes?\b", text) or any(
        term in text for term in ("pn category", "pn stage", "pathological n")
    ):
        return "nodal"
    pt_question = bool(
        re.search(
            r"\b(?:pT\s+(?:category|stage)|pathological\s+T\s*(?:\(pT\))?\s*"
            r"(?:category|stage)\b|primary[-\s]+tumou?r\s+category\s*\(pT\))",
            str((current_question() or {}).get("question") or ""),
            flags=re.IGNORECASE,
        )
    )
    if pt_question or any(
        term in text
        for term in (
            "deepest layer",
            "deepest level",
            "deepest tissue",
            "serosa",
            "subserosa",
            "adjacent site",
            "adjacent structure",
        )
    ):
        return "depth_interface"
    if "who/isup" in text or any(
        term in text for term in ("nuclear grade", "highest grade", "mitotic")
    ):
        return "cellular_extremum"
    if any(
        term in text
        for term in (
            "necros",
            "necrot",
            "hemorrhage",
            "haemorrhage",
            "calcification",
            "mineralization",
            "cystic change",
            "secondary morphological change",
        )
    ):
        return "tissue_composition"
    if any(
        term in text
        for term in (
            "tumor regression grade",
            "tumour regression grade",
            "trg of the tumor",
            "trg of the tumour",
            "original tumor bed",
            "original tumour bed",
            "treatment-related changes",
        )
    ):
        return "tumor_bed_response"
    if any(
        term in text
        for term in (
            "lauren",
            "differentiation",
            "histological type",
            "pathological type",
            "tumor type",
            "tumour type",
            "histological component",
            "histological components",
            "histological finding",
            "stromal architectural feature",
            "sarcomatoid",
            "rhabdoid",
        )
    ):
        return "tumor_phenotype"
    if any(term in text for term in ("tumor foci", "tumour foci", "multifocal")):
        return "tumor_distribution"
    if any(term in text for term in ("immune", "lymphocyte", "inflammatory")):
        return "immune_context"
    # Coarse tissue classes are not allowed to perturb unrelated tasks.
    return "semantic_only"


def tissue_navigation_applicable() -> bool:
    return component_route() not in {"semantic_only", "fallback"}


def histoplus_applicable() -> bool:
    return component_route() in {
        "cellular_extremum",
        "nodal",
        "vascular_interface",
        "neural_interface",
        "depth_interface",
        "renal_boundary",
        "collecting_system_interface",
        "margin_interface",
        "immune_context",
        "tumor_phenotype",
        "tumor_bed_response",
        "tumor_distribution",
    }


def routing_policy(cancer: str, plan: EvidencePlan) -> TissueRoutingPolicy:
    route = component_route()
    if route == "fallback":
        return r23_5_routing_policy(cancer, plan)

    renal = cancer == "renal"
    tumor = ("viable_renal_tumor",) if renal else (
        "gland_forming_tumor",
        "diffuse_or_poorly_cohesive_tumor",
    )
    stroma = "interstitium_or_fibrosis" if renal else "desmoplastic_stroma_or_tumor_bed"
    adipose = "adipose_or_soft_tissue" if renal else "adipose_or_perigastric_soft_tissue"
    necrosis = "necrosis_or_debris" if renal else "necrosis_or_ulcer_debris"
    immune = "immune_or_lymphoid"
    targets = list(tumor)
    interfaces: list[tuple[str, str]] = []
    histo_policy = "off"
    histo_targets: list[str] = []
    reason = f"r25-question-anchored:{route}"

    if route == "margin_interface":
        interfaces.extend((item, adipose) for item in tumor)
        interfaces.extend((item, stroma) for item in tumor)
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "interface_confirmation"]
        reason = "tumor-to-cut-surface interface search"
    elif route == "renal_boundary":
        interfaces.extend((item, adipose) for item in tumor)
        interfaces.extend((item, stroma) for item in tumor)
        if renal:
            interfaces.extend((item, "artifact_or_background") for item in tumor)
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "interface_confirmation"]
        text, _ = _question_text()
        reason = (
            "tumor-perirenal-fat interface search"
            if any(term in text for term in ("perirenal", "perinephric", "sinus fat"))
            else "tumor-capsule interface search"
        )
    elif route == "collecting_system_interface":
        interfaces.extend((item, stroma) for item in tumor)
        interfaces.extend((item, adipose) for item in tumor)
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "epithelial", "interface_confirmation"]
        reason = "tumor to renal collecting-system or sinus interface search"
    elif route == "vascular_interface":
        interfaces.extend((item, stroma) for item in tumor)
        histo_policy = "required"
        histo_targets = ["neoplastic_like", "endothelial", "spatial_contact"]
        reason = "tumor-vessel spatial-relation search"
    elif route == "neural_interface":
        interfaces.extend((item, stroma) for item in tumor)
        histo_policy = "required"
        histo_targets = ["neoplastic_like", "spatial_contact"]
        reason = "tumor-nerve spatial-relation search"
    elif route == "nodal":
        targets.append(immune)
        interfaces.extend((item, immune) for item in tumor)
        histo_policy = "required"
        histo_targets = ["neoplastic_like", "epithelial", "immune", "epithelial_cluster"]
        reason = "nodal epithelial-deposit search"
    elif route == "depth_interface":
        interfaces.extend((item, adipose) for item in tumor)
        interfaces.extend((item, stroma) for item in tumor)
        if not renal:
            interfaces.extend((item, "muscularis") for item in tumor)
            interfaces.extend((item, "artifact_or_background") for item in tumor)
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "interface_confirmation"]
        reason = "tumor-boundary or depth interface search"
    elif route == "cellular_extremum":
        histo_policy = "required"
        histo_targets = ["neoplastic_like", "nuclear_heterogeneity", "cellular_events"]
        reason = "cellular or nuclear extremum search"
    elif route == "tissue_composition":
        targets.append(necrosis)
        interfaces.extend((item, necrosis) for item in tumor)
        reason = "tumor-necrosis co-localization"
    elif route == "tumor_phenotype":
        # Cell labels do not diagnose a subtype.  They only nominate a
        # representative epithelial/neoplastic field for the independent
        # 40x morphology transcription.
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "epithelial", "stromal_context"]
        if not renal:
            # Preserve the rare mucin hypothesis independently from the much
            # more common gland-forming and diffuse tumour hypotheses.  This
            # remains a localization class; it is never copied to the answer.
            targets.append("mucin_pool")
        reason = "tumor architecture and cellular-composition search"
    elif route == "tumor_bed_response":
        targets.append(stroma)
        targets.append(immune)
        interfaces.extend((item, stroma) for item in tumor)
        histo_policy = "required"
        histo_targets = ["neoplastic_like", "immune", "stromal_context"]
        reason = "residual-tumor to treatment-bed composition search"
    elif route == "immune_context":
        targets.append(immune)
        histo_policy = "required"
        histo_targets = ["immune", "immune_subtypes"]
        reason = "tumor-microenvironment cellular analysis"
    elif route == "tumor_distribution":
        histo_policy = "conditional"
        histo_targets = ["neoplastic_like", "epithelial"]
        reason = "spatially separated tumor-bearing region search"

    return TissueRoutingPolicy(
        target_labels=tuple(dict.fromkeys(targets)),
        interface_pairs=tuple(dict.fromkeys(interfaces)),
        histoplus_policy=histo_policy,
        histoplus_targets=tuple(histo_targets),
        reason=reason,
    )
