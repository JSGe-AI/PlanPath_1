"""Deterministic question-to-tissue and HistoPLUS routing policies."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import EvidencePlan


@dataclass(frozen=True)
class TissueRoutingPolicy:
    target_labels: tuple[str, ...]
    interface_pairs: tuple[tuple[str, str], ...]
    histoplus_policy: str
    histoplus_targets: tuple[str, ...]
    reason: str


def _text(plan: EvidencePlan) -> str:
    return " ".join(
        [
            plan.target_finding,
            *plan.question_attributes,
            plan.low_power_query,
            plan.high_power_query,
            plan.evidence_goal,
        ]
    ).casefold()


def routing_policy(cancer: str, plan: EvidencePlan) -> TissueRoutingPolicy:
    text = _text(plan)
    renal = cancer == "renal"
    renal_perirenal_fat = renal and any(
        term in text
        for term in (
            "fat capsule",
            "perirenal adipose",
            "perirenal fat",
            "perinephric adipose",
            "perinephric fat",
            "between mature adipocytes",
        )
    )
    tumor = ("viable_renal_tumor",) if renal else (
        "gland_forming_tumor",
        "diffuse_or_poorly_cohesive_tumor",
    )
    immune = "immune_or_lymphoid"
    necrosis = "necrosis_or_debris" if renal else "necrosis_or_ulcer_debris"
    adipose = "adipose_or_soft_tissue" if renal else "adipose_or_perigastric_soft_tissue"
    stroma = "interstitium_or_fibrosis" if renal else "desmoplastic_stroma_or_tumor_bed"

    targets: list[str] = list(tumor)
    interfaces: list[tuple[str, str]] = []
    policy = "off"
    cellular_targets: list[str] = []
    reason = "default tumor-bearing tissue navigation"

    # Route by the relation being assessed before generic cellular words.
    # Interface plans often mention nuclear atypia as the appearance of the
    # cells at that interface; that incidental word must not turn a margin
    # task into a grade task and discard the boundary tissue.
    if any(term in text for term in ("margin", "resection", "cut edge", "inked", "dissection surface")):
        interfaces.extend((item, adipose) for item in tumor)
        interfaces.extend((item, stroma) for item in tumor)
        policy = "conditional"
        cellular_targets = ["neoplastic_like", "interface_confirmation"]
        reason = "tumor-to-cut-surface interface search"
    elif any(term in text for term in ("capsule", "capsular", "fibrous boundary", "outer fibrous")):
        interfaces.extend((item, adipose) for item in tumor)
        if not renal_perirenal_fat:
            interfaces.extend((item, stroma) for item in tumor)
        # The renal fibrous capsule is an outer surface.  On partial and
        # radical nephrectomy sections it is often a thin collagenous strip
        # between tumour and glass/background, while the prototype map may
        # classify the strip itself as either tumour-associated stroma or
        # unknown.  Reserving the tumour-to-background edge therefore
        # localizes the actual capsule instead of spending every high-power
        # field on intratumoral fibrous septa.  This remains navigation only;
        # a positive answer still requires directly observed cellular entry
        # into a fibrous boundary in the controller.
        if renal and not renal_perirenal_fat:
            interfaces.extend((item, "artifact_or_background") for item in tumor)
        policy = "conditional"
        cellular_targets = ["neoplastic_like", "interface_confirmation"]
        reason = (
            "tumor-perirenal-fat interface search"
            if renal_perirenal_fat
            else "tumor-capsule interface search"
        )
    elif any(term in text for term in ("perineural", "nerve sheath", "neural invasion")):
        interfaces.extend((item, stroma) for item in tumor)
        policy = "required"
        cellular_targets = ["neoplastic_like", "spatial_contact"]
        reason = "tumor-nerve spatial-relation search"
    elif any(term in text for term in ("lymphovascular", "vascular invasion", "tumor embol", "endothelial-lined")):
        interfaces.extend((item, stroma) for item in tumor)
        policy = "required"
        cellular_targets = ["neoplastic_like", "endothelial", "spatial_contact"]
        reason = "tumor-vessel spatial-relation search"
    elif any(term in text for term in ("lymph node", "nodal", "metastatic", "metastasis", "pn")):
        targets.extend([immune])
        interfaces.extend((item, immune) for item in tumor)
        policy = "required"
        cellular_targets = ["neoplastic_like", "immune", "epithelial_cluster"]
        reason = "nodal epithelial-deposit search"
    elif any(
        term in text
        for term in (
            "omental nodule",
            "separately submitted renal cyst",
            "mature adipose",
            "cyst wall",
        )
    ):
        targets.extend([adipose, stroma, necrosis])
        policy = "conditional"
        cellular_targets = ["stromal_context", "immune", "epithelial"]
        reason = "adipose-fibrous specimen-composition search"
    # Gastric heterogeneity plans deliberately mention the invasive front in
    # their retrieval query.  Classify their explicit architecture/composition
    # intent before the generic depth branch, otherwise Lauren and
    # differentiation questions are incorrectly routed to boundary sampling.
    elif any(
        term in text
        for term in (
            "lauren",
            "differentiation",
            "histological type",
            "histological category",
            "tumor histology",
            "pathological type",
            "tumor type",
            "tumor typing",
            "subtype",
            "heterogeneity",
            "gland formation",
            "least cohesive",
            "least differentiated",
        )
    ):
        policy = "conditional"
        cellular_targets = ["neoplastic_like", "epithelial", "stromal_context"]
        reason = "tumor architecture and cellular-composition search"
    elif "necro" in text:
        targets.append(necrosis)
        interfaces.extend((item, necrosis) for item in tumor)
        policy = "conditional"
        cellular_targets = ["neoplastic_like", "apoptotic_or_mitotic_events"]
        reason = "tumor-necrosis co-localization"
    elif any(term in text for term in ("grade", "nuclear", "nucleol", "mitotic", "pleomorph", "rhabdoid", "sarcomatoid")):
        policy = "required"
        cellular_targets = ["neoplastic_like", "nuclear_heterogeneity", "cellular_events"]
        reason = "cellular or nuclear extremum search"
    elif any(term in text for term in ("deepest", "pt", "invasive front", "wall compartment")):
        interfaces.extend((item, adipose) for item in tumor)
        interfaces.extend((item, stroma) for item in tumor)
        if not renal:
            interfaces.extend((item, "muscularis") for item in tumor)
            # Gastric serosa is often a very thin mesothelial boundary between
            # tumour-bearing fibroadipose tissue and glass/background.  The
            # tissue map is used only to navigate to that exterior interface;
            # controller-side morphology must still prove true serosal reach
            # and rejects cut, resection, cautery, and dissection surfaces.
            interfaces.extend((item, "artifact_or_background") for item in tumor)
        policy = "conditional"
        cellular_targets = ["neoplastic_like", "interface_confirmation"]
        reason = "tumor-boundary or depth interface search"
    elif any(term in text for term in ("vessel", "vascular channel")):
        interfaces.extend((item, stroma) for item in tumor)
        policy = "required"
        cellular_targets = ["neoplastic_like", "endothelial", "spatial_contact"]
        reason = "microinvasion spatial-relation search"
    elif any(term in text for term in ("immune", "lymphocyte", "til", "plasma", "neutrophil", "eosinophil", "macrophage")):
        targets.append(immune)
        policy = "required"
        cellular_targets = ["immune", "immune_subtypes"]
        reason = "tumor-microenvironment cellular analysis"

    return TissueRoutingPolicy(
        target_labels=tuple(dict.fromkeys(targets)),
        interface_pairs=tuple(dict.fromkeys(interfaces)),
        histoplus_policy=policy,
        histoplus_targets=tuple(cellular_targets),
        reason=reason,
    )
