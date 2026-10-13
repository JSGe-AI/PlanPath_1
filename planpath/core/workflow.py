"""Compact deterministic v6 state machine.

Qwen chooses the plan and diagnoses the evidence gap. Python always performs
the next concrete operation, so there is no free-form action loop.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .config import AgentConfig
from .controller import (
    _lymph_node_metastasis_support_ids,
    is_gastric_extent_question,
    is_microinvasion_question,
    is_renal_capsule_question,
    is_renal_perirenal_fat_question,
    is_regional_node_question,
    is_spatial_interface_question,
    normalize_plan_for_question,
    question_specific_focus_roi_ids,
    question_spatial_positive_support_ids,
)
from .precompute_cache import PrecomputedRetrievalCache
from .schemas import EvidencePlan, EvidenceReview, FinalAnswer, ROI, ROIObservation, SlideScore
from .tissue_routing import routing_policy


METHOD = "PlanPath Core"
Checkpoint = Callable[[dict[str, Any]], None]


class ControllerProtocol(Protocol):
    audit: list[dict[str, Any]]

    def plan(self, question: dict[str, Any]) -> EvidencePlan: ...

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
    ) -> EvidenceReview: ...

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
    ) -> FinalAnswer: ...


class RetrieverProtocol(Protocol):
    metrics: dict[str, int]
    last_skipped_slide_ids: list[str]

    def rank(
        self, slides: list[dict[str, Any]], descriptions: list[str]
    ) -> list[SlideScore]: ...


class VisualProtocol(Protocol):
    metrics: dict[str, int]
    audit: list[dict[str, Any]]

    def low_power(
        self, slide: dict[str, Any], plan: EvidencePlan, evidence_start: int
    ) -> tuple[list[ROI], list[ROIObservation]]: ...

    def zoom(
        self,
        slide: dict[str, Any],
        parents: list[ROI],
        magnification: str,
        plan: EvidencePlan,
        evidence_start: int,
    ) -> tuple[list[ROI], list[ROIObservation]]: ...


class PlanPathCore:
    def __init__(
        self,
        controller: ControllerProtocol,
        retriever: RetrieverProtocol,
        visual: VisualProtocol,
        config: AgentConfig,
        retrieval_cache: PrecomputedRetrievalCache | None = None,
    ) -> None:
        self.controller = controller
        self.retriever = retriever
        self.visual = visual
        self.config = config
        self.retrieval_cache = retrieval_cache
        self._complete_visual_scan_cache_patient_id: str | None = None
        self._complete_visual_scan_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _accelerated_special_question(question: dict[str, Any]) -> bool:
        return bool(
            is_gastric_extent_question(question)
            or is_microinvasion_question(question)
            or is_renal_capsule_question(question)
            or is_regional_node_question(question)
        )

    @staticmethod
    def _complete_visual_scan_family(question: dict[str, Any]) -> str | None:
        """Return a family whose questions consume the same neutral visual scan.

        Deepest-layer and gastric pT questions are normalized to one invasive-
        front search.  Regional-node presence and pN questions likewise share
        one label-free nodal-deposit search.  Other superficially related
        questions are intentionally excluded because their ROI queries differ.
        """

        if is_gastric_extent_question(question):
            return "gastric_extent"
        if is_regional_node_question(question):
            return "regional_node"
        return None

    @staticmethod
    def _complete_visual_scan_signature(
        plan: EvidencePlan, family: str
    ) -> tuple[Any, ...]:
        """Describe inputs that can change the selected visual evidence.

        Regional-node presence and pN differ in decision semantics and stop
        policy, but their normalized retrieval/routing/zoom inputs are the
        same.  Those answer-side fields are therefore deliberately omitted.
        """

        return (
            family,
            tuple(plan.slide_retrieval_descriptions),
            plan.low_power_query,
            plan.high_power_query,
            tuple(plan.preferred_zoom_magnifications),
            plan.required_magnification,
        )

    def _prepare_complete_visual_scan_cache(self, patient_id: str) -> None:
        if patient_id == self._complete_visual_scan_cache_patient_id:
            return
        self._complete_visual_scan_cache_patient_id = patient_id
        self._complete_visual_scan_cache.clear()

    def _accelerated_plan(
        self, plan: EvidencePlan, question: dict[str, Any]
    ) -> EvidencePlan:
        """Use 20x as a floor when it can resolve the requested relation.

        Perirenal-fat involvement remains a true high-power endpoint.  For the
        fibrous capsule itself, the controller already accepts a verified 20x
        entry relation; keeping a hard 40x floor only repeated that relation.
        Regional-node morphology is likewise directly assessable at 20x; a
        suspicious small deposit may still explicitly request 40x.  The nodal
        relaxation has its own flag so the validated 6.2.59 behavior remains
        reproducible.
        """

        if not self.config.accelerated_evaluation_enabled:
            return plan
        renal_capsule_floor = bool(
            is_renal_capsule_question(question)
            and not is_renal_perirenal_fat_question(question)
        )
        regional_node_floor = bool(
            self.config.accelerated_regional_node_20x_floor_enabled
            and is_regional_node_question(question)
        )
        if not renal_capsule_floor and not regional_node_floor:
            return plan
        preferred = list(plan.preferred_zoom_magnifications)
        if "20x" not in preferred:
            preferred.append("20x")
        if "40x" not in preferred:
            preferred.append("40x")
        preferred.sort(key=lambda value: int(value.rstrip("x")))
        return plan.model_copy(
            update={
                "required_magnification": "20x",
                "preferred_zoom_magnifications": preferred,
            }
        )

    def _promote_slide_from_low_power(
        self,
        question: dict[str, Any],
        slide_id: str,
        titan_rank: int,
        low_rois: list[ROI],
        low_observations: list[ROIObservation],
    ) -> tuple[bool, str]:
        """Keep full low-power coverage but bound expensive high-power work."""

        if not self.config.accelerated_evaluation_enabled:
            return True, "acceleration_disabled"
        if not self._accelerated_special_question(question):
            return True, "ordinary_question"
        if titan_rank <= self.config.accelerated_slide_promotion_top_k:
            return True, "top_rank_recall_floor"
        focus_ids = question_specific_focus_roi_ids(
            question,
            low_observations,
            slide_id,
            [item.roi_id for item in low_rois],
        )
        if focus_ids:
            return True, "question_relevant_low_power_morphology"
        return False, "low_power_only_no_question_relevant_morphology"

    @staticmethod
    def clean_question(
        question: dict[str, Any], clinical_domain: str | None = None
    ) -> dict[str, Any]:
        """Keep only fields that are allowed to reach inference prompts."""

        value = {
            "patient_id": str(question["patient_id"]),
            "qa_id": str(question["qa_id"]),
            "question_type": str(question["question_type"]),
            "question": str(question["question"]),
            "options": [str(item) for item in question.get("options", [])],
        }
        if clinical_domain:
            value["clinical_domain"] = str(clinical_domain)
        return value

    def _stop_allowed(
        self,
        plan: EvidencePlan,
        review: EvidenceReview,
        observations: list[ROIObservation],
        remaining_slide_ids: list[str],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if review.status != "sufficient":
            return False, [f"review status is {review.status}"]
        by_id = {item.evidence_id: item for item in observations}
        supporting = [
            by_id[evidence_id]
            for evidence_id in review.supporting_evidence_ids
            if evidence_id in by_id
        ]
        if len(supporting) != len(review.supporting_evidence_ids) or not supporting:
            reasons.append("supporting evidence IDs are missing")
        if any(item.image_quality == "inadequate" for item in supporting):
            reasons.append("inadequate image cannot support stopping")
        if any(
            item.morphology
            == "No valid neutral morphology transcription was returned."
            for item in supporting
        ):
            reasons.append("unavailable neutral transcription cannot support stopping")
        if review.conclusion_direction == "indeterminate":
            reasons.append("an indeterminate conclusion cannot be sufficient")
        judgments = {item.evidence_id: item for item in review.evidence_judgments}
        direction = review.conclusion_direction
        direct = [
            item
            for item in supporting
            if item.evidence_id in judgments
            and judgments[item.evidence_id].relevance == "direct"
            and judgments[item.evidence_id].direction == direction
        ]
        required = int(plan.required_magnification.rstrip("x"))
        direct_at_required = [
            item
            for item in direct
            if int(item.magnification.rstrip("x")) >= required
        ]
        required_direct_count = (
            1
            if plan.decision_semantics == "extremum"
            else self.config.min_direct_evidence
        )
        if len(direct_at_required) < required_direct_count:
            reasons.append(
                "conclusion lacks enough direct question-side evidence judgments at "
                f"required magnification {plan.required_magnification}"
            )
        if direction == "negative":
            if remaining_slide_ids:
                reasons.append("negative conclusion requires candidate-pool coverage")
        if plan.stop_policy == "candidate_pool_coverage" and remaining_slide_ids:
            reasons.append("the plan requires inspection of all candidate slides")
        return not reasons, reasons

    @staticmethod
    def _review_dump(review: EvidenceReview, allowed: bool, reasons: list[str]) -> dict[str, Any]:
        value = review.model_dump(mode="json")
        value["stop_gate_allowed"] = allowed
        value["stop_gate_reasons"] = reasons
        return value

    @staticmethod
    def _final_evidence_subset(
        observations: list[ROIObservation],
        reviews: list[EvidenceReview],
        required_magnification: str,
        decision_semantics: str = "exhaustive",
    ) -> tuple[list[ROIObservation], dict[str, Any]]:
        """Keep question-side relevant evidence and its low-power lineage."""

        by_id = {item.evidence_id: item for item in observations}
        by_roi = {item.roi_id: item for item in observations}
        required = int(required_magnification.rstrip("x"))
        selected_ids: list[str] = []
        lineage_ids: list[str] = []
        for review in reviews:
            for judgment in review.evidence_judgments:
                source = by_id.get(judgment.evidence_id)
                supporting_floor = min(required, 20)
                usable = judgment.relevance == "direct" or (
                    judgment.relevance == "supporting"
                    and source is not None
                    and int(source.magnification.rstrip("x")) >= supporting_floor
                )
                if usable:
                    if judgment.evidence_id not in selected_ids:
                        selected_ids.append(judgment.evidence_id)
                    if source and source.parent_roi_id:
                        parent = by_roi.get(source.parent_roi_id)
                        if parent and parent.evidence_id not in lineage_ids:
                            lineage_ids.append(parent.evidence_id)
        decisive_ids: list[str] = []
        decisive_lineage_ids: list[str] = []
        focus_evidence_ids: list[str] = []
        for review in reviews:
            for roi_id in review.focus_roi_ids:
                source = by_roi.get(roi_id)
                if source and source.evidence_id not in focus_evidence_ids:
                    focus_evidence_ids.append(source.evidence_id)
        for review in reversed(reviews):
            if review.status != "sufficient":
                continue
            for evidence_id in review.supporting_evidence_ids:
                if evidence_id in by_id and evidence_id not in decisive_ids:
                    decisive_ids.append(evidence_id)
                    source = by_id[evidence_id]
                    if source.parent_roi_id:
                        parent = by_roi.get(source.parent_roi_id)
                        if parent and parent.evidence_id not in decisive_lineage_ids:
                            decisive_lineage_ids.append(parent.evidence_id)
        fallback_used = not selected_ids
        if fallback_used:
            selected = [
                item for item in observations if item.image_quality != "inadequate"
            ]
        else:
            selected = [by_id[item] for item in selected_ids if item in by_id]
        # Keep the evidence prompt bounded, but pin the most recent sufficient
        # review's decisive evidence before filling from the broader history.
        fallback_ids = [item.evidence_id for item in selected] if fallback_used else []
        priority_ids = list(
            dict.fromkeys(
                decisive_ids
                + focus_evidence_ids
                + selected_ids
                + decisive_lineage_ids
                + lineage_ids
                + fallback_ids
            )
        )
        slide_order = list(
            dict.fromkeys(
                item.slide_id for item in observations if item.evidence_id in priority_ids
            )
        )
        by_slide: dict[str, list[str]] = {
            slide_id: [
                evidence_id
                for evidence_id in priority_ids
                if by_id[evidence_id].slide_id == slide_id
            ]
            for slide_id in slide_order
        }
        evidence_limit = 40 if decision_semantics == "extremum" else 15
        prioritized_ids: list[str] = []
        depth = 0
        while len(prioritized_ids) < evidence_limit and any(
            depth < len(items) for items in by_slide.values()
        ):
            for slide_id in slide_order:
                items = by_slide[slide_id]
                if depth < len(items):
                    prioritized_ids.append(items[depth])
                    if len(prioritized_ids) == evidence_limit:
                        break
            depth += 1
        selected_set = set(prioritized_ids)
        ordered = [item for item in observations if item.evidence_id in selected_set]
        return ordered, {
            "strategy": "direct_or_resolution_qualified_supporting_judgments_plus_parent",
            "fallback_to_all_usable": fallback_used,
            "selected_evidence_ids": [item.evidence_id for item in ordered],
            "selected_count": len(ordered),
            "total_observation_count": len(observations),
        }

    def run(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        checkpoint: Checkpoint | None = None,
    ) -> dict[str, Any]:
        clean_question = self.clean_question(question, self.config.clinical_domain)
        controller_audit_start = len(self.controller.audit)
        visual_audit_start = len(self.visual.audit)
        plan_source = "generated"
        plan_cache_path: str | None = None
        plan = (
            self.retrieval_cache.load_plan(clean_question)
            if self.retrieval_cache
            else None
        )
        if plan is None:
            plan_audit_start = len(self.controller.audit)
            plan = self.controller.plan(clean_question)
            if self.retrieval_cache:
                plan_cache_path = str(
                    self.retrieval_cache.save_plan(
                        clean_question,
                        plan,
                        self.controller.audit[plan_audit_start:],
                    )
                )
        else:
            plan_source = "persistent_cache"
            plan_cache_path = str(self.retrieval_cache.plan_path(clean_question))
        plan = normalize_plan_for_question(plan, clean_question)
        plan = self._accelerated_plan(plan, clean_question)
        cancer = str(slides[0].get("cancer") or "unknown") if slides else "unknown"
        modality_policy = routing_policy(cancer, plan)

        ranking_source = "computed"
        ranking_cache_path: str | None = None
        cached_ranking = (
            self.retrieval_cache.load_ranking(clean_question, slides, plan)
            if self.retrieval_cache
            else None
        )
        if cached_ranking is None:
            ranked = self.retriever.rank(slides, plan.slide_retrieval_descriptions)
            unavailable_slide_ids = list(self.retriever.last_skipped_slide_ids)
            if self.retrieval_cache:
                ranking_cache_path = str(
                    self.retrieval_cache.save_ranking(
                        clean_question,
                        slides,
                        plan,
                        ranked,
                        unavailable_slide_ids,
                    )
                )
        else:
            ranked, unavailable_slide_ids, cached_path = cached_ranking
            ranking_source = "persistent_cache"
            ranking_cache_path = str(cached_path)
        unavailable_set = set(unavailable_slide_ids)
        candidate_limit = self.config.top_k_slides
        if (
            self.config.accelerated_protect_complex_and_interface_slide_pool_enabled
            and (
                modality_policy.interface_pairs
                or plan.decision_semantics in {"extremum", "quantitative", "composite"}
            )
        ):
            # A top-three candidate is only for ordinary tumour-bearing-tissue
            # searches.  Preserve the validated five-slide floor for focal
            # interfaces and aggregate/extremum decisions even when a wording
            # variant is not recognized by a question-text helper.
            candidate_limit = max(
                candidate_limit,
                self.config.accelerated_protected_roi_floor,
            )
        if plan.decision_semantics == "extremum":
            candidate_limit = max(candidate_limit, 8)
        if is_gastric_extent_question(clean_question):
            # Deepest invasion and pT are patient-level extrema.  A bounded
            # top-eight pool can establish adipose invasion yet omit the only
            # section containing the true serosal/free surface, systematically
            # collapsing pT4 to pT3.  Retain the complete label-free ranking.
            candidate_limit = len(ranked)
        if is_microinvasion_question(clean_question):
            # PNI and LVI are minute, spatially focal findings.  The former
            # top-eight slide pool missed every direct LVI relation in a
            # twenty-slide positive case even after exhaustive 40x descent,
            # and treating a truncated top-twelve pool as ``exhausted`` made
            # the subsequent patient-level negative logically invalid. Keep
            # the complete label-free ranking for these questions. Positives
            # can still stop early, but only with an explicit same-clause
            # cell/structure relationship.
            candidate_limit = len(ranked)
        if is_renal_capsule_question(clean_question):
            # A negative capsule call is a whole-specimen absence claim. The
            # previous eight-slide ceiling could omit the only block carrying
            # the outer tumour/capsule interface, yet still report the bounded
            # subset as exhausted. Preserve every label-free ranked slide;
            # a verified positive may still stop early.
            candidate_limit = len(ranked)
        if is_spatial_interface_question(clean_question):
            # Focal interfaces are easy to miss in a slide-level top-5 pool.
            # Eight remains bounded but materially improves coverage for a
            # negative patient-level relationship claim.
            candidate_limit = max(candidate_limit, 8)
        if is_regional_node_question(clean_question):
            # A negative nodal conclusion or a pN category is a patient-level
            # claim over submitted nodes. A fixed top-k can omit the only
            # involved node, so retain the complete label-free ranking. An
            # existential positive may still stop early through the evidence
            # gate once valid nodal morphology is found.
            candidate_limit = len(ranked)
        candidate_scores = [
            item for item in ranked if item.slide_id not in unavailable_set
        ][:candidate_limit]
        candidate_ids = [item.slide_id for item in candidate_scores]
        slide_by_id = {str(slide["slide_id"]): slide for slide in slides}
        trace: dict[str, Any] = {
            "method": self.config.method or METHOD,
            "schema_version": "1.0",
            "patient_id": clean_question["patient_id"],
            "qa_id": clean_question["qa_id"],
            "question": clean_question,
            "reference_answer_used_in_prompt": False,
            "site_labels_used": False,
            "cohort_domain_context": self.config.clinical_domain,
            "patho_question_conditioning": False,
            "neutral_roi_observations": True,
            "conch_tissue_maps_are_diagnostic_evidence": False,
            "histoplus_cell_types_are_direct_diagnostic_evidence": False,
            "config": self.config.as_dict(),
            "plan": plan.model_dump(mode="json"),
            "modality_routing": {
                "target_tissue_classes": list(modality_policy.target_labels),
                "preferred_interfaces": [list(item) for item in modality_policy.interface_pairs],
                "histoplus_policy": modality_policy.histoplus_policy,
                "histoplus_targets": list(modality_policy.histoplus_targets),
                "reason": modality_policy.reason,
                "navigation_outputs_are_diagnostic_evidence": False,
            },
            "precomputed_retrieval": {
                "enabled": self.retrieval_cache is not None,
                "plan_source": plan_source,
                "plan_cache_path": plan_cache_path,
                "ranking_source": ranking_source,
                "ranking_cache_path": ranking_cache_path,
            },
            "titan_retrieval": {
                "aggregation": "maximum cosine similarity across descriptions",
                "candidate_limit": candidate_limit,
                "candidate_pool": [item.model_dump(mode="json") for item in candidate_scores],
                "complete_ranking": [item.model_dump(mode="json") for item in ranked],
                "ranked_slide_count": len(ranked),
                "analysis_eligible_slide_count": len(ranked) - len(
                    set(unavailable_slide_ids) & {item.slide_id for item in ranked}
                ),
                "ranked_but_outside_top_k": max(
                    0,
                    len(ranked)
                    - len(set(unavailable_slide_ids) & {item.slide_id for item in ranked})
                    - len(candidate_scores),
                ),
                "unavailable_slide_ids": unavailable_slide_ids,
                "scores_are_diagnostic_evidence": False,
            },
            "slide_rounds": [],
            "acceleration": {
                "enabled": self.config.accelerated_evaluation_enabled,
                "patient_visual_cache_enabled": self.config.patient_visual_cache_enabled,
                "skip_low_power_review_enabled": self.config.accelerated_skip_low_power_review_enabled,
                "skip_non_actionable_extent_review_enabled": self.config.accelerated_skip_non_actionable_extent_review_enabled,
                "reuse_complete_visual_scan_enabled": self.config.accelerated_reuse_complete_visual_scan_enabled,
                "protect_complex_and_interface_slide_pool_enabled": self.config.accelerated_protect_complex_and_interface_slide_pool_enabled,
                "protected_roi_floor": self.config.accelerated_protected_roi_floor,
                "regional_node_20x_floor_enabled": self.config.accelerated_regional_node_20x_floor_enabled,
                "regional_node_20x_top_k": self.config.accelerated_regional_node_20x_top_k,
                "special_low_power_top_k": self.config.accelerated_special_low_power_top_k,
                "special_20x_top_k": self.config.accelerated_special_20x_top_k,
                "special_40x_top_k": self.config.accelerated_special_40x_top_k,
                "slide_promotion_top_k": self.config.accelerated_slide_promotion_top_k,
            },
            "status": "running",
        }

        def save() -> None:
            if checkpoint:
                checkpoint(trace)

        save()
        observations: list[ROIObservation] = []
        evidence_next = 1
        evidence_sufficient = False
        last_review: EvidenceReview | None = None
        review_history: list[EvidenceReview] = []
        authorized_answer_candidate: str | None = None
        termination_reason = "candidate_slides_exhausted"
        failed_slide_ids: list[str] = []

        expected_candidate_ids = list(candidate_ids)
        scan_family = self._complete_visual_scan_family(clean_question)
        scan_signature = (
            self._complete_visual_scan_signature(plan, scan_family)
            if scan_family is not None
            else None
        )
        reused_complete_visual_scan = False
        if self.config.accelerated_reuse_complete_visual_scan_enabled:
            patient_id = str(clean_question["patient_id"])
            self._prepare_complete_visual_scan_cache(patient_id)
            cached_scan = (
                self._complete_visual_scan_cache.get(scan_family)
                if scan_family is not None
                else None
            )
            if (
                cached_scan is not None
                and cached_scan["signature"] == scan_signature
                and cached_scan["candidate_ids"] == expected_candidate_ids
            ):
                # ROI observations are question-independent neutral morphology.
                # Evidence IDs and parent lineage are retained exactly, while
                # the new question still receives its own deterministic
                # aggregation and final answer generation below.
                observations = list(cached_scan["observations"])
                evidence_next = len(observations) + 1
                candidate_ids = []
                reused_complete_visual_scan = True
                trace["acceleration"]["complete_visual_scan_reuse"] = {
                    "reused": True,
                    "family": scan_family,
                    "source_task_key": cached_scan["source_task_key"],
                    "candidate_slide_count": len(expected_candidate_ids),
                    "observation_count": len(observations),
                    "eligibility": (
                        "source exhausted the identical complete candidate pool "
                        "without slide failures"
                    ),
                }
                save()

        for slide_index, slide_id in enumerate(candidate_ids):
            slide = slide_by_id[slide_id]
            remaining = candidate_ids[slide_index + 1 :]
            slide_round: dict[str, Any] = {
                "slide_id": slide_id,
                "titan_rank": slide_index + 1,
                "low_power": None,
                "zoom_rounds": [],
                "reviews": [],
            }
            trace["slide_rounds"].append(slide_round)
            try:
                low_rois, low_observations = self.visual.low_power(
                    slide, plan, evidence_next
                )
            except Exception as error:
                slide_round["error"] = f"{type(error).__name__}: {error}"
                failed_slide_ids.append(slide_id)
                save()
                continue
            evidence_next += len(low_observations)
            observations.extend(low_observations)
            all_rois: dict[str, ROI] = {item.roi_id: item for item in low_rois}
            current_roi_ids = [item.roi_id for item in low_rois]
            attempted = ["5x"]
            slide_round["low_power"] = {
                "rois": [item.model_dump(mode="json") for item in low_rois],
                "observations": [item.model_dump(mode="json") for item in low_observations],
            }

            promote_slide, promotion_reason = self._promote_slide_from_low_power(
                clean_question,
                slide_id,
                slide_index + 1,
                low_rois,
                low_observations,
            )
            slide_round["high_power_promotion"] = {
                "promoted": promote_slide,
                "reason": promotion_reason,
            }
            if not promote_slide:
                save()
                continue

            for zoom_round_index in range(self.config.max_zoom_rounds_per_slide + 1):
                required = int(plan.required_magnification.rstrip("x"))
                attempted_max = max(int(item.rstrip("x")) for item in attempted)
                skip_low_power_review = bool(
                    self.config.accelerated_evaluation_enabled
                    and self.config.accelerated_skip_low_power_review_enabled
                    and (
                        self._accelerated_special_question(clean_question)
                        or is_spatial_interface_question(clean_question)
                    )
                    and attempted == ["5x"]
                    and required > 5
                    and "20x" in plan.preferred_zoom_magnifications
                )
                skip_non_actionable_extent_review = bool(
                    self.config.accelerated_evaluation_enabled
                    and self.config.accelerated_skip_non_actionable_extent_review_enabled
                    and is_gastric_extent_question(clean_question)
                    and plan.stop_policy == "candidate_pool_coverage"
                    and attempted_max >= required
                    and not any(
                        int(item.rstrip("x")) > attempted_max
                        for item in plan.preferred_zoom_magnifications
                    )
                )
                review: EvidenceReview | None = None
                if skip_low_power_review:
                    # Accelerated special and spatial-interface questions
                    # retain every routed 5x parent for the first 20x
                    # localization. A Qwen review at 5x therefore cannot change
                    # either the next resolution or its parent coverage. Defer
                    # the first question-side judgment until the unchanged
                    # evidence is available at 20x, where the same review still
                    # sees both magnifications.
                    requested = "20x"
                    slide_round["low_power_review_skipped"] = {
                        "skipped": True,
                        "reason": "all_routed_parents_advance_to_20x_before_review",
                    }
                elif skip_non_actionable_extent_review:
                    # Once the required 20x invasive-front fields have been
                    # collected, a slide-local review cannot change the zoom
                    # path and candidate-pool coverage forbids early stopping.
                    # The final integrator still sees the unchanged complete
                    # observation set and applies the deterministic deepest-
                    # layer/pT mapping across all slides.
                    requested = None
                    slide_round["non_actionable_extent_review_skipped"] = {
                        "skipped": True,
                        "reason": (
                            "required_resolution_reached_and_no_unattempted_zoom; "
                            "candidate_pool_coverage_is_enforced"
                        ),
                    }
                else:
                    review = self.controller.review(
                        clean_question,
                        plan,
                        observations,
                        current_slide_id=slide_id,
                        remaining_candidate_slide_ids=remaining + failed_slide_ids,
                        available_focus_roi_ids=current_roi_ids,
                        attempted_magnifications=attempted,
                    )
                    last_review = review
                    review_history.append(review)
                    allowed, gate_reasons = self._stop_allowed(
                        plan, review, observations, remaining + failed_slide_ids
                    )
                    slide_round["reviews"].append(
                        self._review_dump(review, allowed, gate_reasons)
                    )
                    save()
                    if allowed:
                        evidence_sufficient = True
                        termination_reason = "evidence_sufficient"
                        authorized_answer_candidate = review.answer_candidate
                        break
                    requested = (
                        str(review.next_magnification)
                        if review.status == "need_higher_magnification"
                        else None
                    )
                    current_slide_adequate_ids = {
                        item.evidence_id
                        for item in observations
                        if item.slide_id == slide_id
                        and item.image_quality != "inadequate"
                    }
                    has_question_relevant_current_evidence = any(
                        judgment.relevance in {"direct", "supporting"}
                        for judgment in review.evidence_judgments
                        if judgment.evidence_id in current_slide_adequate_ids
                    )
                    resolution_coverage_required = (
                        plan.decision_semantics == "extremum"
                        or is_regional_node_question(clean_question)
                        or is_spatial_interface_question(clean_question)
                        or has_question_relevant_current_evidence
                    )
                    if (
                        attempted_max < required
                        and resolution_coverage_required
                        and (
                            requested is None
                            or requested in attempted
                            or int(requested.rstrip("x")) <= attempted_max
                        )
                    ):
                        requested = min(
                            (
                                item
                                for item in plan.preferred_zoom_magnifications
                                if attempted_max < int(item.rstrip("x")) <= required
                            ),
                            key=lambda item: int(item.rstrip("x")),
                        )
                        slide_round["forced_zoom_to_required_magnification"] = requested
                if requested is None:
                    break
                if zoom_round_index >= self.config.max_zoom_rounds_per_slide:
                    slide_round["zoom_limit_reached"] = True
                    break
                if requested in attempted or int(requested.rstrip("x")) <= max(
                    int(item.rstrip("x")) for item in attempted
                ):
                    slide_round["invalid_zoom_request"] = (
                        f"magnification is not above attempted levels: {requested}"
                    )
                    break
                preferred_focus_ids = question_specific_focus_roi_ids(
                    clean_question,
                    observations,
                    slide_id,
                    current_roi_ids,
                )
                parent_ids = [
                    roi_id
                    for roi_id in preferred_focus_ids[:2]
                    if roi_id in all_rois
                ]
                parent_ids.extend(
                    roi_id
                    for roi_id in (review.focus_roi_ids[:2] if review else [])
                    if roi_id in all_rois and roi_id not in parent_ids
                )
                parent_ids = parent_ids[:2]
                if preferred_focus_ids and parent_ids != [
                    roi_id
                    for roi_id in (review.focus_roi_ids[:2] if review else [])
                    if roi_id in all_rois
                ]:
                    slide_round["question_specific_focus_override_used"] = True
                    slide_round["question_specific_focus_roi_ids"] = parent_ids
                accelerated_special = (
                    self.config.accelerated_evaluation_enabled
                    and self._accelerated_special_question(clean_question)
                )
                if accelerated_special:
                    configured_parent_count = (
                        self.config.accelerated_special_40x_parent_top_k
                        if requested == "40x"
                        else self.config.accelerated_special_20x_parent_top_k
                    )
                    required_parent_count = min(
                        configured_parent_count, len(current_roi_ids)
                    )
                elif is_microinvasion_question(clean_question):
                    # Preserve all twelve low-power focal-search parents at
                    # 20x. The selector now retains two child fields per
                    # parent; at 40x these twelve parents are re-ranked so the
                    # structure-bearing localizations receive a second crop.
                    required_parent_count = min(12, len(current_roi_ids))
                elif is_renal_capsule_question(clean_question):
                    # Capsule routing deliberately reserves twelve low-power
                    # fields (including peripheral-surface fields).  The
                    # generic spatial cap of eight silently discarded two of
                    # the former ten already localized parents before the first 20x
                    # pass.  In real positive cases the dropped fields were
                    # often the highest-scoring tumour/fibrous interfaces.
                    # Preserve all twelve parents for 5x -> 20x, where the
                    # selector spends the fixed 24-observation budget on two
                    # spatially separated children per parent.  The resulting
                    # 24 localized 20x parents must *all* survive 20x -> 40x;
                    # the same fixed budget then assigns one child per parent.
                    # Capping both transitions at twelve silently discarded
                    # half of the localized interfaces immediately before the
                    # required-resolution review.
                    required_parent_count = min(24, len(current_roi_ids))
                elif is_spatial_interface_question(clean_question):
                    required_parent_count = min(8, len(current_roi_ids))
                elif is_gastric_extent_question(clean_question):
                    # The outer/free surface is sparse even within a relevant
                    # section. Preserve all twelve tissue-routed parents so a
                    # later zoom cannot silently discard the serosal field.
                    required_parent_count = min(12, len(current_roi_ids))
                elif plan.decision_semantics == "extremum":
                    required_parent_count = min(8, len(current_roi_ids))
                elif is_regional_node_question(clean_question):
                    # Nodal deposits may occupy a tiny subcapsular focus while
                    # most of a lymph-node section is reactive. Preserve all
                    # twelve routed parents at both zoom levels; limiting the
                    # complete slide pool to five parents still collapsed
                    # high pN categories after the relevant slides were found.
                    required_parent_count = min(12, len(current_roi_ids))
                else:
                    required_parent_count = self.config.min_zoom_parent_rois
                if len(parent_ids) < required_parent_count:
                    ranked_fallbacks = sorted(
                        (
                            all_rois[roi_id]
                            for roi_id in current_roi_ids
                            if roi_id in all_rois and roi_id not in parent_ids
                        ),
                        key=lambda item: item.retrieval_score,
                        reverse=True,
                    )
                    needed = required_parent_count - len(parent_ids)
                    parent_ids.extend(item.roi_id for item in ranked_fallbacks[:needed])
                    if ranked_fallbacks[:needed]:
                        slide_round["focus_roi_diversity_fallback_used"] = True
                        if plan.decision_semantics == "extremum":
                            slide_round["extremum_parent_coverage_used"] = True
                        if is_regional_node_question(clean_question):
                            slide_round["regional_node_parent_coverage_used"] = True
                        if is_spatial_interface_question(clean_question):
                            slide_round["interface_parent_coverage_used"] = True
                if not parent_ids:
                    parent_ids = [
                        roi_id for roi_id in current_roi_ids[-2:] if roi_id in all_rois
                    ]
                    slide_round["focus_roi_fallback_used"] = True
                parents = [all_rois[roi_id] for roi_id in parent_ids]
                if not parents:
                    slide_round["invalid_zoom_request"] = "no valid focus ROI"
                    break
                zoom_rois, zoom_observations = self.visual.zoom(
                    slide,
                    parents,
                    requested,
                    plan,
                    evidence_next,
                )
                if not zoom_rois:
                    slide_round["invalid_zoom_request"] = "no higher-power candidates"
                    break
                evidence_next += len(zoom_observations)
                observations.extend(zoom_observations)
                all_rois.update({item.roi_id: item for item in zoom_rois})
                # The next zoom may only descend from the newest localization
                # level. Retaining old 5x IDs here allowed a later 40x request
                # to jump back to broad normal-tissue fields after 20x had
                # already localized a more relevant subregion.
                current_roi_ids = [item.roi_id for item in zoom_rois]
                attempted.append(requested)
                slide_round["zoom_rounds"].append(
                    {
                        "magnification": requested,
                        "parent_roi_ids": parent_ids,
                        "rois": [item.model_dump(mode="json") for item in zoom_rois],
                        "observations": [
                            item.model_dump(mode="json") for item in zoom_observations
                        ],
                    }
                )
                save()
            if evidence_sufficient:
                break

        if (
            self.config.accelerated_reuse_complete_visual_scan_enabled
            and not reused_complete_visual_scan
            and scan_family is not None
            and not failed_slide_ids
            and len(trace["slide_rounds"]) == len(expected_candidate_ids)
        ):
            self._complete_visual_scan_cache[scan_family] = {
                "signature": scan_signature,
                "candidate_ids": expected_candidate_ids,
                "observations": list(observations),
                "source_task_key": (
                    f"{clean_question['patient_id']}/{clean_question['qa_id']}"
                ),
            }
            trace["acceleration"]["complete_visual_scan_cache_write"] = {
                "family": scan_family,
                "candidate_slide_count": len(expected_candidate_ids),
                "observation_count": len(observations),
            }
            save()

        final_observations, final_evidence_selection = self._final_evidence_subset(
            observations,
            review_history,
            plan.required_magnification,
            plan.decision_semantics,
        )
        spatial_support_ids = question_spatial_positive_support_ids(
            clean_question, observations
        )
        regional_node_support_ids = _lymph_node_metastasis_support_ids(
            clean_question, observations
        )
        final_positive_support_ids = list(
            dict.fromkeys([*spatial_support_ids, *regional_node_support_ids])
        )
        if final_positive_support_ids:
            by_evidence_id = {item.evidence_id: item for item in observations}
            by_roi_id = {item.roi_id: item for item in observations}
            pinned_ids = list(final_positive_support_ids)
            for evidence_id in final_positive_support_ids:
                source = by_evidence_id.get(evidence_id)
                if source is None or not source.parent_roi_id:
                    continue
                parent = by_roi_id.get(source.parent_roi_id)
                if parent is not None:
                    pinned_ids.append(parent.evidence_id)
            selected_ids = {
                item.evidence_id for item in final_observations
            }
            final_observations = [
                *final_observations,
                *(
                    by_evidence_id[evidence_id]
                    for evidence_id in dict.fromkeys(pinned_ids)
                    if evidence_id in by_evidence_id
                    and evidence_id not in selected_ids
                ),
            ]
            final_evidence_selection["spatial_positive_pinned_evidence_ids"] = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in pinned_ids
                    if evidence_id in spatial_support_ids
                )
            )
            final_evidence_selection["regional_node_positive_pinned_evidence_ids"] = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in pinned_ids
                    if evidence_id in regional_node_support_ids
                )
            )
            final_evidence_selection["selected_evidence_ids"] = [
                item.evidence_id for item in final_observations
            ]
            final_evidence_selection["selected_count"] = len(final_observations)
        final = self.controller.final(
            clean_question,
            plan,
            final_observations,
            termination_reason=termination_reason,
            evidence_sufficient=evidence_sufficient,
            last_review=last_review,
            review_history=review_history,
            authorized_answer_candidate=authorized_answer_candidate,
            all_observations=observations,
        )
        trace.update(
            status="success",
            termination_reason=termination_reason,
            final_answer=final.model_dump(mode="json"),
            final_evidence_selection=final_evidence_selection,
            evidence_summary={
                "observation_count": len(observations),
                "visited_slide_ids": [
                    item["slide_id"] for item in trace["slide_rounds"]
                ],
                "analyzed_slide_ids": [
                    item["slide_id"]
                    for item in trace["slide_rounds"]
                    if item.get("low_power") is not None
                ],
                "failed_slide_ids": failed_slide_ids,
                "candidate_slide_ids": candidate_ids,
            },
            metrics={
                "titan": dict(self.retriever.metrics),
                "precomputed_retrieval_cache": (
                    dict(self.retrieval_cache.metrics)
                    if self.retrieval_cache
                    else {}
                ),
                "visual": dict(self.visual.metrics),
                "qwen_generation_audit": self.controller.audit[controller_audit_start:],
                "patho_generation_audit": self.visual.audit[visual_audit_start:],
            },
        )
        save()
        return trace
