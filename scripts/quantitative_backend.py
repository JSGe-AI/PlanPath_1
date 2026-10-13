"""Internal builder for PlanPath's automatically routed quantitative workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.quantitative.config import AgentConfig  # noqa: E402
from planpath.quantitative.controller import QwenController  # noqa: E402
from planpath.quantitative.histoplus import HistoPLUSAnalyzer  # noqa: E402
from planpath.quantitative.io_utils import read_json  # noqa: E402
from planpath.quantitative.llm_client import OpenAICompatClient  # noqa: E402
from planpath.quantitative.observer import PathoObserver, VisualAnalyzer  # noqa: E402
from planpath.quantitative.plip import LocalPLIPEncoder, ROISelector  # noqa: E402
from planpath.quantitative.precompute_cache import (  # noqa: E402
    PrecomputedRetrievalCache,
)
from planpath.quantitative.structured import StructuredGenerator  # noqa: E402
from planpath.quantitative.titan import (  # noqa: E402
    LazyTitanTextEncoder,
    TitanSlideRetriever,
)
from planpath.quantitative.workflow import PlanPathQuantitativeBackend  # noqa: E402


def defaults_get(config_path: Path, key: str, default: Any) -> Any:
    """Read a setting used only by the internal quantitative workflow."""

    return read_json(config_path).get(key, default)


def build_agent(
    args: Any,
    *,
    titan_encoder: Any | None = None,
    plip_encoder: Any | None = None,
) -> PlanPathQuantitativeBackend:
    """Build the internal backend from the unified PlanPath arguments."""

    config = AgentConfig(
        method=str(defaults_get(args.config, "method", "PlanPath")),
        titan_model_path=args.titan_model,
        plip_model_path=args.plip_model,
        tissue_map_root=args.tissue_map_root,
        histoplus_model_root=args.histoplus_model_root,
        cell_seg_root=args.cell_seg_root,
        cache_root=args.cache_root,
        clinical_domain=args.clinical_domain,
        device=args.device,
        top_k_slides=args.top_k_slides,
        low_power_top_k=args.low_power_top_k,
        high_power_top_k=args.high_power_top_k,
        max_zoom_rounds_per_slide=args.max_zoom_rounds_per_slide,
        min_direct_evidence=args.min_direct_evidence,
        roi_min_center_distance=args.roi_min_center_distance,
        min_zoom_parent_rois=args.min_zoom_parent_rois,
        patho_concurrency=args.patho_concurrency,
        tissue_navigation_enabled=not args.no_tissue_navigation,
        histoplus_enabled=not args.no_histoplus,
        precomputed_cell_seg_enabled=not args.no_precomputed_cell_seg,
        histoplus_required_top_k=int(
            defaults_get(args.config, "histoplus_required_top_k", 5)
        ),
        histoplus_conditional_top_k=int(
            defaults_get(args.config, "histoplus_conditional_top_k", 2)
        ),
        histoplus_batch_size=int(
            defaults_get(args.config, "histoplus_batch_size", 5)
        ),
        histoplus_inference_image_size=int(
            defaults_get(args.config, "histoplus_inference_image_size", 784)
        ),
        histoplus_min_type_confidence=float(
            defaults_get(args.config, "histoplus_min_type_confidence", 0.5)
        ),
        histoplus_mixed_precision=bool(
            defaults_get(args.config, "histoplus_mixed_precision", True)
        ),
        histoplus_save_overlay=bool(
            defaults_get(args.config, "histoplus_save_overlay", False)
        ),
        histoplus_fail_open=not args.histoplus_strict,
        quantification_enabled=bool(
            defaults_get(args.config, "quantification_enabled", True)
        ),
        quantification_low_power_top_k=int(
            defaults_get(args.config, "quantification_low_power_top_k", 12)
        ),
        quantification_high_power_top_k=int(
            defaults_get(args.config, "quantification_high_power_top_k", 12)
        ),
        quantification_slide_enrichment_top_k=int(
            defaults_get(args.config, "quantification_slide_enrichment_top_k", 3)
        ),
        quantification_min_tissue_coverage=float(
            defaults_get(args.config, "quantification_min_tissue_coverage", 0.80)
        ),
        quantification_min_stratum_coverage=float(
            defaults_get(args.config, "quantification_min_stratum_coverage", 0.60)
        ),
        quantification_min_cellular_rois=int(
            defaults_get(args.config, "quantification_min_cellular_rois", 4)
        ),
        quantification_min_tumor_cells=int(
            defaults_get(args.config, "quantification_min_tumor_cells", 40)
        ),
        quantification_min_focus_local_fraction=float(
            defaults_get(
                args.config,
                "quantification_min_focus_local_fraction",
                0.10,
            )
        ),
        quantification_rounding_percent=int(
            defaults_get(args.config, "quantification_rounding_percent", 0)
        ),
        quantification_estimator_mode=str(
            defaults_get(
                args.config,
                "quantification_estimator_mode",
                "calibrated_continuous",
            )
        ),
        quantification_output_decimals=int(
            defaults_get(args.config, "quantification_output_decimals", 2)
        ),
        request_timeout=float(defaults_get(args.config, "request_timeout", 120.0)),
        qwen_endpoint=args.qwen_endpoint,
        qwen_model=args.qwen_model,
        patho_endpoint=args.patho_endpoint,
        patho_model=args.patho_model,
        generation_seed=args.generation_seed,
        accelerated_evaluation_enabled=bool(
            defaults_get(args.config, "accelerated_evaluation_enabled", False)
        ),
        patient_visual_cache_enabled=bool(
            defaults_get(args.config, "patient_visual_cache_enabled", False)
        ),
        accelerated_skip_low_power_review_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_skip_low_power_review_enabled",
                False,
            )
        ),
        accelerated_skip_non_actionable_extent_review_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_skip_non_actionable_extent_review_enabled",
                False,
            )
        ),
        accelerated_reuse_complete_visual_scan_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_reuse_complete_visual_scan_enabled",
                False,
            )
        ),
        accelerated_protect_complex_and_interface_slide_pool_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_protect_complex_and_interface_slide_pool_enabled",
                False,
            )
        ),
        accelerated_protected_roi_floor=int(
            defaults_get(args.config, "accelerated_protected_roi_floor", 5)
        ),
        accelerated_regional_node_20x_floor_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_regional_node_20x_floor_enabled",
                False,
            )
        ),
        accelerated_regional_node_20x_top_k=int(
            defaults_get(args.config, "accelerated_regional_node_20x_top_k", 16)
        ),
        accelerated_special_low_power_top_k=int(
            defaults_get(args.config, "accelerated_special_low_power_top_k", 12)
        ),
        accelerated_special_20x_top_k=int(
            defaults_get(args.config, "accelerated_special_20x_top_k", 16)
        ),
        accelerated_special_40x_top_k=int(
            defaults_get(args.config, "accelerated_special_40x_top_k", 8)
        ),
        accelerated_special_20x_parent_top_k=int(
            defaults_get(args.config, "accelerated_special_20x_parent_top_k", 12)
        ),
        accelerated_special_40x_parent_top_k=int(
            defaults_get(args.config, "accelerated_special_40x_parent_top_k", 8)
        ),
        accelerated_slide_promotion_top_k=int(
            defaults_get(args.config, "accelerated_slide_promotion_top_k", 8)
        ),
    )
    qwen_client = OpenAICompatClient(
        config.qwen_endpoint,
        config.qwen_model,
        config.request_timeout,
        config.generation_seed,
    )
    patho_client = OpenAICompatClient(
        config.patho_endpoint,
        config.patho_model,
        config.request_timeout,
        config.generation_seed,
    )
    qwen_verifier_client = OpenAICompatClient(
        config.qwen_endpoint,
        config.qwen_model,
        config.request_timeout,
        config.generation_seed + 1,
    )
    if not args.skip_health_check:
        print(f"Qwen service: {qwen_client.health()}", flush=True)
        print(f"Patho-R1 service: {patho_client.health()}", flush=True)

    controller = QwenController(StructuredGenerator(qwen_client), config)
    titan = TitanSlideRetriever(
        titan_encoder
        if titan_encoder is not None
        else LazyTitanTextEncoder(config.titan_model_path, config.device)
    )
    selector = ROISelector(
        plip_encoder
        if plip_encoder is not None
        else LocalPLIPEncoder(config.plip_model_path, config.device),
        low_power_top_k=config.low_power_top_k,
        high_power_top_k=config.high_power_top_k,
        min_center_distance=config.roi_min_center_distance,
        tissue_navigation_enabled=config.tissue_navigation_enabled,
        accelerated_evaluation_enabled=config.accelerated_evaluation_enabled,
        accelerated_protected_roi_floor=config.accelerated_protected_roi_floor,
        accelerated_special_low_power_top_k=(
            config.accelerated_special_low_power_top_k
        ),
        accelerated_regional_node_20x_top_k=(
            config.accelerated_regional_node_20x_top_k
        ),
        accelerated_special_20x_top_k=config.accelerated_special_20x_top_k,
        accelerated_special_40x_top_k=config.accelerated_special_40x_top_k,
        quantification_low_power_top_k=config.quantification_low_power_top_k,
        quantification_high_power_top_k=config.quantification_high_power_top_k,
    )
    visual = VisualAnalyzer(
        selector,
        PathoObserver(
            StructuredGenerator(patho_client),
            config,
            verifier_generator=StructuredGenerator(qwen_verifier_client),
        ),
        HistoPLUSAnalyzer(config) if config.histoplus_enabled else None,
    )
    retrieval_cache = (
        None
        if args.no_retrieval_cache
        else PrecomputedRetrievalCache(
            config.cache_root,
            config.qwen_model,
            config.titan_model_path,
            config.generation_seed,
        )
    )
    return PlanPathQuantitativeBackend(
        controller,
        titan,
        visual,
        config,
        retrieval_cache=retrieval_cache,
    )


__all__ = ["build_agent"]
