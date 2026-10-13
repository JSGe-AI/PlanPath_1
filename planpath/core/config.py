"""Runtime configuration for the PlanPath workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentConfig:
    method: str = ""
    titan_model_path: Path = PROJECT_ROOT / "models" / "titan"
    plip_model_path: Path = PROJECT_ROOT / "models" / "plip"
    tissue_map_root: Path = PROJECT_ROOT / "artifacts" / "tissue_maps"
    histoplus_model_root: Path = PROJECT_ROOT / "models" / "histoplus"
    cell_seg_root: Path = PROJECT_ROOT / "artifacts" / "cell_seg"
    cache_root: Path = PROJECT_ROOT / "artifacts" / "cache"
    clinical_domain: str | None = None
    device: str = "cuda"
    top_k_slides: int = 5
    low_power_top_k: int = 5
    high_power_top_k: int = 5
    max_zoom_rounds_per_slide: int = 2
    min_direct_evidence: int = 2
    roi_min_center_distance: int = 12288
    min_zoom_parent_rois: int = 2
    patho_concurrency: int = 5
    tissue_navigation_enabled: bool = True
    histoplus_enabled: bool = True
    precomputed_cell_seg_enabled: bool = True
    histoplus_required_top_k: int = 5
    histoplus_conditional_top_k: int = 2
    histoplus_batch_size: int = 5
    histoplus_inference_image_size: int = 784
    histoplus_min_type_confidence: float = 0.50
    histoplus_mixed_precision: bool = True
    histoplus_save_overlay: bool = False
    histoplus_fail_open: bool = True
    request_timeout: float = 120.0
    qwen_endpoint: str = "http://127.0.0.1:7050/v1"
    qwen_model: str = "PlanPath-Qwen3.5-4B"
    patho_endpoint: str = "http://127.0.0.1:7051/v1"
    patho_model: str = "PlanPath-Patho-R1-7B"
    generation_seed: int = 0
    plan_max_tokens: int = 1800
    review_max_tokens: int = 1800
    patho_max_tokens: int = 900
    final_max_tokens: int = 1200
    accelerated_evaluation_enabled: bool = False
    patient_visual_cache_enabled: bool = False
    accelerated_skip_low_power_review_enabled: bool = False
    accelerated_skip_non_actionable_extent_review_enabled: bool = False
    accelerated_reuse_complete_visual_scan_enabled: bool = False
    accelerated_protect_complex_and_interface_slide_pool_enabled: bool = False
    accelerated_protected_roi_floor: int = 5
    accelerated_regional_node_20x_floor_enabled: bool = False
    accelerated_regional_node_20x_top_k: int = 16
    accelerated_special_low_power_top_k: int = 12
    accelerated_special_20x_top_k: int = 16
    accelerated_special_40x_top_k: int = 8
    accelerated_special_20x_parent_top_k: int = 12
    accelerated_special_40x_parent_top_k: int = 8
    accelerated_slide_promotion_top_k: int = 8

    def __post_init__(self) -> None:
        positive = {
            "top_k_slides": self.top_k_slides,
            "low_power_top_k": self.low_power_top_k,
            "high_power_top_k": self.high_power_top_k,
            "min_direct_evidence": self.min_direct_evidence,
            "patho_concurrency": self.patho_concurrency,
            "min_zoom_parent_rois": self.min_zoom_parent_rois,
            "histoplus_required_top_k": self.histoplus_required_top_k,
            "histoplus_conditional_top_k": self.histoplus_conditional_top_k,
            "histoplus_batch_size": self.histoplus_batch_size,
            "histoplus_inference_image_size": self.histoplus_inference_image_size,
            "plan_max_tokens": self.plan_max_tokens,
            "review_max_tokens": self.review_max_tokens,
            "patho_max_tokens": self.patho_max_tokens,
            "final_max_tokens": self.final_max_tokens,
            "accelerated_special_low_power_top_k": self.accelerated_special_low_power_top_k,
            "accelerated_protected_roi_floor": self.accelerated_protected_roi_floor,
            "accelerated_regional_node_20x_top_k": self.accelerated_regional_node_20x_top_k,
            "accelerated_special_20x_top_k": self.accelerated_special_20x_top_k,
            "accelerated_special_40x_top_k": self.accelerated_special_40x_top_k,
            "accelerated_special_20x_parent_top_k": self.accelerated_special_20x_parent_top_k,
            "accelerated_special_40x_parent_top_k": self.accelerated_special_40x_parent_top_k,
            "accelerated_slide_promotion_top_k": self.accelerated_slide_promotion_top_k,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.max_zoom_rounds_per_slide < 0:
            raise ValueError("max_zoom_rounds_per_slide cannot be negative")
        if self.roi_min_center_distance < 0:
            raise ValueError("roi_min_center_distance cannot be negative")
        if not 0 <= self.histoplus_min_type_confidence <= 1:
            raise ValueError("histoplus_min_type_confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value
