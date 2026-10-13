"""PLIP selection of five low-power or high-power ROIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from PIL import Image

from .image_io import field_size_level0, read_field
from .schemas import EvidencePlan, ROI, TissueNavigation
from .tissue_map import TissueMap
from .tissue_routing import routing_policy


def _is_renal_grade_plan(plan: EvidencePlan | None) -> bool:
    """Keep broad per-slide extremum sampling specific to WHO/ISUP grading."""

    if plan is None:
        return False
    text = " ".join(
        [
            plan.target_finding,
            *plan.question_attributes,
            plan.evidence_goal,
        ]
    ).casefold()
    return "who/isup" in text and "grade" in text


class PLIPEncoder(Protocol):
    def text_feature(self, text: str) -> np.ndarray: ...
    def image_features(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray: ...


class LocalPLIPEncoder:
    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PLIP retrieval requires torch and transformers") from error
        self._torch = torch
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(str(model_path), use_fast=True)
        self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device).eval()

    def _tensor(self, output: Any, kind: str) -> Any:
        if self._torch.is_tensor(output):
            return output
        pooled = getattr(output, "pooler_output", None)
        if self._torch.is_tensor(pooled):
            return pooled
        raise TypeError(f"unsupported PLIP {kind} output: {type(output)!r}")

    def text_feature(self, text: str) -> np.ndarray:
        inputs = self.processor.tokenizer(
            [text],
            return_tensors="pt",
            max_length=77,
            padding="max_length",
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            feature = self._tensor(self.model.get_text_features(**inputs), "text")
        feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feature[0].float().cpu().numpy()

    def image_features(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        if not images:
            raise ValueError("cannot encode an empty image list")
        output: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor.image_processor(
                images=images[start : start + batch_size], return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                features = self._tensor(self.model.get_image_features(**inputs), "image")
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            output.append(features.float().cpu().numpy())
        return np.concatenate(output, axis=0)


@dataclass(frozen=True)
class SelectedROI:
    roi: ROI
    image: Image.Image


def high_power_information(image: Image.Image) -> tuple[bool, float]:
    """Estimate whether an H&E field contains enough cellular detail to inspect.

    PLIP can assign high similarity to nearly empty pale fields when the text
    query contains words such as ``clear`` or ``mucin``.  This gate is wholly
    question-independent: it measures hematoxylin-like pixels and local image
    variation, not a disease feature.  The score is retained only for a
    deterministic fallback when a parent ROI has no field passing the gate.
    """

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    red, green, blue = (array[:, :, index] for index in range(3))
    hematoxylin_like = (
        (blue > green * 1.05)
        & (red > green * 1.03)
        & (gray < 0.80)
    )
    hematoxylin_fraction = float(hematoxylin_like.mean())
    edge_energy = float(
        np.abs(np.diff(gray, axis=1)).mean()
        + np.abs(np.diff(gray, axis=0)).mean()
    )
    score = hematoxylin_fraction + 4.0 * max(0.0, edge_energy - 0.015)
    informative = hematoxylin_fraction >= 0.04 and edge_energy >= 0.020
    return informative, score


def low_power_mature_adipose_score(image: Image.Image) -> float:
    """Estimate mature adipose architecture without diagnosing the pixels.

    Clear-cell renal tumour is frequently assigned the adipose prototype by
    the navigation model.  Mature adipocytes, however, form relatively large,
    enclosed, low-saturation vacuoles, whereas clear tumour cytoplasm produces
    many smaller spaces with a much denser hematoxylin signal.  This generic
    image statistic is used only to place a later reviewed ROI.  Components
    touching the image edge are ignored so empty glass cannot dominate.
    """

    array = np.asarray(
        image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    gray = array.mean(axis=2)
    chroma = array.max(axis=2) - array.min(axis=2)
    pale = (gray >= 0.78) & (chroma <= 0.18)
    visited = np.zeros_like(pale, dtype=bool)
    retained_area = 0
    height, width = pale.shape
    for start_y, start_x in zip(*np.nonzero(pale & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        touches_edge = False
        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            touches_edge = touches_edge or x in {0, width - 1} or y in {
                0,
                height - 1,
            }
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and pale[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        aspect = min(component_width, component_height) / max(
            component_width, component_height
        )
        fill = area / max(component_width * component_height, 1)
        if (
            not touches_edge
            and 30 <= area <= 1250
            and aspect >= 0.30
            and fill >= 0.30
        ):
            retained_area += area
    vacuole_fraction = retained_area / float(width * height)
    hematoxylin_fraction = float(((gray < 0.55) & (chroma > 0.15)).mean())
    # Suppress densely nucleated clear-cell tumour while retaining adipose
    # lobules with sparse septal nuclei.  The squared taper intentionally
    # favors a nearby pure-fat anchor; the ROI is subsequently centered
    # between that anchor and an independently ranked tumour-bearing window.
    nuclear_sparsity = float(
        np.clip((0.10 - hematoxylin_fraction) / 0.08, 0.0, 1.0) ** 2
    )
    return vacuole_fraction * nuclear_sparsity


class ROISelector:
    def __init__(
        self,
        encoder: PLIPEncoder,
        *,
        low_power_top_k: int = 5,
        high_power_top_k: int = 5,
        min_center_distance: int = 4096,
        tissue_navigation_enabled: bool = True,
        accelerated_evaluation_enabled: bool = False,
        accelerated_protected_roi_floor: int = 5,
        accelerated_special_low_power_top_k: int = 12,
        accelerated_regional_node_20x_top_k: int = 16,
        accelerated_special_20x_top_k: int = 16,
        accelerated_special_40x_top_k: int = 8,
    ) -> None:
        self.encoder = encoder
        self.low_power_top_k = low_power_top_k
        self.high_power_top_k = high_power_top_k
        self.min_center_distance = min_center_distance
        self.tissue_navigation_enabled = tissue_navigation_enabled
        self.accelerated_evaluation_enabled = accelerated_evaluation_enabled
        self.accelerated_protected_roi_floor = accelerated_protected_roi_floor
        self.accelerated_special_low_power_top_k = accelerated_special_low_power_top_k
        self.accelerated_regional_node_20x_top_k = (
            accelerated_regional_node_20x_top_k
        )
        self.accelerated_special_20x_top_k = accelerated_special_20x_top_k
        self.accelerated_special_40x_top_k = accelerated_special_40x_top_k
        self._low_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._text_features: dict[str, np.ndarray] = {}
        self._tissue_maps: dict[str, TissueMap] = {}
        self._adipose_navigation_cache: dict[str, np.ndarray] = {}
        self._high_feature_cache_patient_id: str | None = None
        self._high_feature_cache: dict[
            tuple[str, str, tuple[int, int, int, int]], np.ndarray
        ] = {}
        self.metrics = {
            "low_feature_loads": 0,
            "low_feature_cache_hits": 0,
            "plip_text_encoder_calls": 0,
            "plip_text_cache_hits": 0,
            "plip_high_image_encoder_calls": 0,
            "plip_high_feature_cache_hits": 0,
            "plip_high_feature_cache_writes": 0,
            "high_candidates_filtered": 0,
            "high_quality_fallbacks": 0,
            "high_parent_diversity_bypassed": 0,
            "tissue_map_loads": 0,
            "tissue_map_cache_hits": 0,
            "tissue_map_missing": 0,
            "tissue_routed_low_selections": 0,
            "tissue_routed_high_selections": 0,
            "outer_interface_recenters": 0,
            "tissue_interface_recenters": 0,
            "adipose_morphology_scans": 0,
            "adipose_morphology_cache_hits": 0,
            "adipose_interface_recenters": 0,
        }

    def _tissue_map(self, slide: dict[str, Any]) -> TissueMap | None:
        if not self.tissue_navigation_enabled:
            return None
        raw = str(slide.get("tissue_map_cache") or "")
        if not raw or not Path(raw).is_file():
            self.metrics["tissue_map_missing"] += 1
            return None
        key = str(Path(raw).resolve())
        if key in self._tissue_maps:
            self.metrics["tissue_map_cache_hits"] += 1
            return self._tissue_maps[key]
        value = TissueMap(Path(raw))
        self._tissue_maps[key] = value
        self.metrics["tissue_map_loads"] += 1
        return value

    @staticmethod
    def _percentiles(values: np.ndarray) -> np.ndarray:
        if not len(values):
            return np.asarray([], dtype=np.float32)
        order = np.argsort(values, kind="stable")
        output = np.empty(len(values), dtype=np.float32)
        output[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        return output

    def _mature_adipose_scores(
        self,
        slide: dict[str, Any],
        coords: np.ndarray,
    ) -> np.ndarray:
        key = str(Path(str(slide["raw_slide_path"])).resolve())
        cached = self._adipose_navigation_cache.get(key)
        if cached is not None and len(cached) == len(coords):
            self.metrics["adipose_morphology_cache_hits"] += 1
            return cached
        with self._open_slide(str(slide["raw_slide_path"])) as wsi:
            values = np.asarray(
                [
                    low_power_mature_adipose_score(
                        read_field(wsi, int(x), int(y), 5, output_pixels=128)
                    )
                    for x, y in coords
                ],
                dtype=np.float32,
            )
        self._adipose_navigation_cache[key] = values
        self.metrics["adipose_morphology_scans"] += 1
        return values

    @classmethod
    def _adipose_interface_navigation(
        cls,
        coords: np.ndarray,
        adipose_scores: np.ndarray,
        tumor_scores: np.ndarray,
        plip_scores: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
        """Pair a morphologic fat anchor with a nearby tumour candidate."""

        output = np.zeros(len(coords), dtype=np.float32)
        centers: dict[int, tuple[int, int]] = {}
        if not len(coords):
            return output, centers
        tumor_rank = (
            0.70 * cls._percentiles(tumor_scores)
            + 0.30 * cls._percentiles(plip_scores)
        )
        coordinate_centers = coords.astype(np.float64) + 2048.0
        # A pure-fat anchor several coarse tiles away from tumour does not
        # localize a tumour-fat interface: the segment may contain a thick
        # fibrous capsule, renal parenchyma, or blank glass. Keep only
        # immediately neighboring coarse windows. A direct interface on the
        # 4096-pixel grid is normally one axial/diagonal step away; three
        # axial steps is a conservative upper bound for staggered tissue.
        maximum_distance = 12288.0
        for index in range(len(coords)):
            if float(adipose_scores[index]) <= 0:
                continue
            distances = np.linalg.norm(
                coordinate_centers - coordinate_centers[index], axis=1
            )
            proximity = np.clip(1.0 - distances / maximum_distance, 0.0, 1.0)
            pair_scores = tumor_rank * proximity
            # Relative ranks alone are unsafe in a broad pure-fat region: a
            # map value of 0.001 can still be the best local "tumour" window.
            # Require a modest absolute tumour signal before forming a pair.
            pair_scores[np.asarray(tumor_scores) < 0.02] = 0.0
            pair_scores[index] = 0.0
            neighbor = int(np.argmax(pair_scores))
            if float(pair_scores[neighbor]) <= 0:
                continue
            output[index] = float(adipose_scores[index] * pair_scores[neighbor])
            tumor_center = coordinate_centers[neighbor]
            toward_fat = coordinate_centers[index] - tumor_center
            distance = max(float(np.linalg.norm(toward_fat)), 1.0)
            # Center the 4096-pixel field halfway across an adjacent pair so
            # both compartments, rather than just the tumour-facing edge,
            # can enter the same parent ROI. Cap the shift at one full
            # coarse-grid step: for the most distant allowed fallback pairs,
            # this still avoids jumping deep into a broad pure-fat field.
            interface_center = tumor_center + toward_fat / distance * min(
                4096.0, distance / 2.0
            )
            centers[index] = (
                int(round(interface_center[0])),
                int(round(interface_center[1])),
            )
        return output, centers

    @staticmethod
    def _interface_balanced_indices(
        eligible: list[int],
        parent_ids: list[str],
        routing_scores: np.ndarray,
        target_count: int,
    ) -> list[int]:
        """Trace a focal boundary without collapsing onto one broad parent.

        The two strongest global fields may be adjacent descendants of the
        same 5x ROI, which is useful for following a continuous interface.
        The remaining budget is first spent on the best field from each
        unrepresented parent.  This prevents a highly scoring intratumoral
        fibrous focus from consuming all five fields while the true outer
        capsule is present in another low-power window.
        """

        target = min(target_count, len(eligible))
        if target <= 0:
            return []
        ordered = list(eligible[: min(2, target)])
        represented = {parent_ids[index] for index in ordered}
        best_new_parent: dict[str, int] = {}
        for index in eligible:
            parent_id = parent_ids[index]
            if parent_id in represented:
                continue
            best_new_parent.setdefault(parent_id, index)
        diverse = sorted(
            best_new_parent.values(),
            key=lambda index: float(routing_scores[index]),
            reverse=True,
        )
        for index in diverse:
            ordered.append(index)
            represented.add(parent_ids[index])
            if len(ordered) == target:
                return ordered
        for index in eligible:
            if index not in ordered:
                ordered.append(index)
            if len(ordered) == target:
                break
        return ordered

    @staticmethod
    def _best_indices_for_parents(
        eligible: list[int],
        parent_ids: list[str],
        scores: np.ndarray,
        required_parent_ids: set[str],
        limit: int,
        *,
        per_parent_limit: int = 1,
    ) -> list[int]:
        """Reserve the strongest child crops for routed outer parents."""

        candidates: dict[str, list[int]] = {}
        for index in eligible:
            parent_id = parent_ids[index]
            if parent_id not in required_parent_ids:
                continue
            candidates.setdefault(parent_id, []).append(index)
        best: list[int] = []
        for indices in candidates.values():
            indices.sort(key=lambda index: float(scores[index]), reverse=True)
            best.extend(indices[: max(1, per_parent_limit)])
        return sorted(
            best,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:limit]

    @classmethod
    def _capsule_parent_quota_indices(
        cls,
        eligible: list[int],
        parent_ids: list[str],
        scores: np.ndarray,
        target_count: int,
        coordinates: np.ndarray | None = None,
    ) -> list[int]:
        """Preserve the complete capsule lineage within the fixed budget.

        At 5x -> 20x the capsule workflow supplies twelve parents and has a
        twenty-four-field budget, so retain two children from every parent.
        At 20x -> 40x it supplies up to twenty-four parents with the same
        budget, so retain one child from each.  Missing or quality-filtered
        parents simply release their quota back to the global score order.
        """

        target = min(target_count, len(eligible))
        if target <= 0:
            return []
        required_parent_ids = {parent_ids[index] for index in eligible}
        if not required_parent_ids:
            return []
        per_parent_limit = (
            2 if target >= 2 * len(required_parent_ids) else 1
        )
        if per_parent_limit == 1 or coordinates is None:
            ordered = cls._best_indices_for_parents(
                eligible,
                parent_ids,
                scores,
                required_parent_ids,
                target,
                per_parent_limit=per_parent_limit,
            )
        else:
            # Two top-ranked half-stride crops can overlap almost completely
            # and inspect only one side of a broad capsule ROI.  Keep the best
            # semantic match, then require the second crop to cover a distant
            # portion of that same parent.  A relative threshold works at
            # both hierarchy transitions (4096 -> 1024 and 1024 -> 512)
            # without encoding slide- or magnification-specific coordinates.
            per_parent: dict[str, list[int]] = {}
            for index in eligible:
                per_parent.setdefault(parent_ids[index], []).append(index)
            chosen: list[int] = []
            for indices in per_parent.values():
                ranked = sorted(
                    indices,
                    key=lambda index: float(scores[index]),
                    reverse=True,
                )
                first = ranked[0]
                chosen.append(first)
                if len(ranked) < 2:
                    continue
                first_center = np.asarray(
                    [
                        (coordinates[first][0] + coordinates[first][2]) / 2.0,
                        (coordinates[first][1] + coordinates[first][3]) / 2.0,
                    ],
                    dtype=np.float64,
                )
                distances = {
                    index: float(
                        np.linalg.norm(
                            np.asarray(
                                [
                                    (coordinates[index][0] + coordinates[index][2]) / 2.0,
                                    (coordinates[index][1] + coordinates[index][3]) / 2.0,
                                ],
                                dtype=np.float64,
                            )
                            - first_center
                        )
                    )
                    for index in ranked[1:]
                }
                minimum_distance = 0.75 * max(distances.values(), default=0.0)
                distant = [
                    index
                    for index in ranked[1:]
                    if distances[index] >= minimum_distance
                ]
                chosen.append(distant[0] if distant else ranked[1])
            ordered = sorted(
                chosen,
                key=lambda index: float(scores[index]),
                reverse=True,
            )[:target]
        for index in eligible:
            if index not in ordered:
                ordered.append(index)
            if len(ordered) == target:
                break
        return ordered

    @staticmethod
    def _parent_diverse_indices(
        eligible: list[int],
        parent_ids: list[str],
        scores: np.ndarray,
        target_count: int,
    ) -> list[int]:
        """Retain one child from every available parent before extra crops.

        Interface and extremum searches may request eight output fields even
        when the configured generic high-power top-k is five.  The previous
        implementation nevertheless truncated the parent-reservation pass at
        that generic top-k.  Three already-localized parent regions could
        therefore disappear at the next magnification, including a 20x nerve
        profile in a confirmed perineural-invasion case.
        """

        target = min(target_count, len(eligible))
        best_per_parent: dict[str, int] = {}
        for index in eligible:
            best_per_parent.setdefault(parent_ids[index], index)
        ordered = sorted(
            best_per_parent.values(),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:target]
        for index in eligible:
            if index not in ordered:
                ordered.append(index)
            if len(ordered) == target:
                break
        return ordered

    def _hybrid_indices(
        self,
        coords: np.ndarray,
        plip_scores: np.ndarray,
        fusion_scores: np.ndarray,
        tissue_scores: np.ndarray,
        interface_scores: np.ndarray,
        heterogeneity_scores: np.ndarray,
        count: int,
        *,
        interface_priority: bool = False,
        outer_interface_scores: np.ndarray | None = None,
        adipose_interface_priority: bool = False,
        adipose_boundary_scores: np.ndarray | None = None,
    ) -> tuple[list[int], dict[int, str]]:
        target = min(count, len(coords))
        selected: list[int] = []
        routes: dict[int, str] = {}

        def add(
            values: np.ndarray,
            requested: int,
            route: str,
            *,
            keep_best_available: bool = False,
        ) -> None:
            order = [int(item) for item in np.argsort(values, kind="stable")[::-1]]
            added = 0
            # A distance rule is useful for broad coverage, but it must not
            # suppress the strongest candidate for a focal relation merely
            # because an earlier global-fusion window lies nearby.  Perirenal
            # fat invasion is especially vulnerable because clear-cell tumour
            # can resemble adipose in the navigation map.  Pin the strongest
            # still-available interface candidate first, then diversify the
            # remaining members of that route.
            if keep_best_available:
                for index in order:
                    if index not in selected:
                        selected.append(index)
                        routes[index] = route
                        added += 1
                        break
                if added == requested or len(selected) == target:
                    return
            for enforce_distance in (True, False):
                for index in order:
                    if index in selected:
                        continue
                    if enforce_distance and selected:
                        x, y = (int(value) for value in coords[index])
                        if any(
                            math.hypot(x - int(coords[other][0]), y - int(coords[other][1]))
                            < self.min_center_distance
                            for other in selected
                        ):
                            continue
                    selected.append(index)
                    routes[index] = route
                    added += 1
                    if added == requested or len(selected) == target:
                        return

        add(
            fusion_scores,
            min(1 if interface_priority else 2, target),
            "plip_tissue_fusion",
        )
        adipose_boundary_available = bool(
            adipose_interface_priority
            and adipose_boundary_scores is not None
            and float(adipose_boundary_scores.max(initial=0.0)) > 0
        )
        if adipose_boundary_available and len(selected) < target:
            add(
                adipose_boundary_scores,
                min(3, target - len(selected)),
                "morphologic_adipose_interface",
                keep_best_available=True,
            )
        if (
            interface_priority
            and outer_interface_scores is not None
            and len(selected) < target
            and float(outer_interface_scores.max(initial=0.0)) > 0
        ):
            # Keep the outer-surface route broad, but do not let it consume
            # every remaining slot in the twelve-field capsule portfolio.
            # Six outer windows plus the other fixed routes exactly filled
            # the budget and eliminated ``fusion_fill``.  A focal true
            # tumour/capsule transition with a strong combined PLIP/tissue
            # score but only a moderate exterior score could therefore be
            # missed even during an otherwise exhaustive slide search.
            # Five outer windows leave one label-free global fusion reserve
            # without increasing the observation budget.
            add(
                outer_interface_scores,
                min(
                    (
                        1
                        if adipose_boundary_available
                        else (3 if adipose_interface_priority else 5)
                    ),
                    target - len(selected),
                ),
                "outer_tissue_interface",
            )
        if len(selected) < target:
            if float(interface_scores.max(initial=0.0)) > 0:
                add(
                    interface_scores,
                    min(
                        4
                        if adipose_interface_priority
                        and not adipose_boundary_available
                        else 3
                        if adipose_boundary_available
                        else (2 if interface_priority else 1),
                        target - len(selected),
                    ),
                    "tissue_interface",
                    keep_best_available=adipose_interface_priority,
                )
            else:
                add(tissue_scores, 1, "target_tissue")
        if interface_priority and len(selected) < target:
            add(tissue_scores, 1, "target_tissue")
        if len(selected) < target:
            add(plip_scores, 1, "plip_rescue")
        if len(selected) < target:
            add(heterogeneity_scores + tissue_scores * 0.25, 1, "heterogeneity")
        if len(selected) < target:
            add(fusion_scores, target - len(selected), "fusion_fill")
        return selected, routes

    def _text(self, query: str) -> np.ndarray:
        key = " ".join(query.split()).casefold()
        if key in self._text_features:
            self.metrics["plip_text_cache_hits"] += 1
            return self._text_features[key]
        value = np.asarray(self.encoder.text_feature(query), dtype=np.float32).reshape(-1)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        self._text_features[key] = value
        self.metrics["plip_text_encoder_calls"] += 1
        return value

    def _load_low(self, slide: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        path = Path(str(slide["plip_feature_cache"]))
        key = str(path.resolve())
        if key in self._low_features:
            self.metrics["low_feature_cache_hits"] += 1
            return self._low_features[key]
        with np.load(path) as payload:
            coords = np.asarray(payload["coords"], dtype=np.int64)
            features = np.asarray(payload["features"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(features):
            raise ValueError(f"invalid PLIP cache for {slide['slide_id']}: {path}")
        keep = np.isfinite(features).all(axis=1) & (np.linalg.norm(features, axis=1) > 1e-12)
        coords, features = coords[keep], features[keep]
        if not len(coords):
            raise ValueError(f"PLIP cache has no valid features: {path}")
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        self._low_features[key] = (coords, features)
        self.metrics["low_feature_loads"] += 1
        return coords, features

    def _diverse_indices(self, coords: np.ndarray, scores: np.ndarray, count: int) -> list[int]:
        ordered = [int(index) for index in np.argsort(scores, kind="stable")[::-1]]
        target = min(count, len(coords))
        # Preserve the three strongest semantic matches, then spend the
        # remaining slots on spatial exploration.  A pure diversity rule can
        # discard a true lesion cluster; a pure score rule can sample five
        # adjacent fields from the same benign region.
        relevance_count = min(target, max(1, (target + 1) // 2))
        selected: list[int] = ordered[:relevance_count]
        for enforce_distance in (True, False):
            for index in ordered:
                if index in selected:
                    continue
                if enforce_distance:
                    x, y = (int(value) for value in coords[index])
                    if any(
                        math.hypot(x - int(coords[other][0]), y - int(coords[other][1]))
                        < self.min_center_distance
                        for other in selected
                    ):
                        continue
                selected.append(index)
                if len(selected) == target:
                    return selected
        return selected

    @staticmethod
    def _open_slide(path: str) -> Any:
        try:
            import openslide
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("WSI reading requires openslide-python") from error
        return openslide.OpenSlide(path)

    def select_low(
        self,
        slide: dict[str, Any],
        query: str,
        plan: EvidencePlan | None = None,
    ) -> list[SelectedROI]:
        coords, features = self._load_low(slide)
        text = self._text(query)
        if features.shape[1] != len(text):
            raise ValueError(
                f"PLIP dimension mismatch for {slide['slide_id']}: "
                f"image={features.shape[1]}, text={len(text)}"
            )
        scores = features @ text
        tissue_map = self._tissue_map(slide) if plan is not None else None
        summaries: list[dict[str, Any]] = []
        fusion_scores = self._percentiles(scores)
        routes: dict[int, str] = {}
        adipose_scores = np.zeros(len(coords), dtype=np.float32)
        adipose_boundary_scores: np.ndarray | None = None
        adipose_interface_centers: dict[int, tuple[int, int]] = {}
        if tissue_map is not None and plan is not None:
            policy = routing_policy(str(slide.get("cancer") or ""), plan)
            summaries = [
                tissue_map.window_summary(
                    (int(x), int(y), int(x) + 4096, int(y) + 4096),
                    policy.target_labels,
                    policy.interface_pairs,
                )
                for x, y in coords
            ]
            tissue_scores = np.asarray([item["target_score"] for item in summaries], dtype=np.float32)
            interface_scores = np.asarray([item["interface_score"] for item in summaries], dtype=np.float32)
            heterogeneity = np.asarray([item["heterogeneity"] for item in summaries], dtype=np.float32)
            outer_pairs = tuple(
                pair
                for pair in policy.interface_pairs
                if "artifact_or_background" in pair
            )
            perirenal_fat_search = (
                policy.reason == "tumor-perirenal-fat interface search"
            )
            outer_interface_scores: np.ndarray | None = None
            if outer_pairs or perirenal_fat_search:
                # Fat-capsule questions combine two independent navigation
                # hypotheses: the prototype tumour/adipose boundary and the
                # peripheral tumour surface.  The latter is only exploratory;
                # pixels in a reviewed high-power ROI must still show direct
                # tumour entry between mature adipocytes before a positive
                # answer is allowed.
                outer_interface_scores = np.asarray(
                    [
                        tissue_map.exterior_interface_score(
                            (int(x), int(y), int(x) + 4096, int(y) + 4096),
                            policy.target_labels,
                        )
                        for x, y in coords
                    ],
                    dtype=np.float32,
                )
            if perirenal_fat_search:
                adipose_scores = self._mature_adipose_scores(slide, coords)
                (
                    adipose_boundary_scores,
                    adipose_interface_centers,
                ) = self._adipose_interface_navigation(
                    coords,
                    adipose_scores,
                    tissue_scores,
                    scores,
                )
            if outer_interface_scores is not None:
                fusion_scores = (
                    0.35 * self._percentiles(scores)
                    + 0.20 * self._percentiles(tissue_scores)
                    + 0.20 * self._percentiles(interface_scores)
                    + 0.20 * self._percentiles(outer_interface_scores)
                    + 0.05 * self._percentiles(heterogeneity)
                )
            elif policy.interface_pairs:
                fusion_scores = (
                    0.45 * self._percentiles(scores)
                    + 0.20 * self._percentiles(tissue_scores)
                    + 0.30 * self._percentiles(interface_scores)
                    + 0.05 * self._percentiles(heterogeneity)
                )
            else:
                fusion_scores = (
                    0.55 * self._percentiles(scores)
                    + 0.25 * self._percentiles(tissue_scores)
                    + 0.12 * self._percentiles(interface_scores)
                    + 0.08 * self._percentiles(heterogeneity)
                )
            # A focal boundary relation needs broader low-power coverage than
            # an ordinary tumour-typing task.  Keep the configured count for
            # non-interface questions, while reserving eight independent
            # windows for capsule, margin, depth, PNI, and LVI searches. A
            # renal capsule additionally reserves six tumour/background
            # edge windows, so use twelve fields for that rarer outer relation.
            microinvasion_search = policy.reason in {
                "tumor-nerve spatial-relation search",
                "tumor-vessel spatial-relation search",
                "microinvasion spatial-relation search",
            }
            nodal_deposit_search = policy.reason == "nodal epithelial-deposit search"
            capsule_search = policy.reason in {
                "tumor-capsule interface search",
                "tumor-perirenal-fat interface search",
            }
            if policy.interface_pairs:
                expanded_count = (
                    self.accelerated_special_low_power_top_k
                    if self.accelerated_evaluation_enabled
                    and (
                        outer_pairs
                        or capsule_search
                        or microinvasion_search
                        or nodal_deposit_search
                    )
                    else 12
                    if (
                        outer_pairs
                        or capsule_search
                        or microinvasion_search
                        or nodal_deposit_search
                    )
                    else 8
                )
                selection_count = max(
                    self.low_power_top_k,
                    expanded_count,
                )
            elif _is_renal_grade_plan(plan):
                # WHO/ISUP grade-defining components can be focal. Five broad
                # fields left clinically important parent regions unavailable
                # to the otherwise exhaustive true-10x/40x pass. Do not apply
                # this renal-specific expansion to every extremum question:
                # it destabilized gastric differentiation by doubling noisy
                # high-power captions despite already complete slide coverage.
                selection_count = max(self.low_power_top_k, 8)
            elif plan.decision_semantics in {"extremum", "quantitative", "composite"}:
                # A candidate may reduce the ordinary ROI budget to three,
                # but patient-level maxima/minima (including differentiation,
                # Lauren classification, pT, and grade) must retain the
                # validated five-field floor.  Renal WHO/ISUP grade keeps its
                # still broader eight-field specialization above.
                selection_count = max(
                    self.low_power_top_k,
                    self.accelerated_protected_roi_floor,
                )
            else:
                selection_count = self.low_power_top_k
            indices, routes = self._hybrid_indices(
                coords,
                scores,
                fusion_scores,
                tissue_scores,
                interface_scores,
                heterogeneity,
                selection_count,
                interface_priority=bool(policy.interface_pairs),
                outer_interface_scores=outer_interface_scores,
                adipose_interface_priority=perirenal_fat_search,
                adipose_boundary_scores=adipose_boundary_scores,
            )
            self.metrics["tissue_routed_low_selections"] += len(indices)
        else:
            indices = self._diverse_indices(coords, scores, self.low_power_top_k)
        output: list[SelectedROI] = []
        with self._open_slide(str(slide["raw_slide_path"])) as wsi:
            for rank, index in enumerate(indices, 1):
                x, y = (int(value) for value in coords[index])
                recenter_metric: str | None = None
                center: tuple[int, int] | None = None
                if (
                    tissue_map is not None
                    and outer_interface_scores is not None
                    and routes.get(index) == "outer_tissue_interface"
                ):
                    center = tissue_map.exterior_interface_center(
                        (x, y, x + 4096, y + 4096), policy.target_labels
                    )
                    recenter_metric = "outer_interface_recenters"
                elif (
                    routes.get(index) == "morphologic_adipose_interface"
                    and index in adipose_interface_centers
                ):
                    center = adipose_interface_centers[index]
                    recenter_metric = "adipose_interface_recenters"
                elif (
                    tissue_map is not None
                    and policy.interface_pairs
                    and routes.get(index) == "tissue_interface"
                ):
                    # ``window_summary`` can rank a coarse 4096-pixel window
                    # because its opposite corners contain the two requested
                    # tissues.  Reading that unchanged grid box may still put
                    # their actual boundary on the edge, after which every
                    # child crop sees only one tissue.  Center this dedicated
                    # interface route on the strongest local tissue pair. The
                    # map remains navigation-only; pixels must still establish
                    # the relation in the observer and controller.
                    center = tissue_map.interface_center(
                        (x, y, x + 4096, y + 4096), policy.interface_pairs
                    )
                    recenter_metric = "tissue_interface_recenters"
                if center is not None and recenter_metric is not None:
                    width, height = (int(value) for value in wsi.dimensions)
                    centered_x = min(
                        max(center[0] - 2048, 0), max(width - 4096, 0)
                    )
                    centered_y = min(
                        max(center[1] - 2048, 0), max(height - 4096, 0)
                    )
                    if (centered_x, centered_y) != (x, y):
                        x, y = centered_x, centered_y
                        self.metrics[recenter_metric] += 1
                navigation = None
                if summaries:
                    summary = tissue_map.window_summary(
                        (x, y, x + 4096, y + 4096),
                        policy.target_labels,
                        policy.interface_pairs,
                    )
                    navigation = TissueNavigation(
                        map_available=True,
                        target_score=summary["target_score"],
                        interface_score=summary["interface_score"],
                        heterogeneity=summary["heterogeneity"],
                        map_confidence=summary["map_confidence"],
                        unknown_fraction=summary.get("unknown_fraction", 0.0),
                        fusion_score=float(fusion_scores[index]),
                        adipose_morphology_score=float(adipose_scores[index]),
                        selection_route=routes.get(index, "fusion_fill"),
                        target_labels=list(policy.target_labels),
                        class_fractions=summary["class_fractions"],
                    )
                roi = ROI(
                    roi_id=f"{slide['slide_id']}_L{rank:02d}",
                    slide_id=str(slide["slide_id"]),
                    coordinates=(x, y, x + 4096, y + 4096),
                    magnification="5x",
                    retrieval_score=float(scores[index]),
                    matched_query=query,
                    tissue_navigation=navigation,
                )
                output.append(SelectedROI(roi, read_field(wsi, x, y, 5)))
        return output

    def select_high(
        self,
        slide: dict[str, Any],
        parents: list[ROI],
        magnification: Literal["10x", "20x", "40x"],
        query: str,
        *,
        ensure_parent_diversity: bool = True,
        plan: EvidencePlan | None = None,
    ) -> list[SelectedROI]:
        target_mag = int(magnification.rstrip("x"))
        target_size = field_size_level0(target_mag)
        raw: list[tuple[ROI, Image.Image]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()
        selection_policy = (
            routing_policy(str(slide.get("cancer") or ""), plan)
            if plan is not None
            else None
        )
        # A non-overlapping grid can place a thin capsule exactly on the edge
        # of two child patches, leaving neither child with both tumour and
        # boundary. Half-stride crops preserve that relationship while the
        # ranking still bounds the number sent to the observer.
        stride = (
            max(1, target_size // 2)
            if selection_policy is not None and selection_policy.interface_pairs
            else target_size
        )
        with self._open_slide(str(slide["raw_slide_path"])) as wsi:
            for parent in parents:
                parent_mag = int(parent.magnification.rstrip("x"))
                if target_mag <= parent_mag:
                    continue
                x1, y1, x2, y2 = parent.coordinates
                for y in range(y1, y2, stride):
                    for x in range(x1, x2, stride):
                        box = (x, y, min(x + target_size, x2), min(y + target_size, y2))
                        if box in seen_boxes or box[2] - box[0] < target_size or box[3] - box[1] < target_size:
                            continue
                        seen_boxes.add(box)
                        placeholder = ROI(
                            roi_id="pending",
                            slide_id=str(slide["slide_id"]),
                            coordinates=box,
                            magnification=magnification,
                            retrieval_score=0.0,
                            matched_query=query,
                            parent_roi_id=parent.roi_id,
                        )
                        raw.append((placeholder, read_field(wsi, x, y, target_mag)))
        if not raw:
            return []
        if not self.accelerated_evaluation_enabled:
            image_features = np.asarray(
                self.encoder.image_features([item[1] for item in raw]),
                dtype=np.float32,
            )
            self.metrics["plip_high_image_encoder_calls"] += 1
        else:
            patient_id = str(slide.get("patient_id") or "")
            if patient_id != self._high_feature_cache_patient_id:
                self._high_feature_cache_patient_id = patient_id
                self._high_feature_cache.clear()
            cached_features: list[np.ndarray | None] = []
            missing_indices: list[int] = []
            for index, item in enumerate(raw):
                key = (
                    str(slide.get("slide_id") or ""),
                    magnification,
                    tuple(item[0].coordinates),
                )
                feature = self._high_feature_cache.get(key)
                cached_features.append(feature)
                if feature is None:
                    missing_indices.append(index)
                else:
                    self.metrics["plip_high_feature_cache_hits"] += 1
            if missing_indices:
                encoded = np.asarray(
                    self.encoder.image_features(
                        [raw[index][1] for index in missing_indices]
                    ),
                    dtype=np.float32,
                )
                self.metrics["plip_high_image_encoder_calls"] += 1
                for offset, index in enumerate(missing_indices):
                    feature = encoded[offset]
                    cached_features[index] = feature
                    key = (
                        str(slide.get("slide_id") or ""),
                        magnification,
                        tuple(raw[index][0].coordinates),
                    )
                    self._high_feature_cache[key] = feature
                    self.metrics["plip_high_feature_cache_writes"] += 1
            if any(feature is None for feature in cached_features):
                raise RuntimeError("PLIP high-power feature cache assembly is incomplete")
            image_features = np.stack(
                [
                    np.asarray(feature, dtype=np.float32)
                    for feature in cached_features
                    if feature is not None
                ],
                axis=0,
            )
        image_features /= np.maximum(
            np.linalg.norm(image_features, axis=1, keepdims=True), 1e-12
        )
        text = self._text(query)
        if image_features.shape[1] != len(text):
            raise ValueError(
                f"PLIP high-power dimension mismatch: image={image_features.shape[1]}, text={len(text)}"
            )
        scores = image_features @ text
        tissue_map = self._tissue_map(slide) if plan is not None else None
        summaries: list[dict[str, Any]] = []
        routing_scores = scores.copy()
        policy = None
        outer_interface_scores: np.ndarray | None = None
        perirenal_fat_search = False
        if tissue_map is not None and plan is not None:
            policy = selection_policy
            assert policy is not None
            perirenal_fat_search = (
                policy.reason == "tumor-perirenal-fat interface search"
            )
            summaries = [
                tissue_map.window_summary(item[0].coordinates, policy.target_labels, policy.interface_pairs)
                for item in raw
            ]
            tissue_scores = np.asarray([item["target_score"] for item in summaries], dtype=np.float32)
            interface_scores = np.asarray([item["interface_score"] for item in summaries], dtype=np.float32)
            outer_pairs = tuple(
                pair
                for pair in policy.interface_pairs
                if "artifact_or_background" in pair
            )
            if outer_pairs or perirenal_fat_search:
                outer_interface_scores = np.asarray(
                    [
                        tissue_map.exterior_interface_score(
                            item[0].coordinates,
                            policy.target_labels,
                        )
                        for item in raw
                    ],
                    dtype=np.float32,
                )
                routing_scores = (
                    0.35 * self._percentiles(scores)
                    + 0.20 * self._percentiles(tissue_scores)
                    + 0.25 * self._percentiles(interface_scores)
                    + 0.20 * self._percentiles(outer_interface_scores)
                )
            elif policy.interface_pairs:
                routing_scores = (
                    0.45 * self._percentiles(scores)
                    + 0.20 * self._percentiles(tissue_scores)
                    + 0.35 * self._percentiles(interface_scores)
                )
            else:
                routing_scores = (
                    0.70 * self._percentiles(scores)
                    + 0.20 * self._percentiles(tissue_scores)
                    + 0.10 * self._percentiles(interface_scores)
                )
        global_order = [int(index) for index in np.argsort(routing_scores, kind="stable")[::-1]]
        quality = [high_power_information(item[1]) for item in raw]
        eligible = [index for index in global_order if quality[index][0]]
        if (
            outer_interface_scores is not None
            and not perirenal_fat_search
            and any(
            summaries[index]["target_score"] >= 0.02 for index in eligible
            )
        ):
            eligible = [
                index
                for index in eligible
                if summaries[index]["target_score"] >= 0.02
            ]
        self.metrics["high_candidates_filtered"] += len(raw) - len(eligible)
        # Retain a deterministic escape hatch for unusually pale stains.  The
        # quality score is used only after every field failed the generic gate;
        # it never uses the pathology question or retrieval query.
        if not eligible:
            eligible = sorted(
                global_order,
                key=lambda index: (quality[index][1], float(routing_scores[index])),
                reverse=True,
            )
            self.metrics["high_quality_fallbacks"] += 1
        if policy is not None and policy.interface_pairs:
            microinvasion_search = policy.reason in {
                "tumor-nerve spatial-relation search",
                "tumor-vessel spatial-relation search",
                "microinvasion spatial-relation search",
            }
            nodal_deposit_search = policy.reason == "nodal epithelial-deposit search"
            capsule_search = policy.reason in {
                "tumor-capsule interface search",
                "tumor-perirenal-fat interface search",
            }
            if self.accelerated_evaluation_enabled and (
                outer_interface_scores is not None
                or capsule_search
                or microinvasion_search
                or nodal_deposit_search
            ):
                reserved_count = (
                    self.accelerated_special_40x_top_k
                    if magnification == "40x"
                    else self.accelerated_regional_node_20x_top_k
                    if nodal_deposit_search
                    else self.accelerated_special_20x_top_k
                )
            else:
                reserved_count = (
                    24
                    if outer_interface_scores is not None or capsule_search
                    else (24 if microinvasion_search else (12 if nodal_deposit_search else 8))
                )
            target_count = min(max(self.high_power_top_k, reserved_count), len(raw))
        elif _is_renal_grade_plan(plan):
            target_count = min(max(self.high_power_top_k, 8), len(raw))
        elif plan is not None and plan.decision_semantics in {
            "extremum",
            "quantitative",
            "composite",
        }:
            target_count = min(
                max(self.high_power_top_k, self.accelerated_protected_roi_floor),
                len(raw),
            )
        else:
            target_count = min(self.high_power_top_k, len(raw))
        if policy is not None and policy.reason in {
            "tumor-capsule interface search",
            "tumor-perirenal-fat interface search",
        }:
            # Capsule involvement is often a tiny relation inside a broad
            # low-power window.  The previous outer-interface specialization
            # reserved two children only for outer-routed parents, leaving a
            # global-fusion parent with one child (and occasionally leaving a
            # parent out entirely).  Apply the fixed 24-field budget evenly
            # across every parent at both hierarchy transitions.
            parent_ids = [str(item[0].parent_roi_id) for item in raw]
            # The generic image-quality gate is intentionally conservative,
            # but a real fat-invasion focus may contain only a few tumour
            # nuclei among mostly pale adipocytes.  Before applying the fixed
            # capsule quota, restore the best rejected crops only for parents
            # that would otherwise have fewer than their allotted fields.
            # This preserves complete parent lineage without increasing the
            # observer-call budget or weakening the gate for other tasks.
            all_parent_ids = {parent_ids[index] for index in global_order}
            per_parent_limit = (
                2 if target_count >= 2 * len(all_parent_ids) else 1
            )
            capsule_eligible = list(eligible)
            for parent_id in all_parent_ids:
                present = sum(
                    parent_ids[index] == parent_id
                    for index in capsule_eligible
                )
                if present >= per_parent_limit:
                    continue
                rejected_for_parent = [
                    index
                    for index in global_order
                    if parent_ids[index] == parent_id
                    and index not in capsule_eligible
                ]
                rejected_for_parent.sort(
                    key=lambda index: (
                        quality[index][1],
                        float(routing_scores[index]),
                    ),
                    reverse=True,
                )
                capsule_eligible.extend(
                    rejected_for_parent[: per_parent_limit - present]
                )
            capsule_eligible.sort(
                key=lambda index: float(routing_scores[index]),
                reverse=True,
            )
            ordered = self._capsule_parent_quota_indices(
                capsule_eligible,
                parent_ids,
                routing_scores,
                target_count,
                coordinates=np.asarray(
                    [item[0].coordinates for item in raw], dtype=np.int64
                ),
            )
        elif ensure_parent_diversity:
            # Avoid spending all five high-power calls on one parent field.
            # Keep the best PLIP-ranked patch from each requested parent first,
            # then fill remaining slots by the global ranking.
            ordered = self._parent_diverse_indices(
                eligible,
                [str(item[0].parent_roi_id) for item in raw],
                routing_scores,
                target_count,
            )
        else:
            # Keep two globally ranked fields for continuous boundary
            # tracing, then reserve the remaining budget for distinct 5x
            # parents so a strong intratumoral fibrous focus cannot hide the
            # true outer capsule.
            parent_ids = [str(item[0].parent_roi_id) for item in raw]
            outer_reserved: list[int] = []
            if outer_interface_scores is not None:
                outer_parent_ids = self._outer_interface_parent_ids(parents)
                outer_reserved = self._best_indices_for_parents(
                    eligible,
                    parent_ids,
                    outer_interface_scores,
                    outer_parent_ids,
                    min(12, target_count),
                    per_parent_limit=2,
                )
            remaining = [index for index in eligible if index not in outer_reserved]
            ordered = outer_reserved + self._interface_balanced_indices(
                remaining,
                parent_ids,
                routing_scores,
                target_count - len(outer_reserved),
            )
            self.metrics["high_parent_diversity_bypassed"] += 1
        if len(ordered) < target_count:
            rejected = [index for index in global_order if index not in ordered]
            rejected.sort(
                key=lambda index: (quality[index][1], float(routing_scores[index])),
                reverse=True,
            )
            for index in rejected:
                ordered.append(index)
                if len(ordered) == target_count:
                    break
        output: list[SelectedROI] = []
        parent_by_id = {parent.roi_id: parent for parent in parents}
        for rank, index in enumerate(ordered, 1):
            base, image = raw[index]
            navigation = None
            if summaries and policy is not None:
                summary = summaries[index]
                parent = parent_by_id.get(str(base.parent_roi_id))
                parent_is_outer = bool(
                    parent is not None
                    and parent.roi_id
                    in self._outer_interface_parent_ids([parent])
                )
                parent_is_adipose_interface = bool(
                    parent is not None
                    and parent.roi_id
                    in self._adipose_interface_parent_ids([parent])
                )
                navigation = TissueNavigation(
                    map_available=True,
                    target_score=summary["target_score"],
                    interface_score=summary["interface_score"],
                    heterogeneity=summary["heterogeneity"],
                    map_confidence=summary["map_confidence"],
                    unknown_fraction=summary.get("unknown_fraction", 0.0),
                    fusion_score=float(routing_scores[index]),
                    # Keep the outer-surface identity across 5x -> 20x ->
                    # 40x.  Replacing it with a generic high-power route at
                    # 20x previously made the final 40x pass forget which
                    # parents actually came from the capsule/background edge.
                    selection_route=(
                        "outer_interface_descendant"
                        if parent_is_outer
                        else "adipose_interface_descendant"
                        if parent_is_adipose_interface
                        else "high_power_plip_tissue_fusion"
                    ),
                    target_labels=list(policy.target_labels),
                    class_fractions=summary["class_fractions"],
                )
                self.metrics["tissue_routed_high_selections"] += 1
            roi = base.model_copy(
                update={
                    "roi_id": f"{slide['slide_id']}_{magnification.upper()}_{rank:02d}",
                    "retrieval_score": float(scores[index]),
                    "tissue_navigation": navigation,
                }
            )
            output.append(SelectedROI(roi, image))
        return output

    @staticmethod
    def _outer_interface_parent_ids(parents: list[ROI]) -> set[str]:
        return {
            parent.roi_id
            for parent in parents
            if parent.tissue_navigation is not None
            and parent.tissue_navigation.selection_route
            in {"outer_tissue_interface", "outer_interface_descendant"}
        }

    @staticmethod
    def _adipose_interface_parent_ids(parents: list[ROI]) -> set[str]:
        """Keep a morphology-localized fat lineage visible across zooms."""

        return {
            parent.roi_id
            for parent in parents
            if parent.tissue_navigation is not None
            and parent.tissue_navigation.selection_route
            in {"morphologic_adipose_interface", "adipose_interface_descendant"}
        }
