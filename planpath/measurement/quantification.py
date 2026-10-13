"""Label-free Tissue Map + HistoPLUS measurements for r24.

The module intentionally contains no mapping from Patho-R1 wording such as
``rare`` or ``diffuse`` to a fixed percentage.  Tissue fractions come directly
from the precomputed soft tissue maps.  HistoPLUS is used as a cellular-quality
gate and, for focus counting, as independent tumour-cell confirmation.

The current tissue taxonomy has no signet-ring or hepatoid class.  A signet-ring
answer is therefore explicitly marked as a coarse poorly-cohesive-tissue proxy;
hepatoid proportion is refused.  These limitations remain visible in every
trace instead of being hidden by a plausible-looking language-model answer.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


GASTRIC_TUMOR_LABELS = (
    "gland_forming_tumor",
    "diffuse_or_poorly_cohesive_tumor",
    "mucin_pool",
)
RENAL_TUMOR_LABELS = ("viable_renal_tumor",)


@dataclass(frozen=True)
class QuantificationProfile:
    kind: str
    target: str
    phenotype: str
    unit: str


@dataclass
class SlideTissueMeasurement:
    slide_id: str
    coverage_fraction: float
    known_area_px2: float
    total_area_px2: float
    class_areas_px2: dict[str, float]
    class_fractions: dict[str, float]


@dataclass
class HistoFieldMeasurement:
    slide_id: str
    magnification: str
    coordinates: tuple[int, int, int, int]
    quality: str
    instance_count: int
    valid_area_mm2: float
    grouped_counts: dict[str, int]
    grouped_densities_per_mm2: dict[str, float]
    annotation_path: str

    @property
    def tumor_like_count(self) -> int:
        return int(self.grouped_counts.get("neoplastic_like", 0)) + int(
            self.grouped_counts.get("epithelial", 0)
        )


@dataclass
class QuantitativeResult:
    profile: QuantificationProfile
    status: str
    estimate: float | int | None
    unit: str
    formatted_answer: str
    measurement_method: str
    tissue_coverage_fraction: float
    eligible_slide_count: int
    histoplus_adequate_roi_count: int
    histoplus_valid_area_mm2: float
    histoplus_tumor_like_cell_count: int
    raw_measurements: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["profile"] = asdict(self.profile)
        return value


def quantitative_profile(question: dict[str, Any]) -> QuantificationProfile | None:
    """Route only questions that explicitly request a numerical measurement."""

    text = str(question.get("question") or "").casefold()
    proportion = any(
        term in text
        for term in ("proportion", "percentage", "percent", "fraction", "ratio")
    )
    if proportion and "primary tumor" in text:
        if "signet" in text:
            return QuantificationProfile(
                "component_fraction", "signet-ring cell component", "signet_ring", "percent"
            )
        if "mucin" in text:
            return QuantificationProfile(
                "component_fraction", "mucinous component", "mucinous", "percent"
            )
        if "hepatoid" in text:
            return QuantificationProfile(
                "component_fraction", "hepatoid component", "hepatoid", "percent"
            )
    if proportion and "residual" in text and any(
        term in text for term in ("tumor bed", "tumour bed", "original renal tumor")
    ):
        return QuantificationProfile(
            "residual_tumor_bed_fraction",
            "residual viable renal tumor",
            "viable_renal_tumor",
            "percent",
        )
    if any(term in text for term in ("how many", "number of", "count of")) and any(
        term in text
        for term in ("tumor foci", "tumour foci", "tumor focus", "tumour focus")
    ):
        return QuantificationProfile(
            "focus_count", "independent primary-tumor foci", "tumor", "count"
        )
    if any(term in text for term in ("cells per", "cell density", "cells/mm")):
        return QuantificationProfile(
            "cell_density", "requested cell population", "cell_population", "cells/mm2"
        )
    if any(term in text for term in ("distance", "millimeter", "millimetre", " mm")):
        return QuantificationProfile(
            "spatial_distance", "requested spatial distance", "distance", "mm"
        )
    return None


def _round_percent(value: float, step: int = 5) -> int:
    """Round half upward; Python's banker rounding is unsuitable at 2.5%."""

    bounded = min(100.0, max(0.0, float(value)))
    return min(100, int(step * math.floor(bounded / step + 0.5)))


def _read_tissue_map(slide: dict[str, Any]) -> SlideTissueMeasurement | None:
    raw = str(slide.get("tissue_map_cache") or "")
    path = Path(raw)
    if not raw or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        labels = tuple(json.loads(str(payload["labels_json"].item())))
        probabilities = np.asarray(payload["class_probs"], dtype=np.float64)
        patch_size = np.asarray(payload["patch_size"], dtype=np.float64)
        unknown = np.asarray(payload["unknown"], dtype=bool)
    if probabilities.shape != (len(patch_size), len(labels)):
        raise ValueError(f"invalid tissue map probability shape: {path}")
    areas = np.square(patch_size)
    known = ~unknown
    known_area = float(areas[known].sum())
    total_area = float(areas.sum())
    weighted = (probabilities[known] * areas[known, None]).sum(axis=0)
    class_areas = {label: float(weighted[index]) for index, label in enumerate(labels)}
    denominator = max(float(weighted.sum()), 1e-12)
    return SlideTissueMeasurement(
        slide_id=str(slide["slide_id"]),
        coverage_fraction=known_area / max(total_area, 1e-12),
        known_area_px2=known_area,
        total_area_px2=total_area,
        class_areas_px2=class_areas,
        class_fractions={label: value / denominator for label, value in class_areas.items()},
    )


def _load_histo_fields(
    cell_seg_root: Path,
    cancer: str,
    patient_id: str,
    slide_ids: set[str],
) -> list[HistoFieldMeasurement]:
    root = cell_seg_root / cancer / patient_id
    if not root.is_dir():
        return []
    deduplicated: dict[
        tuple[str, str, tuple[int, int, int, int]], HistoFieldMeasurement
    ] = {}
    for path in root.rglob("*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            slide_id = str(payload.get("slide_id") or "")
            if slide_id not in slide_ids:
                continue
            roi = payload.get("roi") or {}
            summary = payload.get("summary") or {}
            coordinates = tuple(int(item) for item in roi.get("coordinates", ()))
            if len(coordinates) != 4 or str(summary.get("source")) != "HistoPLUS":
                continue
            field = HistoFieldMeasurement(
                slide_id=slide_id,
                magnification=str((roi.get("magnification") or "")),
                coordinates=coordinates,  # type: ignore[arg-type]
                quality=str(summary.get("quality") or "unavailable"),
                instance_count=int(summary.get("instance_count") or 0),
                valid_area_mm2=float(summary.get("valid_area_mm2") or 0.0),
                grouped_counts={
                    str(key): int(value)
                    for key, value in (summary.get("grouped_counts") or {}).items()
                },
                grouped_densities_per_mm2={
                    str(key): float(value)
                    for key, value in (
                        summary.get("grouped_densities_per_mm2") or {}
                    ).items()
                },
                annotation_path=str(path),
            )
            key = (field.slide_id, field.magnification, field.coordinates)
            previous = deduplicated.get(key)
            if previous is None or (
                field.quality == "adequate",
                field.instance_count,
            ) > (
                previous.quality == "adequate",
                previous.instance_count,
            ):
                deduplicated[key] = field
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return list(deduplicated.values())


def _aggregate_areas(
    measurements: Iterable[SlideTissueMeasurement], labels: Iterable[str]
) -> dict[str, float]:
    output = {label: 0.0 for label in labels}
    for measurement in measurements:
        for label in output:
            output[label] += float(measurement.class_areas_px2.get(label, 0.0))
    return output


def _connected_target_components(
    path: Path,
    labels: Iterable[str],
    *,
    probability_threshold: float = 0.10,
    min_patch_count: int = 5,
) -> list[dict[str, Any]]:
    """Mirror the v7 navigation candidate extraction without importing v7."""

    with np.load(path, allow_pickle=False) as payload:
        names = tuple(json.loads(str(payload["labels_json"].item())))
        x = np.asarray(payload["x"], dtype=np.int64)
        y = np.asarray(payload["y"], dtype=np.int64)
        size = np.asarray(payload["patch_size"], dtype=np.int64)
        probabilities = np.asarray(payload["class_probs"], dtype=np.float64)
    ids = [names.index(label) for label in labels if label in names]
    if not ids or not len(x):
        return []
    scores = probabilities[:, ids].sum(axis=1)
    active = np.flatnonzero(scores >= probability_threshold)
    if not len(active):
        return []
    step = max(1, int(np.median(size)))
    origin_x, origin_y = int(x.min()), int(y.min())
    grid_to_index = {
        (
            int(round((int(x[index]) - origin_x) / step)),
            int(round((int(y[index]) - origin_y) / step)),
        ): int(index)
        for index in active
    }
    remaining = set(grid_to_index)
    groups: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        members: list[int] = []
        while queue:
            gx, gy = queue.popleft()
            members.append(grid_to_index[(gx, gy)])
            for dx, dy in (
                (-1, -1), (0, -1), (1, -1),
                (-1, 0), (1, 0),
                (-1, 1), (0, 1), (1, 1),
            ):
                neighbor = (gx + dx, gy + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        if len(members) >= min_patch_count:
            groups.append(members)
    output: list[dict[str, Any]] = []
    for members in groups:
        indices = np.asarray(members, dtype=np.int64)
        output.append(
            {
                "box": (
                    int(x[indices].min()),
                    int(y[indices].min()),
                    int((x[indices] + size[indices]).max()),
                    int((y[indices] + size[indices]).max()),
                ),
                "patch_count": int(len(indices)),
                "score": float(scores[indices].sum()),
            }
        )
    return sorted(output, key=lambda item: item["score"], reverse=True)


def _inside(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= point[0] < x2 and y1 <= point[1] < y2


class R24Quantifier:
    """Deterministic measurement branch; it never receives a reference answer."""

    def __init__(
        self,
        cell_seg_root: Path = Path("artifacts/cell_seg"),
        *,
        rounding_percent: int = 5,
        min_tissue_coverage: float = 0.80,
        min_histoplus_rois: int = 4,
        min_histoplus_tumor_cells: int = 40,
        min_focus_tumor_cells: int = 100,
    ) -> None:
        self.cell_seg_root = Path(cell_seg_root)
        self.rounding_percent = int(rounding_percent)
        self.min_tissue_coverage = float(min_tissue_coverage)
        self.min_histoplus_rois = int(min_histoplus_rois)
        self.min_histoplus_tumor_cells = int(min_histoplus_tumor_cells)
        self.min_focus_tumor_cells = int(min_focus_tumor_cells)

    def _base_measurements(
        self, slides: list[dict[str, Any]]
    ) -> tuple[list[SlideTissueMeasurement], list[HistoFieldMeasurement], dict[str, Any]]:
        tissue = [item for slide in slides if (item := _read_tissue_map(slide)) is not None]
        cancer = str(slides[0].get("cancer") or "unknown") if slides else "unknown"
        patient_id = str(slides[0].get("patient_id") or "unknown") if slides else "unknown"
        slide_ids = {str(slide["slide_id"]) for slide in slides}
        histo = _load_histo_fields(self.cell_seg_root, cancer, patient_id, slide_ids)
        adequate = [item for item in histo if item.quality == "adequate"]
        summary = {
            "map_slide_count": len(tissue),
            "expected_slide_count": len(slides),
            "map_slide_coverage": len(tissue) / max(len(slides), 1),
            "map_known_area_coverage": (
                sum(item.known_area_px2 for item in tissue)
                / max(sum(item.total_area_px2 for item in tissue), 1e-12)
            ),
            "histoplus_adequate_roi_count": len(adequate),
            "histoplus_valid_area_mm2": sum(item.valid_area_mm2 for item in adequate),
            "histoplus_tumor_like_cell_count": sum(
                item.tumor_like_count for item in adequate
            ),
            "histoplus_analyzed_slide_count": len(
                {item.slide_id for item in adequate}
            ),
        }
        summary["tissue_coverage_fraction"] = (
            summary["map_slide_coverage"] * summary["map_known_area_coverage"]
        )
        return tissue, adequate, summary

    def _gate_reasons(self, summary: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if summary["tissue_coverage_fraction"] < self.min_tissue_coverage:
            reasons.append("Tissue Map coverage is below the configured minimum")
        if summary["histoplus_adequate_roi_count"] < self.min_histoplus_rois:
            reasons.append("too few adequate HistoPLUS fields")
        if summary["histoplus_tumor_like_cell_count"] < self.min_histoplus_tumor_cells:
            reasons.append("too few confidence-qualified HistoPLUS tumor-like cells")
        return reasons

    def measure(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        profile: QuantificationProfile | None = None,
    ) -> QuantitativeResult:
        profile = profile or quantitative_profile(question)
        if profile is None:
            raise ValueError("r24 quantifier received a non-quantitative question")
        tissue, histo, summary = self._base_measurements(slides)
        reasons = self._gate_reasons(summary)
        common = {
            "profile": profile,
            "tissue_coverage_fraction": float(summary["tissue_coverage_fraction"]),
            "eligible_slide_count": len(tissue),
            "histoplus_adequate_roi_count": int(summary["histoplus_adequate_roi_count"]),
            "histoplus_valid_area_mm2": float(summary["histoplus_valid_area_mm2"]),
            "histoplus_tumor_like_cell_count": int(
                summary["histoplus_tumor_like_cell_count"]
            ),
        }
        if profile.kind == "component_fraction":
            tumor_labels = GASTRIC_TUMOR_LABELS
            eligible = [
                item
                for item in tissue
                if sum(item.class_areas_px2.get(label, 0.0) for label in tumor_labels)
                / max(item.known_area_px2, 1e-12)
                >= 0.02
            ]
            areas = _aggregate_areas(eligible, tumor_labels)
            denominator = sum(areas.values())
            if denominator <= 0:
                reasons.append("no measurable gastric primary-tumor denominator")
            if profile.phenotype == "hepatoid":
                reasons.append("Tissue Map and HistoPLUS have no hepatoid-specific class")
                estimate = None
                status = "insufficient_target_specificity"
                answer = "Insufficient quantitative specificity"
                method = "no unsupported tissue-class substitution"
            else:
                target_label = (
                    "mucin_pool"
                    if profile.phenotype == "mucinous"
                    else "diffuse_or_poorly_cohesive_tumor"
                )
                raw_percent = 100.0 * areas.get(target_label, 0.0) / max(denominator, 1e-12)
                estimate = _round_percent(raw_percent, self.rounding_percent)
                answer = f"Approximately {estimate}%"
                status = (
                    "authorized_direct_measurement"
                    if profile.phenotype == "mucinous"
                    else "authorized_coarse_proxy"
                )
                method = (
                    "whole-map extracellular mucin-pool area / gastric tumor area"
                    if profile.phenotype == "mucinous"
                    else "whole-map diffuse-or-poorly-cohesive area / gastric tumor area"
                )
            if reasons and estimate is not None:
                status = "insufficient_measurement_coverage"
                answer = "Insufficient quantitative evidence"
                estimate = None
            limitations = [
                "HistoPLUS supplies cellular validity but its generic cell labels do not establish a gastric subtype."
            ]
            if profile.phenotype == "signet_ring":
                limitations.append(
                    "Diffuse/poorly-cohesive tissue is a measured but non-specific proxy for signet-ring morphology."
                )
            return QuantitativeResult(
                status=status,
                estimate=estimate,
                unit=profile.unit,
                formatted_answer=answer,
                measurement_method=method,
                raw_measurements={
                    "tumor_class_areas_px2": areas,
                    "raw_percent_before_rounding": (
                        raw_percent if profile.phenotype != "hepatoid" else None
                    ),
                    "rounding_percent": self.rounding_percent,
                    **summary,
                },
                reasons=reasons,
                limitations=limitations,
                **common,
            )
        if profile.kind == "focus_count":
            by_slide = {item.slide_id: [] for item in tissue}
            for field_item in histo:
                by_slide.setdefault(field_item.slide_id, []).append(field_item)
            counts: dict[str, int] = {}
            component_audit: dict[str, list[dict[str, Any]]] = {}
            for slide in slides:
                raw = str(slide.get("tissue_map_cache") or "")
                if not raw or not Path(raw).is_file():
                    continue
                components = _connected_target_components(
                    Path(raw), RENAL_TUMOR_LABELS
                )
                audited: list[dict[str, Any]] = []
                confirmed = 0
                for component in components:
                    matching = []
                    for field_item in by_slide.get(str(slide["slide_id"]), []):
                        x1, y1, x2, y2 = field_item.coordinates
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        if _inside(component["box"], center):
                            matching.append(field_item)
                    strongest = max(
                        (item.tumor_like_count for item in matching), default=0
                    )
                    passed = strongest >= self.min_focus_tumor_cells
                    confirmed += int(passed)
                    audited.append(
                        {
                            **component,
                            "matching_histoplus_fields": len(matching),
                            "maximum_tumor_like_cell_count": strongest,
                            "confirmed": passed,
                        }
                    )
                counts[str(slide["slide_id"])] = confirmed
                component_audit[str(slide["slide_id"])] = audited
            estimate = max(counts.values(), default=0)
            if estimate < 1:
                reasons.append("no Tissue component passed HistoPLUS confirmation")
            if reasons:
                estimate_value: int | None = None
                answer = "Insufficient quantitative evidence"
                status = "insufficient_measurement_coverage"
            else:
                estimate_value = estimate
                words = {1: "One focus", 2: "Two foci", 3: "Three foci", 4: "Four foci"}
                answer = words.get(estimate, f"{estimate} foci")
                status = "authorized_direct_measurement"
            return QuantitativeResult(
                status=status,
                estimate=estimate_value,
                unit="count",
                formatted_answer=answer,
                measurement_method=(
                    "maximum per-section connected viable-tumor components with "
                    "HistoPLUS tumor-like-cell confirmation"
                ),
                raw_measurements={
                    "confirmed_components_by_slide": counts,
                    "component_audit": component_audit,
                    "minimum_focus_tumor_like_cells": self.min_focus_tumor_cells,
                    **summary,
                },
                reasons=reasons,
                limitations=[
                    "The count is a 2-D per-section maximum; serial sections are not summed."
                ],
                **common,
            )
        if profile.kind == "residual_tumor_bed_fraction":
            selected = [
                item
                for item in tissue
                if item.class_fractions.get("interstitium_or_fibrosis", 0.0) >= 0.40
                and item.class_fractions.get("viable_renal_tumor", 0.0) >= 0.005
            ]
            labels = (
                "viable_renal_tumor",
                "interstitium_or_fibrosis",
                "immune_or_lymphoid",
                "necrosis_or_debris",
            )
            areas = _aggregate_areas(selected, labels)
            denominator = sum(areas.values())
            raw_percent = 100.0 * areas["viable_renal_tumor"] / max(denominator, 1e-12)
            estimate = _round_percent(raw_percent, self.rounding_percent)
            if not selected:
                reasons.append("no fibrosis-rich original-tumor-bed sampling stratum")
            if reasons:
                estimate_value: int | None = None
                answer = "Insufficient quantitative evidence"
                status = "insufficient_measurement_coverage"
            else:
                estimate_value = estimate
                answer = f"Approximately {estimate}%"
                status = "authorized_direct_measurement"
            return QuantitativeResult(
                status=status,
                estimate=estimate_value,
                unit="percent",
                formatted_answer=answer,
                measurement_method=(
                    "viable renal-tumor area / fibrosis-rich treatment-bed area, "
                    "with HistoPLUS cellular confirmation"
                ),
                raw_measurements={
                    "selected_tumor_bed_slide_ids": [item.slide_id for item in selected],
                    "tumor_bed_class_areas_px2": areas,
                    "raw_percent_before_rounding": raw_percent,
                    "rounding_percent": self.rounding_percent,
                    **summary,
                },
                reasons=reasons,
                limitations=[
                    "The treatment-bed stratum is inferred from tissue composition and has no gross specimen label."
                ],
                **common,
            )
        reasons.append(
            f"{profile.kind} is recognized but has no validated Tissue Map + HistoPLUS estimator"
        )
        return QuantitativeResult(
            status="unsupported_measurement",
            estimate=None,
            unit=profile.unit,
            formatted_answer="Insufficient quantitative specificity",
            measurement_method="recognized-and-refused",
            raw_measurements=summary,
            reasons=reasons,
            limitations=[],
            **common,
        )
