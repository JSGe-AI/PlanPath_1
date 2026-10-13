"""Lazy, fail-open HistoPLUS analysis of selected 20x/40x WSI regions."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .cell_store import PrecomputedCellStore, cell_store_path
from .config import AgentConfig
from .plip import SelectedROI
from .schemas import CellularAnalysis, EvidencePlan
from .tissue_routing import routing_policy
from .quantification import is_quantitative_measurement_plan


HISTOPLUS_ADAPTER_VERSION = "planpath-quantitative-histoplus-roi-v3"
HISTOPLUS_CLASS_GROUPS = {
    "Cancer cell": "neoplastic_like",
    "Epithelial": "epithelial",
    "Lymphocytes": "immune",
    "Plasmocytes": "immune",
    "Eosinophils": "immune",
    "Neutrophils": "immune",
    "Macrophages": "immune",
    "Fibroblasts": "stromal",
    "Minor Stromal Cell": "stromal",
    "Muscle Cell": "vascular_muscular",
    "Endothelial Cell": "vascular_muscular",
    "Red blood cell": "vascular_muscular",
    "Apoptotic Body": "cellular_events",
    "Mitotic Figures": "cellular_events",
}
OVERLAY_COLORS = {
    "neoplastic_like": (230, 25, 75),
    "epithelial": (60, 180, 75),
    "immune": (70, 240, 240),
    "stromal": (145, 30, 180),
    "vascular_muscular": (66, 99, 235),
    "cellular_events": (255, 225, 25),
}


def _objective_mpp(slide: Any) -> float:
    for key in ("openslide.mpp-x", "aperio.MPP"):
        raw = slide.properties.get(key)
        try:
            value = float(raw)
            if 0.1 <= value <= 2.0:
                return value
        except (TypeError, ValueError):
            pass
    try:
        objective = float(slide.properties.get("openslide.objective-power"))
        if objective > 0:
            return 10.0 / objective
    except (TypeError, ValueError):
        pass
    return 0.25


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _contact_fraction(
    source: np.ndarray, target: np.ndarray, *, distance_um: float, mpp: float
) -> float | None:
    if not len(source) or not len(target):
        return None
    threshold2 = (distance_um / mpp) ** 2
    contacts = 0
    for start in range(0, len(source), 256):
        delta = source[start : start + 256, None, :] - target[None, :, :]
        contacts += int((np.square(delta).sum(axis=2).min(axis=1) <= threshold2).sum())
    return contacts / len(source)


class HistoPLUSAnalyzer:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._segmentors: dict[float, Any] = {}
        self._fatal_error: str | None = None
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.skipped = 0
        self.precomputed_hits = 0
        self.precomputed_misses = 0
        self.batch_forward_calls = 0
        self.audit: list[dict[str, Any]] = []
        self.model_root = config.histoplus_model_root
        self.cache_root = config.cell_seg_root

    @property
    def metrics(self) -> dict[str, int]:
        return {
            "histoplus_calls": self.calls,
            "histoplus_cache_hits": self.cache_hits,
            "histoplus_failures": self.failures,
            "histoplus_skipped": self.skipped,
            "histoplus_precomputed_hits": self.precomputed_hits,
            "histoplus_precomputed_misses": self.precomputed_misses,
            "histoplus_batch_forward_calls": self.batch_forward_calls,
        }

    def _gpu(self) -> int:
        if self.config.device == "cpu":
            return -1
        if ":" in self.config.device:
            return int(self.config.device.rsplit(":", 1)[1])
        return 0

    def _segmentor(self, mpp: float) -> Any:
        if mpp in self._segmentors:
            return self._segmentors[mpp]
        os.environ.setdefault("HISTOWMICS_HOME", str(self.model_root))
        vendored_root = Path(__file__).resolve().parents[2] / "vendor" / "histoplus"
        if vendored_root.is_dir() and str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        try:
            from histoplus.helpers.hub import (
                histoplus_cellvit_segmentor_20x,
                histoplus_cellvit_segmentor_40x,
            )
            from histoplus.helpers.segmentor import CellViTSegmentor
            from histoplus.helpers.nn.cellvit import CellViT
        except ImportError as error:
            raise RuntimeError(
                "HistoPLUS is not installed or its dependencies are unavailable; "
                "run scripts/install_histoplus.sh"
            ) from error
        model_fn = (
            histoplus_cellvit_segmentor_20x if mpp == 0.5 else histoplus_cellvit_segmentor_40x
        )
        # Production inference must not unexpectedly contact the network.  The
        # installation script downloads gated weights explicitly after access
        # has been granted; fail quickly and let the v6 workflow fall back when
        # a weight is absent.
        from huggingface_hub import hf_hub_download

        direct_weight = self.model_root / model_fn.filename
        if direct_weight.is_file():
            weight_path = str(direct_weight)
        else:
            weight_path = hf_hub_download(
                repo_id="Owkin-Bioptimus/histoplus",
                filename=model_fn.filename,
                cache_dir=str(self.model_root / "hf_cache"),
                local_files_only=True,
            )
        import torch

        model = CellViT(
            cell_type_mapping=model_fn.class_mapping,
            mpp=model_fn.mpp,
            output_layers=model_fn.output_layers,
            train_image_size=model_fn.train_image_size,
            inference_image_size=self.config.histoplus_inference_image_size,
        )
        model.load_state_dict(
            torch.load(weight_path, weights_only=False, map_location="cpu")
        )
        model.eval()
        segmentor = CellViTSegmentor(
            model=model,
            segmentor_name=model_fn.name,
            mean=model_fn.mean,
            std=model_fn.std,
            gpu=self._gpu(),
            mixed_precision=self.config.histoplus_mixed_precision,
        )
        self._segmentors[mpp] = segmentor
        return segmentor

    def _read_context(
        self, slide: Any, selected: SelectedROI, mpp: float
    ) -> tuple[Image.Image, tuple[int, int], float]:
        base_mpp = _objective_mpp(slide)
        output_size = self.config.histoplus_inference_image_size
        field_level0 = max(1, int(round(output_size * mpp / base_mpp)))
        x1, y1, x2, y2 = selected.roi.coordinates
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        read_x = max(0, min(center_x - field_level0 // 2, slide.dimensions[0] - field_level0))
        read_y = max(0, min(center_y - field_level0 // 2, slide.dimensions[1] - field_level0))
        target_downsample = mpp / base_mpp
        level = slide.get_best_level_for_downsample(max(1.0, target_downsample))
        native_downsample = float(slide.level_downsamples[level])
        native_size = max(1, int(math.ceil(field_level0 / native_downsample)))
        image = slide.read_region((read_x, read_y), level, (native_size, native_size)).convert("RGB")
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)
        return image, (read_x, read_y), base_mpp

    def _model_signature(self, mpp: float) -> str:
        filename = (
            "histoplus_cellvit_segmentor_20x.pt"
            if mpp == 0.5
            else "histoplus_cellvit_segmentor_40x.pt"
        )
        path = self.model_root / filename
        stat_signature = (
            f"{path.stat().st_size}|{path.stat().st_mtime_ns}"
            if path.is_file()
            else "hf-cache"
        )
        return hashlib.sha256(
            (
                f"HistoPLUS-1.0|{mpp}|{stat_signature}|"
                f"input={self.config.histoplus_inference_image_size}|"
                f"{HISTOPLUS_ADAPTER_VERSION}"
            ).encode()
        ).hexdigest()

    def _cache_key(
        self,
        slide_record: dict[str, Any],
        selected: SelectedROI,
        mpp: float,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(self._model_signature(mpp).encode())
        digest.update(str(self.config.histoplus_min_type_confidence).encode())
        digest.update(str(self.config.histoplus_save_overlay).encode())
        digest.update(str(slide_record.get("cancer") or "").encode())
        digest.update(str(slide_record.get("slide_id") or selected.roi.slide_id).encode())
        raw_path = Path(str(slide_record.get("raw_slide_path") or ""))
        try:
            stat = raw_path.stat()
            source_signature = f"{raw_path}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            source_signature = f"{raw_path}|missing"
        digest.update(source_signature.encode())
        digest.update(str(selected.roi.coordinates).encode())
        return digest.hexdigest()

    def _paths(
        self,
        slide_record: dict[str, Any],
        selected: SelectedROI,
        key: str,
    ) -> tuple[Path, Path]:
        x1, y1, x2, y2 = selected.roi.coordinates
        directory = (
            self.cache_root
            / str(slide_record.get("cancer") or "unknown")
            / str(slide_record.get("patient_id") or "unknown")
            / str(slide_record.get("slide_id") or selected.roi.slide_id)
            / selected.roi.magnification
        )
        stem = f"{x1}_{y1}_{x2 - x1}_{y2 - y1}_{key[:16]}"
        return directory / f"{stem}.json.gz", directory / f"{stem}.png"

    @staticmethod
    def _read_patch_cache(path: Path) -> dict[str, Any]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_patch_cache(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _summarize_instances(
        self,
        selected: SelectedROI,
        instances: list[dict[str, Any]],
        *,
        base_mpp: float,
        input_mpp: float,
        model_name: str,
        model_signature: str,
        annotation_path: str | None,
        overlay_path: str | None,
        precomputed: bool = False,
    ) -> CellularAnalysis:
        accepted = [
            item
            for item in instances
            if float(item["confidence"]) >= self.config.histoplus_min_type_confidence
        ]
        centroids_by_type: dict[str, list[list[float]]] = {}
        for item in accepted:
            centroids_by_type.setdefault(str(item["cell_type"]), []).append(
                [float(value) for value in item["centroid_level0"]]
            )
        areas = [float(item["area_um2"]) for item in accepted]
        counts = Counter(str(item["cell_type"]) for item in accepted)
        grouped = Counter(
            {
                group: sum(
                    count
                    for name, count in counts.items()
                    if HISTOPLUS_CLASS_GROUPS.get(name, "other") == group
                )
                for group in set(HISTOPLUS_CLASS_GROUPS.values()) | {"other"}
            }
        )
        grouped = Counter({key: value for key, value in grouped.items() if value})
        x1, y1, x2, y2 = selected.roi.coordinates
        area_mm2 = (x2 - x1) * (y2 - y1) * base_mpp * base_mpp / 1_000_000
        densities = {key: value / max(area_mm2, 1e-12) for key, value in counts.items()}
        grouped_densities = {
            key: value / max(area_mm2, 1e-12) for key, value in grouped.items()
        }
        confidence_mean = (
            float(np.mean([float(item["confidence"]) for item in accepted]))
            if accepted
            else 0.0
        )
        quantiles = (
            {
                name: float(value)
                for name, value in zip(
                    ("q25", "q50", "q75", "q95"),
                    np.quantile(areas, [0.25, 0.5, 0.75, 0.95]),
                    strict=True,
                )
            }
            if areas
            else {}
        )
        cancer = np.asarray(
            centroids_by_type.get("Cancer cell", []), dtype=np.float32
        ).reshape(-1, 2)
        immune_names = (
            "Lymphocytes",
            "Plasmocytes",
            "Eosinophils",
            "Neutrophils",
            "Macrophages",
        )
        immune = np.asarray(
            [point for name in immune_names for point in centroids_by_type.get(name, [])],
            dtype=np.float32,
        ).reshape(-1, 2)
        endothelial = np.asarray(
            centroids_by_type.get("Endothelial Cell", []), dtype=np.float32
        ).reshape(-1, 2)
        limitations = [
            "HistoPLUS cell types are model-derived and are not direct diagnostic evidence."
        ]
        if precomputed:
            limitations.append(
                "Measurements were queried from the precomputed slide-level cell store."
            )
        quality = "adequate" if len(accepted) >= 10 else "limited"
        if quality == "limited":
            limitations.append(
                "Fewer than 10 confidence-qualified instances were present in the linked ROI."
            )
        return CellularAnalysis(
            model_name=model_name,
            model_signature=model_signature,
            input_mpp=input_mpp,
            quality=quality,
            instance_count=len(accepted),
            valid_area_mm2=area_mm2,
            class_counts=dict(sorted(counts.items())),
            class_densities_per_mm2={
                key: round(value, 3) for key, value in sorted(densities.items())
            },
            grouped_counts=dict(sorted(grouped.items())),
            grouped_densities_per_mm2={
                key: round(value, 3)
                for key, value in sorted(grouped_densities.items())
            },
            mean_type_confidence=confidence_mean,
            nuclear_area_um2_quantiles={
                key: round(value, 3) for key, value in quantiles.items()
            },
            cancer_immune_contact_fraction=_contact_fraction(
                cancer, immune, distance_um=20, mpp=base_mpp
            ),
            cancer_endothelial_contact_fraction=_contact_fraction(
                cancer, endothelial, distance_um=20, mpp=base_mpp
            ),
            annotation_path=annotation_path,
            overlay_path=overlay_path,
            limitations=limitations,
        )

    def _precomputed_analysis(
        self,
        slide_record: dict[str, Any],
        selected: SelectedROI,
        mpp: float,
    ) -> CellularAnalysis | None:
        if not self.config.precomputed_cell_seg_enabled:
            return None
        if not slide_record.get("patient_id") or not slide_record.get("slide_id"):
            self.precomputed_misses += 1
            return None
        path = cell_store_path(
            self.config.cell_seg_root,
            slide_record,
            selected.roi.magnification,
        )
        if not path.is_file():
            self.precomputed_misses += 1
            return None
        try:
            query = PrecomputedCellStore(path).query(
                selected.roi.coordinates,
                expected_model_signature=self._model_signature(mpp),
            )
        except Exception as error:
            self.precomputed_misses += 1
            self.audit.append(
                {
                    "roi_id": selected.roi.roi_id,
                    "status": "precomputed_store_invalid",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            return None
        if query is None:
            self.precomputed_misses += 1
            return None
        self.precomputed_hits += 1
        self.audit.append(
            {
                "roi_id": selected.roi.roi_id,
                "status": "precomputed_hit",
                "cell_store": str(path),
                "instances": len(query.instances),
            }
        )
        return self._summarize_instances(
            selected,
            query.instances,
            base_mpp=query.base_mpp,
            input_mpp=query.input_mpp,
            model_name=query.model_name,
            model_signature=query.model_signature,
            annotation_path=str(query.path),
            overlay_path=None,
            precomputed=True,
        )

    @staticmethod
    def _prediction_instances(
        selected: SelectedROI,
        prediction: Any,
        origin: tuple[int, int],
        base_mpp: float,
        input_mpp: float,
    ) -> list[dict[str, Any]]:
        scale_level0 = input_mpp / base_mpp
        x1, y1, x2, y2 = selected.roi.coordinates
        instances: list[dict[str, Any]] = []
        for contour, centroid, cell_type, confidence in zip(
            prediction.contours,
            prediction.centroids,
            prediction.cell_types,
            prediction.cell_type_probabilities,
            strict=True,
        ):
            cx, cy = float(centroid[0]), float(centroid[1])
            global_x = origin[0] + cx * scale_level0
            global_y = origin[1] + cy * scale_level0
            if not (x1 <= global_x < x2 and y1 <= global_y < y2):
                continue
            points = np.asarray(contour, dtype=np.float32)
            global_points = np.column_stack(
                [
                    origin[0] + points[:, 0] * scale_level0,
                    origin[1] + points[:, 1] * scale_level0,
                ]
            )
            instances.append(
                {
                    "cell_type": str(cell_type),
                    "confidence": round(float(confidence), 6),
                    "centroid_level0": [round(global_x, 3), round(global_y, 3)],
                    "area_um2": round(_polygon_area(points) * input_mpp * input_mpp, 3),
                    "contour_level0": np.round(global_points, 2).tolist(),
                }
            )
        return instances

    def _persist_result(
        self,
        slide_record: dict[str, Any],
        selected: SelectedROI,
        image: Image.Image,
        origin: tuple[int, int],
        base_mpp: float,
        input_mpp: float,
        model_name: str,
        instances: list[dict[str, Any]],
        cache_key: str,
        annotation_path: Path,
        overlay_path: Path,
    ) -> CellularAnalysis:
        saved_overlay_path = str(overlay_path) if self.config.histoplus_save_overlay else None
        summary = self._summarize_instances(
            selected,
            instances,
            base_mpp=base_mpp,
            input_mpp=input_mpp,
            model_name=model_name,
            model_signature=self._model_signature(input_mpp),
            annotation_path=str(annotation_path),
            overlay_path=saved_overlay_path,
        )
        if self.config.histoplus_save_overlay:
            rendered = image.copy()
            draw = ImageDraw.Draw(rendered)
            scale_level0 = input_mpp / base_mpp
            for item in instances:
                group = HISTOPLUS_CLASS_GROUPS.get(str(item["cell_type"]), "other")
                color = OVERLAY_COLORS.get(group, (255, 255, 255))
                global_x, global_y = item["centroid_level0"]
                local_x = (float(global_x) - origin[0]) / scale_level0
                local_y = (float(global_y) - origin[1]) / scale_level0
                draw.ellipse(
                    (local_x - 2, local_y - 2, local_x + 2, local_y + 2),
                    outline=color,
                    width=1,
                )
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(overlay_path)
        self._write_patch_cache(
            annotation_path,
            {
                "adapter_version": HISTOPLUS_ADAPTER_VERSION,
                "cache_key": cache_key,
                "cache_kind": "question_routed_online_patch",
                "slide_id": str(slide_record.get("slide_id") or selected.roi.slide_id),
                "roi": selected.roi.model_dump(mode="json"),
                "summary": summary.model_dump(mode="json"),
                "instances": instances,
            },
        )
        return summary

    def analyze_batch(
        self,
        slide_record: dict[str, Any],
        selected: list[SelectedROI],
    ) -> dict[str, CellularAnalysis]:
        """Resolve stores/caches first, then infer remaining ROIs in GPU batches."""

        output: dict[str, CellularAnalysis] = {}
        pending_by_mpp: dict[float, list[tuple[SelectedROI, str, Path, Path]]] = {}
        for item in selected:
            magnification = item.roi.magnification
            if magnification not in {"20x", "40x"}:
                self.skipped += 1
                continue
            input_mpp = 0.5 if magnification == "20x" else 0.25
            precomputed = self._precomputed_analysis(slide_record, item, input_mpp)
            if precomputed is not None:
                output[item.roi.roi_id] = precomputed
                continue
            cache_key = self._cache_key(slide_record, item, input_mpp)
            annotation_path, overlay_path = self._paths(slide_record, item, cache_key)
            if annotation_path.is_file():
                try:
                    payload = self._read_patch_cache(annotation_path)
                    if (
                        payload.get("adapter_version") != HISTOPLUS_ADAPTER_VERSION
                        or payload.get("cache_key") != cache_key
                    ):
                        raise ValueError("patch cache signature mismatch")
                    analysis = CellularAnalysis.model_validate(payload["summary"])
                    output[item.roi.roi_id] = analysis
                    self.cache_hits += 1
                    self.audit.append(
                        {
                            "roi_id": item.roi.roi_id,
                            "status": "patch_cache_hit",
                            "annotation_path": str(annotation_path),
                            "instances": analysis.instance_count,
                        }
                    )
                    continue
                except Exception as error:
                    self.audit.append(
                        {
                            "roi_id": item.roi.roi_id,
                            "status": "patch_cache_invalid",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            pending_by_mpp.setdefault(input_mpp, []).append(
                (item, cache_key, annotation_path, overlay_path)
            )

        if self._fatal_error is not None:
            self.skipped += sum(len(items) for items in pending_by_mpp.values())
            return output

        for input_mpp, pending in pending_by_mpp.items():
            completed: set[str] = set()
            try:
                import openslide
                import torch

                contexts: list[tuple[Image.Image, tuple[int, int], float]] = []
                with openslide.OpenSlide(str(slide_record["raw_slide_path"])) as slide:
                    contexts = [
                        self._read_context(slide, item, input_mpp)
                        for item, _, _, _ in pending
                    ]
                segmentor = self._segmentor(input_mpp)
                batch_size = self.config.histoplus_batch_size
                for start in range(0, len(pending), batch_size):
                    batch_pending = pending[start : start + batch_size]
                    batch_contexts = contexts[start : start + batch_size]
                    tensor = torch.stack(
                        [segmentor.transform(image) for image, _, _ in batch_contexts]
                    )
                    with torch.inference_mode():
                        raw_outputs = segmentor.forward(tensor)
                    self.batch_forward_calls += 1
                    arrays = {
                        name: value.detach().cpu().numpy()
                        for name, value in raw_outputs.items()
                    }
                    predictions = segmentor.get_postprocess_fn()(arrays)
                    for pending_item, context, prediction in zip(
                        batch_pending, batch_contexts, predictions, strict=True
                    ):
                        item, cache_key, annotation_path, overlay_path = pending_item
                        image, origin, base_mpp = context
                        instances = self._prediction_instances(
                            item, prediction, origin, base_mpp, input_mpp
                        )
                        analysis = self._persist_result(
                            slide_record,
                            item,
                            image,
                            origin,
                            base_mpp,
                            input_mpp,
                            str(segmentor.segmentor_name),
                            instances,
                            cache_key,
                            annotation_path,
                            overlay_path,
                        )
                        output[item.roi.roi_id] = analysis
                        completed.add(item.roi.roi_id)
                        self.calls += 1
                        self.audit.append(
                            {
                                "roi_id": item.roi.roi_id,
                                "status": "batch_inference_success",
                                "batch_size": len(batch_pending),
                                "instances": analysis.instance_count,
                                "annotation_path": str(annotation_path),
                            }
                        )
            except Exception as error:
                failed = [
                    item for item, _, _, _ in pending if item.roi.roi_id not in completed
                ]
                self.failures += len(failed)
                message = f"{type(error).__name__}: {error}"
                lower = message.casefold()
                if (
                    "histoplus is not installed" in lower
                    or "local_files_only" in lower
                    or "histoplus_cellvit_segmentor" in lower
                ):
                    self._fatal_error = message
                self.audit.extend(
                    {
                        "roi_id": item.roi.roi_id,
                        "status": "batch_inference_failed",
                        "error": message,
                    }
                    for item in failed
                )
                if not self.config.histoplus_fail_open:
                    raise
        return output

    def analyze(
        self, slide_record: dict[str, Any], selected: SelectedROI
    ) -> CellularAnalysis | None:
        return self.analyze_batch(slide_record, [selected]).get(selected.roi.roi_id)

    def analyze_selected(
        self,
        slide: dict[str, Any],
        selected: list[SelectedROI],
        plan: EvidencePlan,
    ) -> dict[str, CellularAnalysis]:
        if not self.config.histoplus_enabled or not selected:
            self.skipped += len(selected)
            return {}
        policy = routing_policy(str(slide.get("cancer") or ""), plan)
        if policy.histoplus_policy == "off":
            self.skipped += len(selected)
            return {}
        limit = (
            self.config.quantification_high_power_top_k
            if is_quantitative_measurement_plan(plan)
            else self.config.histoplus_required_top_k
            if policy.histoplus_policy == "required"
            else self.config.histoplus_conditional_top_k
        )
        ranked = sorted(
            selected,
            key=lambda item: (
                item.roi.tissue_navigation.fusion_score if item.roi.tissue_navigation else 0.0,
                item.roi.retrieval_score,
            ),
            reverse=True,
        )[:limit]
        output = self.analyze_batch(slide, ranked)
        self.skipped += max(0, len(selected) - len(ranked))
        return output
