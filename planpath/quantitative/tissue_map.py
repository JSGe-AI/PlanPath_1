"""CONCH v1.5 prototype tissue maps used strictly for spatial navigation.

The map is intentionally question independent.  It stores normalized soft
scores for routing, but those scores are never promoted to diagnostic evidence
by the workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


TISSUE_MAP_SCHEMA_VERSION = "1.0"
TISSUE_MAP_ALGORITHM_VERSION = "conch15-cluster-prototype-max-softmax-v1"
PATCH_NAME_RE = re.compile(r"^(?P<x>-?\d+)_(?P<y>-?\d+)(?:_(?P<size>\d+))?$")

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "gastric": [
        (0, 0, 0),
        (60, 180, 75),
        (230, 25, 75),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (128, 128, 128),
        (255, 225, 25),
        (128, 0, 0),
        (250, 190, 212),
        (66, 99, 235),
    ],
    "renal": [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (145, 30, 180),
        (70, 240, 240),
        (128, 128, 128),
        (128, 0, 0),
        (250, 190, 212),
    ],
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    try:
        image.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_patch_indices(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[int] = []
    sizes: list[int] = []
    for value in values:
        stem = Path(str(value)).stem
        match = PATCH_NAME_RE.fullmatch(stem)
        if not match:
            raise ValueError(f"invalid CONCH patch index: {value!r}")
        xs.append(int(match.group("x")))
        ys.append(int(match.group("y")))
        sizes.append(int(match.group("size") or 1024))
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(sizes, dtype=np.int32),
    )


@dataclass(frozen=True)
class TissuePrototypeModel:
    cancer: str
    labels: tuple[str, ...]
    prototypes: np.ndarray
    prototype_label_ids: np.ndarray
    prototype_path: Path
    signature: str
    temperature: float = 0.04
    unknown_confidence: float = 0.30
    unknown_margin: float = 0.01

    @classmethod
    def load(
        cls,
        cancer: str,
        prototype_root: Path,
        *,
        temperature: float = 0.04,
        unknown_confidence: float = 0.30,
        unknown_margin: float = 0.01,
    ) -> "TissuePrototypeModel":
        bank_cancer = "kidney" if cancer == "renal" else cancer
        directory = prototype_root / bank_cancer / "k100_prototype"
        bank_path = directory / f"{bank_cancer}_k100_prototype_bank.npz"
        labels_path = directory / f"{bank_cancer}_classifier_head_output_labels.json"
        labels = tuple(str(item) for item in json.loads(labels_path.read_text(encoding="utf-8")))
        with np.load(bank_path, allow_pickle=False) as payload:
            features = np.asarray(payload["normalized_features"], dtype=np.float32)
            label_ids = np.asarray(payload["label_ids"], dtype=np.int64)
            cluster_ids = np.asarray(payload["cluster_ids"], dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(f"invalid CONCH prototype shape: {features.shape}")
        if len(features) != len(label_ids) or len(features) != len(cluster_ids):
            raise ValueError("prototype features and metadata lengths differ")
        if min(label_ids) < 0 or max(label_ids) >= len(labels):
            raise ValueError("prototype label ID lies outside classifier labels")

        # Ten representative patches were retained per reviewed K-means cluster.
        # Collapse each (cluster, final label) group to one normalized prototype.
        # Including the label in the key preserves curated hemorrhage supplements
        # that intentionally share a source cluster with another final class.
        reduced: list[np.ndarray] = []
        reduced_labels: list[int] = []
        for cluster_id, label_id in sorted(set(zip(cluster_ids.tolist(), label_ids.tolist()))):
            mask = (cluster_ids == cluster_id) & (label_ids == label_id)
            center = features[mask].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            reduced.append(center.astype(np.float32))
            reduced_labels.append(int(label_id))
        prototypes = np.stack(reduced)
        source_hash = _sha256(bank_path)
        signature_payload = {
            "algorithm": TISSUE_MAP_ALGORITHM_VERSION,
            "cancer": cancer,
            "labels": labels,
            "prototype_sha256": source_hash,
            "temperature": temperature,
            "unknown_confidence": unknown_confidence,
            "unknown_margin": unknown_margin,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            cancer=cancer,
            labels=labels,
            prototypes=prototypes,
            prototype_label_ids=np.asarray(reduced_labels, dtype=np.int64),
            prototype_path=bank_path,
            signature=signature,
            temperature=temperature,
            unknown_confidence=unknown_confidence,
            unknown_margin=unknown_margin,
        )

    def predict(self, features: np.ndarray, *, device: str = "cpu") -> dict[str, np.ndarray]:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.prototypes.shape[1]:
            raise ValueError(
                f"CONCH feature/prototype mismatch: {values.shape} vs {self.prototypes.shape}"
            )
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        valid = np.isfinite(values).all(axis=1) & (norms[:, 0] > 1e-12)
        values = values / np.maximum(norms, 1e-12)

        if device != "cpu":
            try:
                import torch

                tensor = torch.from_numpy(values).to(device)
                prototype_tensor = torch.from_numpy(self.prototypes).to(device)
                similarities = (tensor @ prototype_tensor.T).float().cpu().numpy()
            except Exception as error:
                raise RuntimeError(f"GPU prototype inference failed on {device}") from error
        else:
            similarities = values @ self.prototypes.T

        class_scores = np.full((len(values), len(self.labels)), -1.0, dtype=np.float32)
        for label_id in range(len(self.labels)):
            members = np.flatnonzero(self.prototype_label_ids == label_id)
            if len(members):
                class_scores[:, label_id] = similarities[:, members].max(axis=1)
        logits = (class_scores - class_scores.max(axis=1, keepdims=True)) / self.temperature
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        order = np.argsort(class_scores, axis=1)
        predicted = order[:, -1].astype(np.uint8)
        confidence = probabilities[np.arange(len(values)), predicted]
        margin = class_scores[np.arange(len(values)), order[:, -1]] - class_scores[
            np.arange(len(values)), order[:, -2]
        ]
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        entropy /= math.log(max(2, len(self.labels)))
        unknown = (
            (~valid)
            | (confidence < self.unknown_confidence)
            | (margin < self.unknown_margin)
        )
        predicted[~valid] = 0
        probabilities[~valid] = 0
        class_scores[~valid] = -1
        confidence[~valid] = 0
        margin[~valid] = 0
        entropy[~valid] = 1
        return {
            "class_scores": class_scores.astype(np.float16),
            "class_probs": probabilities.astype(np.float16),
            "predicted_class": predicted,
            "confidence": confidence.astype(np.float16),
            "margin": margin.astype(np.float16),
            "entropy": entropy.astype(np.float16),
            "unknown": unknown.astype(np.uint8),
        }

    def build_map(
        self,
        conch_path: Path,
        output_path: Path,
        preview_path: Path,
        *,
        slide_id: str,
        device: str = "cpu",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if output_path.is_file() and preview_path.is_file() and not overwrite:
            with np.load(output_path, allow_pickle=False) as old:
                old_signature = str(old["model_signature"].item())
                old_size = int(old["conch_feature_size"].item())
                old_mtime = int(old["conch_feature_mtime_ns"].item())
            current_stat = conch_path.stat()
            if (
                old_signature == self.signature
                and old_size == current_stat.st_size
                and old_mtime == current_stat.st_mtime_ns
            ):
                return {"status": "cached", "patch_count": -1}

        raw = np.load(conch_path, allow_pickle=True)
        if not isinstance(raw, np.ndarray) or raw.shape != ():
            raise ValueError(f"unexpected CONCH file format: {conch_path}")
        payload = raw.item()
        features = np.asarray(payload["feature"], dtype=np.float32)
        indices = list(payload["index"])
        if len(features) != len(indices):
            raise ValueError(f"feature/index length mismatch: {conch_path}")
        x, y, patch_sizes = parse_patch_indices(indices)
        predictions = self.predict(features, device=device)
        label_json = json.dumps(self.labels, ensure_ascii=False, separators=(",", ":"))
        source_stat = conch_path.stat()
        _atomic_savez(
            output_path,
            schema_version=np.asarray(TISSUE_MAP_SCHEMA_VERSION),
            algorithm_version=np.asarray(TISSUE_MAP_ALGORITHM_VERSION),
            model_signature=np.asarray(self.signature),
            slide_id=np.asarray(slide_id),
            cancer=np.asarray(self.cancer),
            conch_feature_path=np.asarray(str(conch_path)),
            conch_feature_size=np.asarray(source_stat.st_size, dtype=np.int64),
            conch_feature_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            prototype_path=np.asarray(str(self.prototype_path)),
            labels_json=np.asarray(label_json),
            x=x,
            y=y,
            patch_size=patch_sizes,
            **predictions,
        )
        self.write_preview(output_path, preview_path)
        return {
            "status": "written",
            "patch_count": len(features),
            "unknown_count": int(predictions["unknown"].sum()),
        }

    def write_preview(self, map_path: Path, preview_path: Path, scale: int = 4) -> None:
        with np.load(map_path, allow_pickle=False) as payload:
            x = np.asarray(payload["x"], dtype=np.int64)
            y = np.asarray(payload["y"], dtype=np.int64)
            sizes = np.asarray(payload["patch_size"], dtype=np.int64)
            labels = np.asarray(payload["predicted_class"], dtype=np.int64)
            unknown = np.asarray(payload["unknown"], dtype=bool)
        if not len(x):
            raise ValueError(f"cannot preview an empty tissue map: {map_path}")
        step = max(1, int(np.median(sizes)))
        x0, y0 = int(x.min()), int(y.min())
        gx = np.rint((x - x0) / step).astype(int)
        gy = np.rint((y - y0) / step).astype(int)
        width, height = int(gx.max()) + 1, int(gy.max()) + 1
        if width * height > 100_000_000:
            raise ValueError(f"unreasonable tissue-map canvas {width}x{height}: {map_path}")
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        palette = PALETTES[self.cancer]
        for index in range(len(labels)):
            color = (96, 96, 96) if unknown[index] else palette[int(labels[index])]
            canvas[gy[index], gx[index]] = color
        image = Image.fromarray(canvas, mode="RGB")
        if scale > 1:
            image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        _atomic_image(image, preview_path)


class TissueMap:
    """In-memory spatial query facade over one precomputed soft tissue map."""

    def __init__(self, path: Path, *, build_exterior_topology: bool = True) -> None:
        with np.load(path, allow_pickle=False) as payload:
            self.labels = tuple(json.loads(str(payload["labels_json"].item())))
            self.x = np.asarray(payload["x"], dtype=np.int64)
            self.y = np.asarray(payload["y"], dtype=np.int64)
            self.patch_size = np.asarray(payload["patch_size"], dtype=np.int64)
            self.class_probs = np.asarray(payload["class_probs"], dtype=np.float32)
            self.confidence = np.asarray(payload["confidence"], dtype=np.float32)
            self.entropy = np.asarray(payload["entropy"], dtype=np.float32)
            self.unknown = np.asarray(payload["unknown"], dtype=bool)
            self.model_signature = str(payload["model_signature"].item())
        self.path = path
        if self.class_probs.shape != (len(self.x), len(self.labels)):
            raise ValueError(f"invalid tissue map probability shape: {path}")
        self._label_to_id = {label: index for index, label in enumerate(self.labels)}
        if build_exterior_topology:
            self._build_exterior_topology()

    def _build_exterior_topology(self) -> None:
        """Locate map background that is topologically connected to the exterior.

        ``artifact_or_background`` alone also includes internal tears and
        processing clefts.  Rasterizing the sparse patch cloud and flood
        filling only background connected to the padded map boundary lets
        capsule routing distinguish those holes from a genuine tissue
        surface.  This map remains a navigation prior, never answer evidence.
        """

        count = len(self.x)
        self._exterior_distance = np.full(count, np.iinfo(np.int16).max, dtype=np.int16)
        self._nearest_exterior_cell = np.full((count, 2), -1, dtype=np.int32)
        if not count:
            self._exterior_resolution = 512
            self._exterior_origin = (0, 0)
            return
        median_size = max(1, int(np.median(self.patch_size)))
        resolution = max(128, median_size // 2)
        origin_x, origin_y = int(self.x.min()), int(self.y.min())
        maximum_x = int(np.max(self.x + self.patch_size))
        maximum_y = int(np.max(self.y + self.patch_size))
        width = int(math.ceil((maximum_x - origin_x) / resolution)) + 2
        height = int(math.ceil((maximum_y - origin_y) / resolution)) + 2
        self._exterior_resolution = resolution
        self._exterior_origin = (origin_x, origin_y)
        # Guard malformed maps before allocating an unreasonable raster.
        if width <= 2 or height <= 2 or width * height > 25_000_000:
            return
        coverage = np.zeros((height, width), dtype=np.uint16)
        tissue_coverage = np.zeros((height, width), dtype=np.uint16)
        background_id = self.label_id("artifact_or_background")
        hard_background = (
            np.argmax(self.class_probs, axis=1) == background_id
            if background_id is not None
            else np.zeros(count, dtype=bool)
        )
        footprints: list[tuple[int, int, int, int]] = []
        for index, (x, y, size) in enumerate(
            zip(self.x.tolist(), self.y.tolist(), self.patch_size.tolist())
        ):
            gx1 = 1 + max(0, int(math.floor((int(x) - origin_x) / resolution)))
            gy1 = 1 + max(0, int(math.floor((int(y) - origin_y) / resolution)))
            gx2 = 1 + min(
                width - 2,
                int(math.ceil((int(x) + int(size) - origin_x) / resolution)),
            )
            gy2 = 1 + min(
                height - 2,
                int(math.ceil((int(y) + int(size) - origin_y) / resolution)),
            )
            gx2, gy2 = max(gx1 + 1, gx2), max(gy1 + 1, gy2)
            footprints.append((gx1, gy1, gx2, gy2))
            coverage[gy1:gy2, gx1:gx2] += 1
            if not hard_background[index]:
                tissue_coverage[gy1:gy2, gx1:gx2] += 1

        passable = (coverage == 0) | (tissue_coverage == 0)
        exterior = np.zeros_like(passable)
        queue: deque[tuple[int, int]] = deque()
        for gx in range(width):
            for gy in (0, height - 1):
                if passable[gy, gx] and not exterior[gy, gx]:
                    exterior[gy, gx] = True
                    queue.append((gy, gx))
        for gy in range(height):
            for gx in (0, width - 1):
                if passable[gy, gx] and not exterior[gy, gx]:
                    exterior[gy, gx] = True
                    queue.append((gy, gx))
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
        while queue:
            gy, gx = queue.popleft()
            for dy, dx in neighbors:
                ny, nx = gy + dy, gx + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and passable[ny, nx]
                    and not exterior[ny, nx]
                ):
                    exterior[ny, nx] = True
                    queue.append((ny, nx))

        distance = np.full((height, width), np.iinfo(np.int16).max, dtype=np.int16)
        nearest_y = np.full((height, width), -1, dtype=np.int32)
        nearest_x = np.full((height, width), -1, dtype=np.int32)
        queue.clear()
        for gy, gx in np.argwhere(exterior):
            distance[gy, gx] = 0
            nearest_y[gy, gx] = int(gy)
            nearest_x[gy, gx] = int(gx)
            queue.append((int(gy), int(gx)))
        while queue:
            gy, gx = queue.popleft()
            next_distance = int(distance[gy, gx]) + 1
            for dy, dx in neighbors:
                ny, nx = gy + dy, gx + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and next_distance < int(distance[ny, nx])
                ):
                    distance[ny, nx] = next_distance
                    nearest_y[ny, nx] = nearest_y[gy, gx]
                    nearest_x[ny, nx] = nearest_x[gy, gx]
                    queue.append((ny, nx))

        for index, (gx1, gy1, gx2, gy2) in enumerate(footprints):
            local = distance[gy1:gy2, gx1:gx2]
            flat = int(np.argmin(local))
            ly, lx = np.unravel_index(flat, local.shape)
            gy, gx = gy1 + int(ly), gx1 + int(lx)
            self._exterior_distance[index] = distance[gy, gx]
            self._nearest_exterior_cell[index] = (
                nearest_x[gy, gx],
                nearest_y[gy, gx],
            )

    def label_id(self, name: str) -> int | None:
        return self._label_to_id.get(name)

    def global_summary(self) -> dict[str, Any]:
        """Return area-weighted whole-map composition for quantification.

        Unlike ``window_summary``, this method is not a navigation score.  It
        supplies the sampling frame and denominator for a later quantitative
        estimator. Unknown patches are excluded from class areas but retained
        in the reported coverage so the Python evidence gate can reject an
        incompletely measured slide.
        """

        if not len(self.x):
            return {
                "available": False,
                "patch_count": 0,
                "known_area_px2": 0.0,
                "total_area_px2": 0.0,
                "known_fraction": 0.0,
                "class_areas_px2": {},
                "class_fractions": {},
            }
        areas = np.square(self.patch_size.astype(np.float64))
        known_weights = areas * (~self.unknown).astype(np.float64)
        total_area = float(areas.sum())
        known_area = float(known_weights.sum())
        class_areas = (self.class_probs.astype(np.float64) * known_weights[:, None]).sum(
            axis=0
        )
        fractions = class_areas / max(known_area, 1e-12)
        return {
            "available": known_area > 0,
            "patch_count": int(len(self.x)),
            "known_area_px2": known_area,
            "total_area_px2": total_area,
            "known_fraction": known_area / max(total_area, 1e-12),
            "class_areas_px2": {
                label: float(class_areas[index])
                for index, label in enumerate(self.labels)
            },
            "class_fractions": {
                label: float(fractions[index])
                for index, label in enumerate(self.labels)
            },
        }

    def connected_target_components(
        self,
        target_labels: Iterable[str],
        *,
        probability_threshold: float = 0.10,
        min_patch_count: int = 5,
    ) -> list[dict[str, Any]]:
        """Return separated soft-map candidates for conservative focus counting.

        Components are navigation hypotheses only. They become countable foci
        only after a sampled region also passes HistoPLUS and visible-
        morphology checks in the quantitative evidence gate.
        """

        target_ids = [self.label_id(name) for name in target_labels]
        target_ids = [index for index in target_ids if index is not None]
        if not target_ids or not len(self.x):
            return []
        scores = self.class_probs[:, target_ids].sum(axis=1)
        # Do not discard uncertain boundary patches here: doing so fragments a
        # real low-confidence tumor region into isolated points. These remain
        # navigation candidates and still require independent HistoPLUS and
        # morphology confirmation before entering a count.
        active = np.flatnonzero(scores >= probability_threshold)
        if not len(active):
            return []
        # Map records are patch centers from a tissue-mask sampling cloud and
        # are not guaranteed to lie on a strict stride lattice. Quantize by
        # the physical patch footprint so touching/overlapping target patches
        # form one candidate while separated tissue fragments remain distinct.
        step = max(1, int(np.median(self.patch_size)))
        origin_x, origin_y = int(self.x.min()), int(self.y.min())
        grid_to_index = {
            (
                int(round((int(self.x[index]) - origin_x) / step)),
                int(round((int(self.y[index]) - origin_y) / step)),
            ): int(index)
            for index in active
        }
        remaining = set(grid_to_index)
        components: list[list[int]] = []
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
                components.append(members)

        output: list[dict[str, Any]] = []
        for members in components:
            indices = np.asarray(members, dtype=np.int64)
            weights = scores[indices].astype(np.float64)
            centers_x = self.x[indices] + self.patch_size[indices] / 2.0
            centers_y = self.y[indices] + self.patch_size[indices] / 2.0
            x1 = int(self.x[indices].min())
            y1 = int(self.y[indices].min())
            x2 = int((self.x[indices] + self.patch_size[indices]).max())
            y2 = int((self.y[indices] + self.patch_size[indices]).max())
            output.append(
                {
                    "center": (
                        int(round(float(np.average(centers_x, weights=weights)))),
                        int(round(float(np.average(centers_y, weights=weights)))),
                    ),
                    "box": (x1, y1, x2, y2),
                    "patch_count": int(len(indices)),
                    "mean_target_probability": float(scores[indices].mean()),
                    "score": float(scores[indices].sum()),
                }
            )
        output.sort(key=lambda item: item["score"], reverse=True)
        return output

    def exterior_interface_score(
        self,
        box: tuple[int, int, int, int],
        target_labels: Iterable[str],
    ) -> float:
        """Score target tissue lying near exterior-connected background."""

        indices = self._overlap_indices(box)
        target_ids = [self.label_id(item) for item in target_labels]
        target_ids = [item for item in target_ids if item is not None]
        if not len(indices) or not target_ids:
            return 0.0
        target = self.class_probs[np.ix_(indices, target_ids)].sum(axis=1)
        # Eight half-patch raster steps span one 4096-pixel low-power field.
        # Interior holes receive no special score because the flood fill did
        # not mark them as exterior; deep interior tissue decays to zero.
        distance = self._exterior_distance[indices].astype(np.float32)
        proximity = np.clip((9.0 - distance) / 8.0, 0.0, 1.0)
        return float(np.max(target * proximity, initial=0.0))

    def exterior_interface_center(
        self,
        box: tuple[int, int, int, int],
        target_labels: Iterable[str],
    ) -> tuple[int, int] | None:
        """Center between the strongest peripheral target patch and exterior."""

        indices = self._overlap_indices(box)
        target_ids = [self.label_id(item) for item in target_labels]
        target_ids = [item for item in target_ids if item is not None]
        if not len(indices) or not target_ids:
            return None
        target = self.class_probs[np.ix_(indices, target_ids)].sum(axis=1)
        distance = self._exterior_distance[indices].astype(np.float32)
        proximity = np.clip((9.0 - distance) / 8.0, 0.0, 1.0)
        scores = target * proximity
        local_index = int(np.argmax(scores))
        if float(scores[local_index]) <= 0:
            return None
        index = int(indices[local_index])
        target_x = float(self.x[index]) + float(self.patch_size[index]) / 2.0
        target_y = float(self.y[index]) + float(self.patch_size[index]) / 2.0
        exterior_gx, exterior_gy = self._nearest_exterior_cell[index]
        if exterior_gx < 0 or exterior_gy < 0:
            return int(round(target_x)), int(round(target_y))
        origin_x, origin_y = self._exterior_origin
        exterior_x = origin_x + (float(exterior_gx) - 0.5) * self._exterior_resolution
        exterior_y = origin_y + (float(exterior_gy) - 0.5) * self._exterior_resolution
        return (
            int(round((target_x + exterior_x) / 2.0)),
            int(round((target_y + exterior_y) / 2.0)),
        )

    def _overlap_indices(self, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        overlap = (
            (self.x < x2)
            & (self.y < y2)
            & (self.x + self.patch_size > x1)
            & (self.y + self.patch_size > y1)
        )
        return np.flatnonzero(overlap)

    def window_summary(
        self,
        box: tuple[int, int, int, int],
        target_labels: Iterable[str] = (),
        interface_pairs: Iterable[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = box
        indices = self._overlap_indices(box)
        if not len(indices):
            return {
                "available": False,
                "target_score": 0.0,
                "interface_score": 0.0,
                "heterogeneity": 0.0,
                "map_confidence": 0.0,
                "class_fractions": {},
            }
        probs = self.class_probs[indices]
        mean_probs = probs.mean(axis=0)
        fractions = {
            self.labels[index]: round(float(value), 6)
            for index, value in enumerate(mean_probs)
            if value >= 0.01
        }
        target_ids = [self.label_id(item) for item in target_labels]
        target_ids = [item for item in target_ids if item is not None]
        target_score = float(mean_probs[target_ids].sum()) if target_ids else 0.0
        interface_score = 0.0
        for left, right in interface_pairs:
            left_id, right_id = self.label_id(left), self.label_id(right)
            if left_id is None or right_id is None:
                continue
            # A soft co-occurrence proxy is more robust than hard-label adjacency
            # at the 1024-pixel grid resolution.
            within_patch = float(
                np.sqrt(np.maximum(probs[:, left_id] * probs[:, right_id], 0)).mean()
            )
            within_window = float(
                math.sqrt(max(float(mean_probs[left_id] * mean_probs[right_id]), 0.0))
            )
            pair_score = max(within_patch, within_window)
            interface_score = max(interface_score, pair_score)
        return {
            "available": True,
            "target_score": target_score,
            "interface_score": interface_score,
            "heterogeneity": float(self.entropy[indices].mean()),
            "map_confidence": float(self.confidence[indices].mean()),
            "unknown_fraction": float(self.unknown[indices].mean()),
            "class_fractions": fractions,
        }

    def interface_center(
        self,
        box: tuple[int, int, int, int],
        interface_pairs: Iterable[tuple[str, str]],
    ) -> tuple[int, int] | None:
        """Localize a soft tissue interface inside a navigation window.

        This coordinate is used only to center a subsequently reviewed image
        field; it is never diagnostic evidence.  Pairwise proximity prevents
        a 4096-pixel outer-surface ROI from placing a thin capsule along the
        very edge where every higher-power child sees only tissue or glass.
        """

        x1, y1, x2, y2 = box
        overlap = (
            (self.x < x2)
            & (self.y < y2)
            & (self.x + self.patch_size > x1)
            & (self.y + self.patch_size > y1)
        )
        indices = np.flatnonzero(overlap)
        if not len(indices):
            return None
        centers_x = self.x[indices].astype(np.float64) + self.patch_size[indices] / 2.0
        centers_y = self.y[indices].astype(np.float64) + self.patch_size[indices] / 2.0
        best: tuple[float, int, int] | None = None
        scale = max(float(np.median(self.patch_size[indices])), 1.0)
        for left, right in interface_pairs:
            left_id, right_id = self.label_id(left), self.label_id(right)
            if left_id is None or right_id is None:
                continue
            left_prob = self.class_probs[indices, left_id].astype(np.float64)
            right_prob = self.class_probs[indices, right_id].astype(np.float64)
            distance = np.hypot(
                centers_x[:, None] - centers_x[None, :],
                centers_y[:, None] - centers_y[None, :],
            )
            proximity = 1.0 / (1.0 + distance / scale)
            scores = np.sqrt(np.maximum(left_prob[:, None] * right_prob[None, :], 0.0))
            scores *= proximity
            flat = int(np.argmax(scores))
            score = float(scores.flat[flat])
            left_index, right_index = np.unravel_index(flat, scores.shape)
            if best is None or score > best[0]:
                best = (score, int(left_index), int(right_index))
        if best is None or best[0] <= 0:
            return None
        _, left_index, right_index = best
        return (
            int(round((centers_x[left_index] + centers_x[right_index]) / 2.0)),
            int(round((centers_y[left_index] + centers_y[right_index]) / 2.0)),
        )
