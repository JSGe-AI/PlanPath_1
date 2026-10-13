"""Compact WSI-level spatial store for precomputed HistoPLUS instances."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CELL_STORE_SCHEMA_VERSION = "1.0"


def cell_store_path(
    root: Path, slide: dict[str, Any], magnification: str
) -> Path:
    return (
        root
        / str(slide.get("cancer") or "unknown")
        / str(slide["patient_id"])
        / str(slide["slide_id"])
        / f"{magnification}.h5"
    )


@dataclass(frozen=True)
class StoredCellQuery:
    path: Path
    model_name: str
    model_signature: str
    input_mpp: float
    base_mpp: float
    instances: list[dict[str, Any]]


class PrecomputedCellStore:
    """Read only the spatial tiles intersecting an arbitrary level-0 ROI."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _metadata(self, handle: h5py.File) -> tuple[list[str], dict[str, Any]]:
        if str(handle.attrs.get("schema_version", "")) != CELL_STORE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cell-store schema: {self.path}")
        if int(handle.attrs.get("complete", 0)) != 1:
            raise ValueError(f"incomplete cell store: {self.path}")
        labels = [str(item) for item in json.loads(str(handle.attrs["class_labels_json"]))]
        metadata = {
            "model_name": str(handle.attrs["model_name"]),
            "model_signature": str(handle.attrs["model_signature"]),
            "input_mpp": float(handle.attrs["input_mpp"]),
            "base_mpp": float(handle.attrs["base_mpp"]),
        }
        return labels, metadata

    def query(
        self,
        coordinates: tuple[int, int, int, int],
        *,
        expected_model_signature: str | None = None,
        include_contours: bool = False,
    ) -> StoredCellQuery | None:
        x1, y1, x2, y2 = coordinates
        with h5py.File(self.path, "r", libver="latest", swmr=True) as handle:
            labels, metadata = self._metadata(handle)
            if (
                expected_model_signature is not None
                and metadata["model_signature"] != expected_model_signature
            ):
                return None
            boxes = np.asarray(handle["tile_boxes"], dtype=np.int64)
            keep = np.flatnonzero(
                (boxes[:, 0] < x2)
                & (boxes[:, 2] > x1)
                & (boxes[:, 1] < y2)
                & (boxes[:, 3] > y1)
            )
            if not len(keep):
                return None
            intersection_width = np.maximum(
                0, np.minimum(boxes[keep, 2], x2) - np.maximum(boxes[keep, 0], x1)
            )
            intersection_height = np.maximum(
                0, np.minimum(boxes[keep, 3], y2) - np.maximum(boxes[keep, 1], y1)
            )
            covered_area = int(np.sum(intersection_width * intersection_height))
            requested_area = max(1, (x2 - x1) * (y2 - y1))
            if covered_area < requested_area:
                return None

            tile_offsets = handle["tile_instance_offsets"]
            class_ids = handle["instance_class_id"]
            confidences = handle["instance_confidence"]
            centroids = handle["instance_centroid_level0"]
            areas = handle["instance_area_um2"]
            contour_offsets = handle["instance_contour_offsets"]
            contour_points = handle["contour_points_local"]
            output: list[dict[str, Any]] = []
            for tile_index in keep.tolist():
                start = int(tile_offsets[tile_index])
                stop = int(tile_offsets[tile_index + 1])
                if stop <= start:
                    continue
                tile_box = boxes[tile_index]
                tile_class_ids = np.asarray(class_ids[start:stop], dtype=np.int64)
                tile_confidences = np.asarray(confidences[start:stop], dtype=np.float32)
                tile_centroids = np.asarray(centroids[start:stop], dtype=np.float32)
                tile_areas = np.asarray(areas[start:stop], dtype=np.float32)
                inside = np.flatnonzero(
                    (tile_centroids[:, 0] >= x1)
                    & (tile_centroids[:, 0] < x2)
                    & (tile_centroids[:, 1] >= y1)
                    & (tile_centroids[:, 1] < y2)
                )
                for local_index in inside.tolist():
                    instance_index = start + local_index
                    global_x, global_y = tile_centroids[local_index]
                    item: dict[str, Any] = {
                        "cell_type": labels[int(tile_class_ids[local_index])],
                        "confidence": float(tile_confidences[local_index]),
                        "centroid_level0": [float(global_x), float(global_y)],
                        "area_um2": float(tile_areas[local_index]),
                    }
                    if include_contours:
                        point_start = int(contour_offsets[instance_index])
                        point_stop = int(contour_offsets[instance_index + 1])
                        local_points = np.asarray(
                            contour_points[point_start:point_stop], dtype=np.float32
                        )
                        local_points += tile_box[:2]
                        item["contour_level0"] = local_points.tolist()
                    output.append(item)
            return StoredCellQuery(
                path=self.path,
                model_name=metadata["model_name"],
                model_signature=metadata["model_signature"],
                input_mpp=metadata["input_mpp"],
                base_mpp=metadata["base_mpp"],
                instances=output,
            )


def spatial_core_boxes(
    parent_coordinates: np.ndarray,
    parent_size: int,
    core_size: int,
) -> np.ndarray:
    """Cover all candidate parent fields once with a fixed global grid."""

    occupied: set[tuple[int, int]] = set()
    for x_value, y_value in np.asarray(parent_coordinates, dtype=np.int64):
        x, y = int(x_value), int(y_value)
        gx_start = math.floor(x / core_size)
        gy_start = math.floor(y / core_size)
        gx_stop = math.ceil((x + parent_size) / core_size)
        gy_stop = math.ceil((y + parent_size) / core_size)
        occupied.update(
            (gx, gy)
            for gy in range(gy_start, gy_stop)
            for gx in range(gx_start, gx_stop)
            if gx >= 0 and gy >= 0
        )
    boxes = np.asarray(
        [
            (gx * core_size, gy * core_size, (gx + 1) * core_size, (gy + 1) * core_size)
            for gx, gy in sorted(occupied, key=lambda item: (item[1], item[0]))
        ],
        dtype=np.int32,
    )
    return boxes.reshape(-1, 4)
