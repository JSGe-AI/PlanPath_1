#!/usr/bin/env python3
"""Precompute compact spatial HistoPLUS stores at both PlanPath ROI scales."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.cell_store import (  # noqa: E402
    CELL_STORE_SCHEMA_VERSION,
    cell_store_path,
    spatial_core_boxes,
)
from planpath.core.config import AgentConfig  # noqa: E402
from planpath.core.histoplus import (  # noqa: E402
    HISTOPLUS_ADAPTER_VERSION,
    HistoPLUSAnalyzer,
    _polygon_area,
)
from planpath.core.io_utils import write_json_atomic  # noqa: E402
from planpath.core.plip import SelectedROI  # noqa: E402
from planpath.core.schemas import ROI  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "cell_seg"
MAGNIFICATION_PARAMETERS = {
    "20x": {"core_size": 1024, "mpp": 0.5},
    "40x": {"core_size": 512, "mpp": 0.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models/histoplus")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--magnification", choices=("20x", "40x"), action="append")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument(
        "--max-tiles-per-slide",
        type=int,
        help="Smoke-test only: write a partial spatial store that falls back outside covered tiles.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda item: (str(item["cancer"]), str(item["patient_id"]), int(item["slide_index"])))
    return rows


def parent_regions(slide: dict[str, Any]) -> tuple[np.ndarray, int, str]:
    plip_path = Path(str(slide.get("plip_feature_cache") or ""))
    if plip_path.is_file():
        with np.load(plip_path, allow_pickle=False) as payload:
            return np.asarray(payload["coords"], dtype=np.int64), 4096, "plip_5x"
    coordinate_path = Path(str(slide.get("coordinate_path") or ""))
    if not coordinate_path.is_file():
        raise FileNotFoundError(
            f"neither PLIP cache nor coordinate HDF5 exists for {slide['slide_id']}"
        )
    with h5py.File(coordinate_path, "r") as handle:
        dataset = handle["coords"]
        coords = np.asarray(dataset, dtype=np.int64)
        patch_size = int(dataset.attrs.get("patch_size", 4096))
    return coords, patch_size, "coordinate_h5"


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if not len(values):
        return
    start = int(dataset.shape[0])
    dataset.resize((start + len(values), *dataset.shape[1:]))
    dataset[start:] = values


def valid_existing(
    path: Path,
    *,
    model_signature: str,
    slide: dict[str, Any],
    magnification: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        source = Path(str(slide["raw_slide_path"]))
        stat = source.stat()
        with h5py.File(path, "r") as handle:
            return bool(
                int(handle.attrs.get("complete", 0)) == 1
                and str(handle.attrs.get("schema_version", "")) == CELL_STORE_SCHEMA_VERSION
                and str(handle.attrs.get("adapter_version", "")) == HISTOPLUS_ADAPTER_VERSION
                and str(handle.attrs.get("model_signature", "")) == model_signature
                and str(handle.attrs.get("slide_id", "")) == str(slide["slide_id"])
                and str(handle.attrs.get("magnification", "")) == magnification
                and int(handle.attrs.get("coverage_limited", 0)) == 0
                and int(handle.attrs.get("wsi_size", -1)) == stat.st_size
                and int(handle.attrs.get("wsi_mtime_ns", -1)) == stat.st_mtime_ns
            )
    except (OSError, KeyError, ValueError):
        return False


def create_datasets(handle: h5py.File, boxes: np.ndarray) -> dict[str, h5py.Dataset]:
    handle.create_dataset("tile_boxes", data=boxes, compression="lzf", shuffle=True)
    datasets = {
        "tile_offsets": handle.create_dataset(
            "tile_instance_offsets", shape=(len(boxes) + 1,), dtype="i8"
        ),
        "class_ids": handle.create_dataset(
            "instance_class_id", shape=(0,), maxshape=(None,), chunks=(16384,), dtype="u1", compression="lzf"
        ),
        "confidences": handle.create_dataset(
            "instance_confidence", shape=(0,), maxshape=(None,), chunks=(16384,), dtype="f2", compression="lzf"
        ),
        "centroids": handle.create_dataset(
            "instance_centroid_level0", shape=(0, 2), maxshape=(None, 2), chunks=(8192, 2), dtype="f4", compression="lzf", shuffle=True
        ),
        "areas": handle.create_dataset(
            "instance_area_um2", shape=(0,), maxshape=(None,), chunks=(16384,), dtype="f4", compression="lzf", shuffle=True
        ),
        "contour_offsets": handle.create_dataset(
            "instance_contour_offsets", shape=(1,), maxshape=(None,), chunks=(16384,), dtype="i8", compression="lzf", shuffle=True
        ),
        "contour_points": handle.create_dataset(
            "contour_points_local", shape=(0, 2), maxshape=(None, 2), chunks=(32768, 2), dtype="i2", compression="lzf", shuffle=True
        ),
    }
    datasets["tile_offsets"][0] = 0
    datasets["contour_offsets"][0] = 0
    return datasets


def process_scale(
    analyzer: HistoPLUSAnalyzer,
    slide: dict[str, Any],
    magnification: str,
    output_path: Path,
    *,
    batch_size: int,
    overwrite: bool,
    max_tiles: int | None = None,
) -> dict[str, Any]:
    import openslide
    import torch

    parameters = MAGNIFICATION_PARAMETERS[magnification]
    core_size = int(parameters["core_size"])
    mpp = float(parameters["mpp"])
    model_signature = analyzer._model_signature(mpp)
    if not overwrite and valid_existing(
        output_path,
        model_signature=model_signature,
        slide=slide,
        magnification=magnification,
    ):
        with h5py.File(output_path, "r") as handle:
            return {
                "status": "cached",
                "tile_count": int(len(handle["tile_boxes"])),
                "instance_count": int(len(handle["instance_class_id"])),
            }

    parents, parent_size, coverage_source = parent_regions(slide)
    boxes = spatial_core_boxes(parents, parent_size, core_size)
    coverage_limited = max_tiles is not None and len(boxes) > max_tiles
    if max_tiles is not None:
        boxes = boxes[:max_tiles]
    if not len(boxes):
        raise ValueError(f"no candidate tissue coordinates for {slide['slide_id']}")
    segmentor = analyzer._segmentor(mpp)
    labels = [str(segmentor.class_mapping[index]) for index in sorted(segmentor.class_mapping)]
    class_ids = {label: index for index, label in enumerate(labels)}
    source = Path(str(slide["raw_slide_path"]))
    source_stat = source.stat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{magnification}.", suffix=".h5.tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    started = time.time()
    total_instances = 0
    total_points = 0
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs.update(
                {
                    "schema_version": CELL_STORE_SCHEMA_VERSION,
                    "adapter_version": HISTOPLUS_ADAPTER_VERSION,
                    "complete": 0,
                    "cancer": str(slide["cancer"]),
                    "patient_id": str(slide["patient_id"]),
                    "slide_id": str(slide["slide_id"]),
                    "magnification": magnification,
                    "core_size_level0": core_size,
                    "input_mpp": mpp,
                    "model_name": str(segmentor.segmentor_name),
                    "model_signature": model_signature,
                    "class_labels_json": json.dumps(labels, ensure_ascii=False),
                    "coverage_source": coverage_source,
                    "coverage_parent_size": parent_size,
                    "coverage_limited": int(coverage_limited),
                    "wsi_path": str(source),
                    "wsi_size": source_stat.st_size,
                    "wsi_mtime_ns": source_stat.st_mtime_ns,
                }
            )
            datasets = create_datasets(handle, boxes)
            with openslide.OpenSlide(str(source)) as wsi:
                base_mpp: float | None = None
                for batch_start in range(0, len(boxes), batch_size):
                    batch_boxes = boxes[batch_start : batch_start + batch_size]
                    selected: list[SelectedROI] = []
                    images = []
                    origins: list[tuple[int, int]] = []
                    for local_index, raw_box in enumerate(batch_boxes):
                        box = tuple(int(value) for value in raw_box)
                        roi = ROI(
                            roi_id=f"precompute_{batch_start + local_index}",
                            slide_id=str(slide["slide_id"]),
                            coordinates=box,
                            magnification=magnification,
                            retrieval_score=0.0,
                            matched_query="question-independent precomputation",
                        )
                        item = SelectedROI(roi=roi, image=None)  # type: ignore[arg-type]
                        image, origin, current_base_mpp = analyzer._read_context(wsi, item, mpp)
                        selected.append(item)
                        images.append(image)
                        origins.append(origin)
                        if base_mpp is None:
                            base_mpp = current_base_mpp
                    tensor = torch.stack([segmentor.transform(image) for image in images])
                    with torch.inference_mode():
                        outputs = segmentor.forward(tensor)
                    arrays = {
                        name: value.detach().cpu().numpy()
                        for name, value in outputs.items()
                    }
                    predictions = segmentor.get_postprocess_fn()(arrays)
                    assert base_mpp is not None
                    scale_level0 = mpp / base_mpp
                    for batch_index, (item, origin, prediction) in enumerate(
                        zip(selected, origins, predictions, strict=True)
                    ):
                        tile_index = batch_start + batch_index
                        x1, y1, x2, y2 = item.roi.coordinates
                        tile_classes: list[int] = []
                        tile_confidences: list[float] = []
                        tile_centroids: list[tuple[float, float]] = []
                        tile_areas: list[float] = []
                        tile_contours: list[np.ndarray] = []
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
                            local_points = np.rint(global_points - np.asarray([x1, y1])).astype(np.int16)
                            tile_classes.append(class_ids[str(cell_type)])
                            tile_confidences.append(float(confidence))
                            tile_centroids.append((global_x, global_y))
                            tile_areas.append(_polygon_area(points) * mpp * mpp)
                            tile_contours.append(local_points)
                        count = len(tile_classes)
                        if count:
                            _append(datasets["class_ids"], np.asarray(tile_classes, dtype=np.uint8))
                            _append(datasets["confidences"], np.asarray(tile_confidences, dtype=np.float16))
                            _append(datasets["centroids"], np.asarray(tile_centroids, dtype=np.float32))
                            _append(datasets["areas"], np.asarray(tile_areas, dtype=np.float32))
                            for contour in tile_contours:
                                _append(datasets["contour_points"], contour)
                                total_points += len(contour)
                                size = int(datasets["contour_offsets"].shape[0])
                                datasets["contour_offsets"].resize((size + 1,))
                                datasets["contour_offsets"][size] = total_points
                        total_instances += count
                        datasets["tile_offsets"][tile_index + 1] = total_instances
                    if batch_start and batch_start % (batch_size * 50) == 0:
                        handle.flush()
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"  {slide['slide_id']} {magnification}: "
                            f"{min(batch_start + batch_size, len(boxes))}/{len(boxes)} tiles "
                            f"({min(batch_start + batch_size, len(boxes)) / elapsed:.2f} tile/s)",
                            flush=True,
                        )
                handle.attrs["base_mpp"] = float(base_mpp or 0.25)
            handle.attrs["instance_count"] = total_instances
            handle.attrs["contour_point_count"] = total_points
            handle.attrs["complete"] = 1
            handle.flush()
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "written",
        "tile_count": len(boxes),
        "instance_count": total_instances,
        "elapsed_seconds": time.time() - started,
        "coverage_source": coverage_source,
        "coverage_limited": coverage_limited,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    rows = load_rows(args.manifest)[args.shard_index :: args.shard_count]
    if args.max_slides is not None:
        rows = rows[: args.max_slides]
    magnifications = args.magnification or ["20x", "40x"]
    config = AgentConfig(
        device=args.device,
        histoplus_model_root=args.model_root,
        histoplus_fail_open=False,
    )
    analyzer = HistoPLUSAnalyzer(config)
    results: list[dict[str, Any]] = []
    run_started = time.time()
    for slide_index, slide in enumerate(rows, 1):
        for magnification in magnifications:
            output = cell_store_path(args.output_root, slide, magnification)
            try:
                summary = process_scale(
                    analyzer,
                    slide,
                    magnification,
                    output,
                    batch_size=args.batch_size,
                    overwrite=args.overwrite,
                    max_tiles=args.max_tiles_per_slide,
                )
                result = {
                    "slide_id": slide["slide_id"],
                    "cancer": slide["cancer"],
                    "magnification": magnification,
                    "path": str(output),
                    **summary,
                }
            except Exception as error:
                result = {
                    "slide_id": slide["slide_id"],
                    "cancer": slide["cancer"],
                    "magnification": magnification,
                    "path": str(output),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                if args.fail_fast:
                    raise
            results.append(result)
            print(
                f"[{slide_index}/{len(rows)}] {slide['slide_id']} {magnification}: "
                f"{result['status']} tiles={result.get('tile_count', 0)} "
                f"instances={result.get('instance_count', 0)}"
                + (f" error={result['error']}" if result["status"] == "error" else ""),
                flush=True,
            )
    audit = {
        "schema_version": CELL_STORE_SCHEMA_VERSION,
        "adapter_version": HISTOPLUS_ADAPTER_VERSION,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "slide_count": len(rows),
        "magnifications": magnifications,
        "written": sum(item["status"] == "written" for item in results),
        "cached": sum(item["status"] == "cached" for item in results),
        "errors": [item for item in results if item["status"] == "error"],
        "tile_count": sum(int(item.get("tile_count", 0)) for item in results),
        "instance_count": sum(int(item.get("instance_count", 0)) for item in results),
        "elapsed_seconds": time.time() - run_started,
        "results": results,
    }
    audit_path = args.audit_output or (
        args.output_root
        / "_audits"
        / f"precompute_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json"
    )
    write_json_atomic(audit_path, audit)
    print(f"audit: {audit_path}", flush=True)
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
