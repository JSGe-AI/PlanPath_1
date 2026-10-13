#!/usr/bin/env python3
"""Generate only the 5x PLIP caches that a PlanPath manifest is missing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.image_io import read_field  # noqa: E402
from planpath.core.io_utils import read_jsonl  # noqa: E402
from planpath.core.plip import LocalPLIPEncoder  # noqa: E402


def tissue_fraction(image) -> float:
    pixels = np.asarray(image.resize((64, 64)), dtype=np.float32)
    brightness = pixels.mean(axis=2)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    return float(((brightness < 235) & ((chroma > 8) | (brightness < 205))).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--plip-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "plip",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_tissue_fraction <= 1:
        parser.error("--min-tissue-fraction must be in [0, 1]")
    try:
        import openslide
    except ImportError as error:
        raise RuntimeError("cache preparation requires openslide-python") from error

    records = read_jsonl(args.manifest)
    pending = [
        record
        for record in records
        if args.overwrite or not Path(str(record["plip_feature_cache"])).is_file()
    ]
    if not pending:
        print("all PLIP caches already exist")
        return
    encoder = LocalPLIPEncoder(args.plip_model, args.device)
    written = filtered = 0
    for index, record in enumerate(pending, 1):
        coordinate_path = Path(str(record.get("coordinate_path") or ""))
        if not coordinate_path.is_file():
            raise FileNotFoundError(
                f"missing coordinate file for {record['slide_id']}: {coordinate_path}"
            )
        with h5py.File(coordinate_path, "r") as handle:
            coords = np.asarray(handle["coords"], dtype=np.int64)
        feature_batches: list[np.ndarray] = []
        tissue_values: list[float] = []
        with openslide.OpenSlide(str(record["raw_slide_path"])) as wsi:
            for start in range(0, len(coords), args.batch_size):
                images = [
                    read_field(wsi, int(x), int(y), 5)
                    for x, y in coords[start : start + args.batch_size]
                ]
                feature_batches.append(
                    encoder.image_features(images, batch_size=args.batch_size)
                )
                tissue_values.extend(tissue_fraction(image) for image in images)
        features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        tissue = np.asarray(tissue_values, dtype=np.float32)
        keep = np.isfinite(features).all(axis=1)
        keep &= np.linalg.norm(features, axis=1) > 1e-12
        keep &= tissue >= args.min_tissue_fraction
        filtered += int((~keep).sum())
        features = features[keep]
        coords = coords[keep]
        tissue = tissue[keep]
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        target = Path(str(record["plip_feature_cache"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            coords=coords,
            features=features,
            tissue_fraction=tissue,
        )
        os.replace(temporary, target)
        written += 1
        print(f"[{index}/{len(pending)}] {record['slide_id']}: {len(coords)} ROIs", flush=True)
    print(f"written={written} background_filtered={filtered}")


if __name__ == "__main__":
    main()
