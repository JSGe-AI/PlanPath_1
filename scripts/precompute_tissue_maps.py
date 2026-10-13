#!/usr/bin/env python3
"""Build question-independent CONCH-prototype tissue maps for PlanPath manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.io_utils import write_json_atomic, write_jsonl_atomic  # noqa: E402
from planpath.core.tissue_map import TissuePrototypeModel  # noqa: E402


DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "artifacts" / "conch_features"
DEFAULT_PROTOTYPE_ROOT = PROJECT_ROOT / "models" / "tissue_prototypes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "tissue_maps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--prototype-root", type=Path, default=DEFAULT_PROTOTYPE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-manifest-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def feature_path(feature_root: Path, row: dict) -> Path:
    raw = Path(str(row["raw_slide_path"]))
    return feature_root / raw.parent.name / "conch_v1_5" / f"{raw.stem}_0_1024.npy"


def main() -> None:
    args = parse_args()
    models: dict[str, TissuePrototypeModel] = {}
    all_rows: list[tuple[Path, dict]] = []
    for manifest in args.manifest:
        all_rows.extend((manifest, item) for item in rows(manifest))
    if args.max_slides is not None:
        all_rows = all_rows[: args.max_slides]
    results: list[dict] = []
    updated_by_manifest: dict[Path, list[dict]] = {path: rows(path) for path in args.manifest}
    updated_index = {
        (path, str(row["slide_id"])): row
        for path, manifest_rows in updated_by_manifest.items()
        for row in manifest_rows
    }
    started = time.time()
    for index, (manifest, row) in enumerate(all_rows, 1):
        cancer = str(row["cancer"])
        if cancer not in models:
            models[cancer] = TissuePrototypeModel.load(cancer, args.prototype_root)
        source = feature_path(args.feature_root, row)
        destination = (
            args.output_root / cancer / str(row["patient_id"]) / f"{row['slide_id']}.npz"
        )
        preview = destination.with_suffix(".png")
        updated = updated_index[(manifest, str(row["slide_id"]))]
        updated["conch_feature_cache"] = str(source)
        updated["tissue_map_cache"] = str(destination)
        updated["tissue_map_preview"] = str(preview)
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            summary = models[cancer].build_map(
                source,
                destination,
                preview,
                slide_id=str(row["slide_id"]),
                device=args.device,
                overwrite=args.overwrite,
            )
            summary.update(slide_id=row["slide_id"], cancer=cancer, source=str(source))
        except Exception as error:
            summary = {
                "slide_id": row["slide_id"],
                "cancer": cancer,
                "source": str(source),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            updated["tissue_map_cache"] = ""
            updated["tissue_map_preview"] = ""
            if args.strict:
                raise
        results.append(summary)
        elapsed = max(time.time() - started, 1e-6)
        print(
            f"[{index}/{len(all_rows)}] {row['slide_id']} {summary['status']} "
            f"({index / elapsed:.2f} slides/s)",
            flush=True,
        )

    args.output_manifest_root.mkdir(parents=True, exist_ok=True)
    for manifest, manifest_rows in updated_by_manifest.items():
        output_manifest = args.output_manifest_root / manifest.name
        write_jsonl_atomic(output_manifest, manifest_rows)
    audit = {
        "schema_version": "1.0",
        "feature_root": str(args.feature_root),
        "prototype_root": str(args.prototype_root),
        "output_root": str(args.output_root),
        "requested_slide_count": len(all_rows),
        "written": sum(item["status"] == "written" for item in results),
        "cached": sum(item["status"] == "cached" for item in results),
        "errors": [item for item in results if item["status"] == "error"],
        "model_signatures": {key: value.signature for key, value in models.items()},
    }
    write_json_atomic(args.output_manifest_root / "tissue_map_precompute_audit.json", audit)
    if audit["errors"]:
        print(
            f"completed with {len(audit['errors'])} missing/failed tissue maps; "
            "those manifest rows are configured for PLIP-only fallback",
            flush=True,
        )


if __name__ == "__main__":
    main()
