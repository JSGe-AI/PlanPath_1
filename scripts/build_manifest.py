#!/usr/bin/env python3
"""Attach exact TITAN aligned-feature paths and remove all site-label fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.io_utils import read_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402


DEFAULT_TITAN_ROOT = PROJECT_ROOT / "artifacts" / "titan_features"
DEFAULT_PREPARED_PLIP_ROOT = PROJECT_ROOT / "artifacts" / "plip_features"
DEFAULT_GENERATED_PLIP_ROOT = PROJECT_ROOT / "artifacts" / "generated_plip_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--titan-feature-root", type=Path, default=DEFAULT_TITAN_ROOT)
    parser.add_argument("--prepared-plip-root", type=Path, default=DEFAULT_PREPARED_PLIP_ROOT)
    parser.add_argument("--generated-plip-root", type=Path, default=DEFAULT_GENERATED_PLIP_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any aligned vector is missing.")
    parser.add_argument(
        "--allow-missing-inputs",
        action="store_true",
        help="Write and audit rows whose WSI or PLIP cache is currently missing.",
    )
    return parser.parse_args()


def feature_path(record: dict, root: Path) -> Path:
    batch = Path(str(record["raw_slide_path"])).parent.name
    stem = str(record.get("slide_stem") or Path(str(record["raw_slide_path"])).stem)
    return root / batch / "aligned" / f"{stem}_0_1024.npy"


def build(
    source_manifest: Path,
    titan_root: Path,
    prepared_plip_root: Path = DEFAULT_PREPARED_PLIP_ROOT,
    generated_plip_root: Path = DEFAULT_GENERATED_PLIP_ROOT,
) -> tuple[list[dict], dict]:
    output: list[dict] = []
    missing_titan: list[str] = []
    missing_inputs: list[str] = []
    for source in read_jsonl(source_manifest):
        titan_path = feature_path(source, titan_root)
        if not titan_path.is_file():
            missing_titan.append(str(source["slide_id"]))
        cancer = str(source.get("cancer") or "unknown")
        patient_id = str(source["patient_id"])
        slide_id = str(source["slide_id"])
        prepared_plip = prepared_plip_root / cancer / patient_id / f"{slide_id}.npz"
        source_plip = Path(str(source["plip_feature_cache"]))
        generated_plip = generated_plip_root / cancer / patient_id / f"{slide_id}.npz"
        selected_plip = (
            prepared_plip
            if prepared_plip.is_file()
            else source_plip
            if source_plip.is_file()
            else generated_plip
        )
        record = {
            "cancer": cancer,
            "patient_id": patient_id,
            "slide_index": int(source["slide_index"]),
            "slide_id": slide_id,
            "raw_slide_path": str(source["raw_slide_path"]),
            "coordinate_path": str(source.get("coordinate_path") or ""),
            "plip_feature_cache": str(selected_plip),
            "titan_aligned_feature": str(titan_path) if titan_path.is_file() else None,
        }
        for key in ("raw_slide_path", "plip_feature_cache"):
            if not Path(record[key]).is_file():
                missing_inputs.append(record[key])
        output.append(record)
    output.sort(key=lambda item: (item["patient_id"], item["slide_index"]))
    audit = {
        "schema_version": "1.0",
        "source_manifest": str(source_manifest),
        "titan_feature_root": str(titan_root),
        "site_labels_copied": False,
        "counts": {
            "slides": len(output),
            "patients": len({item["patient_id"] for item in output}),
            "titan_features_found": len(output) - len(missing_titan),
            "titan_features_missing": len(missing_titan),
        },
        "missing_titan_slide_ids": missing_titan,
        "missing_input_paths": missing_inputs,
    }
    return output, audit


def main() -> None:
    args = parse_args()
    records, audit = build(
        args.source_manifest,
        args.titan_feature_root,
        args.prepared_plip_root,
        args.generated_plip_root,
    )
    if audit["missing_input_paths"] and not args.allow_missing_inputs:
        raise FileNotFoundError(
            f"{len(audit['missing_input_paths'])} input files are missing; "
            f"first={audit['missing_input_paths'][0]}"
        )
    if args.strict and audit["missing_titan_slide_ids"]:
        raise FileNotFoundError(
            f"{len(audit['missing_titan_slide_ids'])} TITAN vectors are missing; "
            f"first slide={audit['missing_titan_slide_ids'][0]}"
        )
    write_jsonl_atomic(args.output, records)
    audit_path = args.output.with_suffix(".audit.json")
    write_json_atomic(audit_path, audit)
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2))
    print(f"manifest: {args.output}")
    print(f"audit: {audit_path}")


if __name__ == "__main__":
    main()
