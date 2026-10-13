#!/usr/bin/env python3
"""Evaluate PlanPath JSONL predictions against a patient-level VQA dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.io_utils import (  # noqa: E402
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from planpath.evaluation import evaluate_records  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=project_path, required=True)
    parser.add_argument("--vqa", type=project_path, required=True)
    parser.add_argument("--output-dir", type=project_path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--percentage-tolerances",
        type=float,
        nargs="+",
        default=[5.0, 10.0],
        metavar="POINTS",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def reference_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("VQA input must be a JSON array")
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for patient in payload:
        patient_id = str(patient["patient_id"])
        for qa in patient.get("qa_pairs", []):
            key = patient_id, str(qa["qa_id"])
            if key in references:
                raise ValueError(f"duplicate VQA key: {key}")
            references[key] = {
                "question_type": str(qa["question_type"]),
                "question": str(qa["question"]),
                "options": [str(value) for value in qa.get("options", [])],
                "answer": str(qa["answer"]),
            }
    return references


def prediction_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_jsonl(path):
        key = str(record.get("patient_id")), str(record.get("qa_id"))
        if key in predictions:
            raise ValueError(f"duplicate prediction key: {key}")
        predictions[key] = record
    return predictions


def scalar_metrics(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from scalar_metrics(nested, path)
    elif not isinstance(value, (list, tuple)):
        yield prefix, value


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in scalar_metrics(metrics):
            writer.writerow({"metric": metric, "value": value})
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    references = reference_lookup(args.vqa)
    predictions = prediction_lookup(args.predictions)
    reference_keys = set(references)
    prediction_keys = set(predictions)
    missing = sorted(reference_keys - prediction_keys)
    extra = sorted(prediction_keys - reference_keys)
    if extra:
        raise ValueError(f"predictions contain {len(extra)} unknown task(s)")
    if missing and not args.allow_partial:
        raise ValueError(
            f"predictions are missing {len(missing)} task(s); use --allow-partial "
            "to evaluate the available subset"
        )

    matched_keys = sorted(reference_keys & prediction_keys)
    if not matched_keys:
        raise ValueError("no matched prediction/reference pairs")
    matched_references = {key: references[key] for key in matched_keys}
    matched_predictions = {key: predictions[key] for key in matched_keys}
    dataset_name = args.dataset_name or args.vqa.stem
    metrics, cases = evaluate_records(
        matched_references,
        matched_predictions,
        dataset_name=dataset_name,
        percentage_tolerances=args.percentage_tolerances,
        reference_count=len(references),
        prediction_count=len(predictions),
        missing_count=len(missing),
        extra_count=len(extra),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    write_metrics_csv(args.output_dir / "metrics.csv", metrics)
    write_jsonl_atomic(args.output_dir / "evaluation_cases.jsonl", cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
