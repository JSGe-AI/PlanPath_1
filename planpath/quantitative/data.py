"""Question and label-free slide-manifest loading."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl


def load_questions(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for patient in read_json(path):
        patient_id = str(patient["patient_id"])
        if patient_id in output:
            raise ValueError(f"duplicate VQA patient: {patient_id}")
        output[patient_id] = [
            {
                "patient_id": patient_id,
                "qa_id": str(qa["qa_id"]),
                "question_type": str(qa["question_type"]),
                "question": str(qa["question"]),
                "options": [str(item) for item in qa.get("options", [])],
                # This field is written to predictions only. The workflow removes it.
                "reference_answer": str(qa["answer"]),
            }
            for qa in patient.get("qa_pairs", [])
        ]
    return output


def load_slides(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    required = {"patient_id", "slide_id", "slide_index", "raw_slide_path", "plip_feature_cache"}
    for raw in read_jsonl(path):
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"slide manifest row lacks fields: {missing}")
        slide_id = str(raw["slide_id"])
        if slide_id in seen:
            raise ValueError(f"duplicate slide_id: {slide_id}")
        seen.add(slide_id)
        # Deliberately whitelist fields. A legacy `site` field can never reach v6.
        slide = {
            "cancer": str(raw.get("cancer") or "unknown"),
            "patient_id": str(raw["patient_id"]),
            "slide_id": slide_id,
            "slide_index": int(raw["slide_index"]),
            "raw_slide_path": str(raw["raw_slide_path"]),
            "coordinate_path": str(raw.get("coordinate_path") or ""),
            "plip_feature_cache": str(raw["plip_feature_cache"]),
            "conch_feature_cache": str(raw.get("conch_feature_cache") or ""),
            "tissue_map_cache": str(raw.get("tissue_map_cache") or ""),
            "tissue_map_preview": str(raw.get("tissue_map_preview") or ""),
            "titan_aligned_feature": (
                str(raw["titan_aligned_feature"])
                if raw.get("titan_aligned_feature")
                else None
            ),
        }
        grouped[slide["patient_id"]].append(slide)
    for slides in grouped.values():
        slides.sort(key=lambda item: item["slide_index"])
    return dict(grouped)


def build_tasks(
    questions: dict[str, list[dict[str, Any]]],
    slides: dict[str, list[dict[str, Any]]],
    *,
    patient_id: str | None = None,
    qa_id: str | None = None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    tasks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for current_patient in sorted(questions):
        if patient_id and current_patient != patient_id:
            continue
        patient_slides = slides.get(current_patient, [])
        if questions[current_patient] and not patient_slides:
            raise ValueError(f"patient has questions but no slides: {current_patient}")
        for question in questions[current_patient]:
            if qa_id and str(question["qa_id"]) != qa_id:
                continue
            tasks.append((question, patient_slides))
    return sorted(tasks, key=lambda item: (item[0]["patient_id"], item[0]["qa_id"]))
