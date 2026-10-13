"""Dataset-level evaluation utilities for PlanPath predictions."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from planpath.model.quantitative_overlay import allowlisted_quantitative_profile


_ANSWER_PREFIX = re.compile(
    r"^(?:the\s+)?(?:answer|final(?:\s+answer)?|response)\s*(?:is|:|-)\s*",
    re.IGNORECASE,
)
_CHOICE_LABEL = re.compile(
    r"^\s*(?:option\s*)?[\[(]?([a-hA-H])[\])\].:\-]?\s*(?:$|\s)",
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def normalize_text(value: Any) -> str:
    """Normalize an answer while retaining clinically meaningful symbols."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = _ANSWER_PREFIX.sub("", text)
    text = re.sub(r"\bgrade\s+(?:i|1)\b", "grade 1", text)
    text = re.sub(r"\bgrade\s+(?:ii|2)\b", "grade 2", text)
    text = re.sub(r"\bgrade\s+(?:iii|3)\b", "grade 3", text)
    text = re.sub(r"\bgrade\s+(?:iv|4)\b", "grade 4", text)
    text = re.sub(r"(?<=[a-z0-9])[-‐‑–—](?=[a-z0-9])", " ", text)
    text = re.sub(r"[^a-z0-9%+./-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,-")


def token_f1(prediction: Any, reference: Any) -> float:
    predicted = normalize_text(prediction).split()
    target = normalize_text(reference).split()
    if not predicted or not target:
        return float(predicted == target)
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def _choice_index(value: Any, options: list[str]) -> int | None:
    raw = str(value or "").strip()
    normalized = normalize_text(raw)
    normalized_options = [normalize_text(option) for option in options]
    for index, option in enumerate(normalized_options):
        if normalized and normalized == option:
            return index

    label = _CHOICE_LABEL.match(raw)
    if label:
        index = ord(label.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return index

    contained = [
        index
        for index, option in enumerate(normalized_options)
        if option
        and normalized
        and (
            normalized.startswith(option + " ")
            or normalized.endswith(" " + option)
            or f" {option} " in f" {normalized} "
        )
    ]
    return contained[0] if len(contained) == 1 else None


def choice_correct(prediction: Any, reference: Any, options: list[str]) -> bool:
    if not str(prediction or "").strip():
        return False
    if options:
        predicted_index = _choice_index(prediction, options)
        reference_index = _choice_index(reference, options)
        if predicted_index is not None and reference_index is not None:
            return predicted_index == reference_index
    return normalize_text(prediction) == normalize_text(reference)


def short_answer_correct(prediction: Any, reference: Any) -> bool:
    predicted = normalize_text(prediction)
    target = normalize_text(reference)
    return bool(predicted and target) and (predicted in target or target in predicted)


def score_answer(
    prediction: Any,
    reference: Any,
    question_type: str,
    options: list[str],
) -> dict[str, Any]:
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    exact = bool(normalized_prediction) and normalized_prediction == normalized_reference
    if question_type == "short_answer":
        correct = short_answer_correct(prediction, reference)
        f1: float | None = token_f1(prediction, reference)
    else:
        correct = choice_correct(prediction, reference, options)
        f1 = None
    return {
        "correct": correct,
        "normalized_exact_match": exact,
        "token_f1": f1,
    }


def numeric_value(value: Any) -> float | None:
    match = _NUMBER.search(str(value or "").replace(",", ""))
    return float(match.group()) if match else None


def count_value(value: Any) -> int | None:
    number = numeric_value(value)
    if number is not None and number.is_integer():
        return int(number)
    normalized = normalize_text(value)
    for word, count in _COUNT_WORDS.items():
        if re.search(rf"\b{word}\b", normalized):
            return count
    return None


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return round(mean(items), 6) if items else None


def _average(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(mean(items), 6) if items else None


def evaluate_records(
    references: dict[tuple[str, str], dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, Any]],
    *,
    dataset_name: str,
    percentage_tolerances: Iterable[float] = (5.0, 10.0),
    reference_count: int | None = None,
    prediction_count: int | None = None,
    missing_count: int = 0,
    extra_count: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate matched prediction/reference records and return metrics and cases."""

    tolerances = sorted({float(value) for value in percentage_tolerances})
    if any(value < 0 for value in tolerances):
        raise ValueError("percentage tolerances must be non-negative")

    cases: list[dict[str, Any]] = []
    for key in sorted(references):
        reference = references[key]
        record = predictions[key]
        prediction = record.get("prediction")
        question_type = str(reference["question_type"])
        options = [str(value) for value in reference.get("options", [])]
        scores = score_answer(
            prediction,
            reference["answer"],
            question_type,
            options,
        )
        question = str(reference["question"])
        profile = allowlisted_quantitative_profile(
            {"question": question, "question_type": question_type, "options": options}
        )
        quantitative: dict[str, Any] | None = None
        if profile is not None:
            if profile.unit == "percent":
                reference_value = numeric_value(reference["answer"])
                prediction_value = numeric_value(prediction)
            else:
                reference_value = count_value(reference["answer"])
                prediction_value = count_value(prediction)
            absolute_error = (
                abs(float(prediction_value) - float(reference_value))
                if prediction_value is not None and reference_value is not None
                else None
            )
            quantitative = {
                "kind": profile.kind,
                "target": profile.target,
                "unit": profile.unit,
                "reference_value": reference_value,
                "prediction_value": prediction_value,
                "absolute_error": absolute_error,
            }

        cases.append(
            {
                "dataset": dataset_name,
                "patient_id": key[0],
                "qa_id": key[1],
                "question_type": question_type,
                "question": question,
                "options": options,
                "reference_answer": reference["answer"],
                "prediction": prediction,
                **scores,
                "quantitative": quantitative,
                "error": record.get("error"),
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[str(case["question_type"])].append(case)
        by_patient[str(case["patient_id"])].append(case)

    type_metrics: dict[str, dict[str, Any]] = {}
    for question_type in sorted(by_type):
        group = by_type[question_type]
        f1_values = [
            float(case["token_f1"])
            for case in group
            if case["token_f1"] is not None
        ]
        type_metrics[question_type] = {
            "count": len(group),
            "accuracy": _rate(bool(case["correct"]) for case in group),
            "normalized_exact_match": _rate(
                bool(case["normalized_exact_match"]) for case in group
            ),
            "mean_token_f1": _average(f1_values),
        }
        if question_type == "short_answer":
            type_metrics[question_type]["containment_accuracy"] = type_metrics[
                question_type
            ]["accuracy"]

    patient_accuracies = [
        mean(bool(case["correct"]) for case in group)
        for group in by_patient.values()
    ]
    quantitative_cases = [case for case in cases if case["quantitative"]]
    percent_cases = [
        case
        for case in quantitative_cases
        if case["quantitative"]["unit"] == "percent"
    ]
    count_cases = [
        case
        for case in quantitative_cases
        if case["quantitative"]["unit"] == "count"
    ]
    parsed_percent = [
        case
        for case in percent_cases
        if case["quantitative"]["absolute_error"] is not None
    ]
    parsed_counts = [
        case
        for case in count_cases
        if case["quantitative"]["absolute_error"] is not None
    ]
    tolerance_metrics = {
        f"{value:g}": _rate(
            bool(
                case["quantitative"]["absolute_error"] is not None
                and case["quantitative"]["absolute_error"] <= value
            )
            for case in percent_cases
        )
        for value in tolerances
    }

    metrics = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "counts": {
            "reference_questions": reference_count
            if reference_count is not None
            else len(references),
            "prediction_records": prediction_count
            if prediction_count is not None
            else len(predictions),
            "evaluated_questions": len(cases),
            "evaluated_patients": len(by_patient),
            "missing_predictions": missing_count,
            "extra_predictions": extra_count,
        },
        "overall": {
            "accuracy": _rate(bool(case["correct"]) for case in cases),
            "normalized_exact_match": _rate(
                bool(case["normalized_exact_match"]) for case in cases
            ),
            "patient_macro_accuracy": _average(patient_accuracies),
            "error_rate": _rate(bool(case.get("error")) for case in cases),
        },
        "by_question_type": type_metrics,
        "quantitative": {
            "count": len(quantitative_cases),
            "percentage_count": len(percent_cases),
            "parsed_percentage_predictions": len(parsed_percent),
            "mean_absolute_percentage_point_error": _average(
                float(case["quantitative"]["absolute_error"])
                for case in parsed_percent
            ),
            "percentage_accuracy_at_tolerance": tolerance_metrics,
            "focus_count": len(count_cases),
            "parsed_focus_count_predictions": len(parsed_counts),
            "focus_count_exact_accuracy": _rate(
                bool(case["quantitative"]["absolute_error"] == 0)
                for case in count_cases
            ),
        },
    }
    return metrics, cases


__all__ = [
    "choice_correct",
    "count_value",
    "evaluate_records",
    "normalize_text",
    "numeric_value",
    "score_answer",
    "short_answer_correct",
    "token_f1",
]
