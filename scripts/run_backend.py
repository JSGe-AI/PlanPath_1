#!/usr/bin/env python3
"""Base runner used by the final PlanPath entry point."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.config import AgentConfig  # noqa: E402
from planpath.core.controller import QwenController, _is_pn_question  # noqa: E402
from planpath.core.data import build_tasks, load_questions, load_slides  # noqa: E402
from planpath.core.histoplus import HistoPLUSAnalyzer  # noqa: E402
from planpath.core.io_utils import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402
from planpath.core.llm_client import OpenAICompatClient  # noqa: E402
from planpath.core.observer import PathoObserver, VisualAnalyzer  # noqa: E402
from planpath.core.plip import LocalPLIPEncoder, ROISelector  # noqa: E402
from planpath.core.precompute_cache import PrecomputedRetrievalCache  # noqa: E402
from planpath.core.structured import StructuredGenerator  # noqa: E402
from planpath.core.titan import LazyTitanTextEncoder, TitanSlideRetriever  # noqa: E402
from planpath.core.workflow import METHOD, PlanPathCore  # noqa: E402


def project_path(value: str | Path) -> Path:
    """Resolve repository-relative config paths independently of the caller's cwd."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, required=True)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = read_json(known.config)
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--manifest", type=project_path, default=project_path(defaults["manifest"]))
    parser.add_argument("--vqa", type=project_path, default=project_path(defaults["vqa"]))
    parser.add_argument("--titan-model", type=project_path, default=project_path(defaults["titan_model"]))
    parser.add_argument("--plip-model", type=project_path, default=project_path(defaults["plip_model"]))
    parser.add_argument(
        "--tissue-map-root",
        type=project_path,
        default=project_path(defaults.get("tissue_map_root", "artifacts/tissue_maps")),
    )
    parser.add_argument(
        "--histoplus-model-root",
        type=project_path,
        default=project_path(defaults.get("histoplus_model_root", "histoplus_weights")),
    )
    parser.add_argument(
        "--cell-seg-root",
        type=project_path,
        default=project_path(defaults.get("cell_seg_root", "artifacts/cell_seg")),
    )
    parser.add_argument("--cache-root", type=project_path, default=project_path(defaults["cache_root"]))
    parser.add_argument("--clinical-domain", default=defaults.get("clinical_domain"))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument("--qwen-endpoint", default=defaults.get("qwen_endpoint", "http://127.0.0.1:7050/v1"))
    parser.add_argument("--qwen-model", default=defaults.get("qwen_model", "PlanPath-Qwen3.5-4B"))
    parser.add_argument("--patho-endpoint", default=defaults.get("patho_endpoint", "http://127.0.0.1:7051/v1"))
    parser.add_argument("--patho-model", default=defaults.get("patho_model", "PlanPath-Patho-R1-7B"))
    parser.add_argument("--generation-seed", type=int, default=int(defaults.get("generation_seed", 0)))
    parser.add_argument("--top-k-slides", type=int, default=int(defaults.get("top_k_slides", 5)))
    parser.add_argument("--low-power-top-k", type=int, default=int(defaults.get("low_power_top_k", 5)))
    parser.add_argument("--high-power-top-k", type=int, default=int(defaults.get("high_power_top_k", 5)))
    parser.add_argument("--max-zoom-rounds-per-slide", type=int, default=int(defaults.get("max_zoom_rounds_per_slide", 2)))
    parser.add_argument("--min-direct-evidence", type=int, default=int(defaults.get("min_direct_evidence", 2)))
    parser.add_argument("--roi-min-center-distance", type=int, default=int(defaults.get("roi_min_center_distance", 12288)))
    parser.add_argument("--min-zoom-parent-rois", type=int, default=int(defaults.get("min_zoom_parent_rois", 2)))
    parser.add_argument("--patho-concurrency", type=int, default=int(defaults.get("patho_concurrency", 5)))
    parser.add_argument(
        "--no-tissue-navigation",
        action="store_true",
        default=not bool(defaults.get("tissue_navigation_enabled", True)),
    )
    parser.add_argument(
        "--no-histoplus",
        action="store_true",
        default=not bool(defaults.get("histoplus_enabled", True)),
    )
    parser.add_argument(
        "--no-precomputed-cell-seg",
        action="store_true",
        default=not bool(defaults.get("precomputed_cell_seg_enabled", True)),
    )
    parser.add_argument(
        "--histoplus-strict",
        action="store_true",
        help="Fail a slide instead of falling back when HistoPLUS is unavailable.",
    )
    parser.add_argument("--task-retries", type=int, default=int(defaults.get("task_retries", 2)))
    parser.add_argument("--patient-id")
    parser.add_argument("--qa-id")
    parser.add_argument(
        "--task-key",
        action="append",
        default=[],
        metavar="PATIENT_ID/QA_ID",
        help="Run only this task key; repeat the option to run a small task set in one process.",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-unit",
        choices=("task", "patient"),
        default=str(defaults.get("shard_unit", "task")),
        help=(
            "Partition individual tasks (legacy behavior) or keep every question "
            "from one patient in the same shard so patient-level caches remain hot."
        ),
    )
    parser.add_argument("--output", type=project_path, default=project_path(defaults["output"]))
    parser.add_argument("--trace-root", type=project_path, default=project_path(defaults["trace_root"]))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument(
        "--no-retrieval-cache",
        action="store_true",
        help="Ignore persistent fixed plans and TITAN rankings.",
    )
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.task_retries < 0:
        parser.error("--task-retries cannot be negative")
    return args


def build_agent(args: argparse.Namespace) -> PlanPathCore:
    config = AgentConfig(
        method=str(defaults_get(args.config, "method", METHOD)),
        titan_model_path=args.titan_model,
        plip_model_path=args.plip_model,
        tissue_map_root=args.tissue_map_root,
        histoplus_model_root=args.histoplus_model_root,
        cell_seg_root=args.cell_seg_root,
        cache_root=args.cache_root,
        clinical_domain=args.clinical_domain,
        device=args.device,
        top_k_slides=args.top_k_slides,
        low_power_top_k=args.low_power_top_k,
        high_power_top_k=args.high_power_top_k,
        max_zoom_rounds_per_slide=args.max_zoom_rounds_per_slide,
        min_direct_evidence=args.min_direct_evidence,
        roi_min_center_distance=args.roi_min_center_distance,
        min_zoom_parent_rois=args.min_zoom_parent_rois,
        patho_concurrency=args.patho_concurrency,
        tissue_navigation_enabled=not args.no_tissue_navigation,
        histoplus_enabled=not args.no_histoplus,
        precomputed_cell_seg_enabled=not args.no_precomputed_cell_seg,
        histoplus_required_top_k=int(defaults_get(args.config, "histoplus_required_top_k", 5)),
        histoplus_conditional_top_k=int(defaults_get(args.config, "histoplus_conditional_top_k", 2)),
        histoplus_batch_size=int(defaults_get(args.config, "histoplus_batch_size", 5)),
        histoplus_inference_image_size=int(defaults_get(args.config, "histoplus_inference_image_size", 784)),
        histoplus_min_type_confidence=float(defaults_get(args.config, "histoplus_min_type_confidence", 0.5)),
        histoplus_mixed_precision=bool(defaults_get(args.config, "histoplus_mixed_precision", True)),
        histoplus_save_overlay=bool(defaults_get(args.config, "histoplus_save_overlay", False)),
        histoplus_fail_open=not args.histoplus_strict,
        request_timeout=float(defaults_get(args.config, "request_timeout", 120.0)),
        qwen_endpoint=args.qwen_endpoint,
        qwen_model=args.qwen_model,
        patho_endpoint=args.patho_endpoint,
        patho_model=args.patho_model,
        generation_seed=args.generation_seed,
        accelerated_evaluation_enabled=bool(
            defaults_get(args.config, "accelerated_evaluation_enabled", False)
        ),
        patient_visual_cache_enabled=bool(
            defaults_get(args.config, "patient_visual_cache_enabled", False)
        ),
        accelerated_skip_low_power_review_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_skip_low_power_review_enabled",
                False,
            )
        ),
        accelerated_skip_non_actionable_extent_review_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_skip_non_actionable_extent_review_enabled",
                False,
            )
        ),
        accelerated_reuse_complete_visual_scan_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_reuse_complete_visual_scan_enabled",
                False,
            )
        ),
        accelerated_protect_complex_and_interface_slide_pool_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_protect_complex_and_interface_slide_pool_enabled",
                False,
            )
        ),
        accelerated_protected_roi_floor=int(
            defaults_get(args.config, "accelerated_protected_roi_floor", 5)
        ),
        accelerated_regional_node_20x_floor_enabled=bool(
            defaults_get(
                args.config,
                "accelerated_regional_node_20x_floor_enabled",
                False,
            )
        ),
        accelerated_regional_node_20x_top_k=int(
            defaults_get(args.config, "accelerated_regional_node_20x_top_k", 16)
        ),
        accelerated_special_low_power_top_k=int(
            defaults_get(args.config, "accelerated_special_low_power_top_k", 12)
        ),
        accelerated_special_20x_top_k=int(
            defaults_get(args.config, "accelerated_special_20x_top_k", 16)
        ),
        accelerated_special_40x_top_k=int(
            defaults_get(args.config, "accelerated_special_40x_top_k", 8)
        ),
        accelerated_special_20x_parent_top_k=int(
            defaults_get(args.config, "accelerated_special_20x_parent_top_k", 12)
        ),
        accelerated_special_40x_parent_top_k=int(
            defaults_get(args.config, "accelerated_special_40x_parent_top_k", 8)
        ),
        accelerated_slide_promotion_top_k=int(
            defaults_get(args.config, "accelerated_slide_promotion_top_k", 8)
        ),
    )
    qwen_client = OpenAICompatClient(
        config.qwen_endpoint, config.qwen_model, config.request_timeout, config.generation_seed
    )
    patho_client = OpenAICompatClient(
        config.patho_endpoint, config.patho_model, config.request_timeout, config.generation_seed
    )
    qwen_verifier_client = OpenAICompatClient(
        config.qwen_endpoint,
        config.qwen_model,
        config.request_timeout,
        config.generation_seed + 1,
    )
    if not args.skip_health_check:
        print(f"Qwen service: {qwen_client.health()}", flush=True)
        print(f"Patho-R1 service: {patho_client.health()}", flush=True)
    controller = QwenController(StructuredGenerator(qwen_client), config)
    titan = TitanSlideRetriever(
        LazyTitanTextEncoder(config.titan_model_path, config.device)
    )
    selector = ROISelector(
        LocalPLIPEncoder(config.plip_model_path, config.device),
        low_power_top_k=config.low_power_top_k,
        high_power_top_k=config.high_power_top_k,
        min_center_distance=config.roi_min_center_distance,
        tissue_navigation_enabled=config.tissue_navigation_enabled,
        accelerated_evaluation_enabled=config.accelerated_evaluation_enabled,
        accelerated_protected_roi_floor=config.accelerated_protected_roi_floor,
        accelerated_special_low_power_top_k=config.accelerated_special_low_power_top_k,
        accelerated_regional_node_20x_top_k=config.accelerated_regional_node_20x_top_k,
        accelerated_special_20x_top_k=config.accelerated_special_20x_top_k,
        accelerated_special_40x_top_k=config.accelerated_special_40x_top_k,
    )
    visual = VisualAnalyzer(
        selector,
        PathoObserver(
            StructuredGenerator(patho_client),
            config,
            verifier_generator=StructuredGenerator(qwen_verifier_client),
        ),
        HistoPLUSAnalyzer(config) if config.histoplus_enabled else None,
    )
    retrieval_cache = (
        None
        if args.no_retrieval_cache
        else PrecomputedRetrievalCache(
            config.cache_root,
            config.qwen_model,
            config.titan_model_path,
            config.generation_seed,
        )
    )
    return PlanPathCore(
        controller, titan, visual, config, retrieval_cache=retrieval_cache
    )


def defaults_get(config_path: Path, key: str, default: Any) -> Any:
    """Read one infrequently overridden setting without widening CLI noise."""

    return read_json(config_path).get(key, default)


def prediction_record(
    question: dict[str, Any], trace: dict[str, Any], trace_path: Path, method: str
) -> dict[str, Any]:
    final = trace["final_answer"]
    return {
        "method": method,
        "patient_id": question["patient_id"],
        "qa_id": question["qa_id"],
        "question_type": question["question_type"],
        "question": question["question"],
        "options": question.get("options", []),
        "reference_answer": question.get("reference_answer"),
        "prediction": final["answer"],
        **final,
        "trace_path": portable_trace_path(trace_path),
        "error": None,
    }


def portable_trace_path(path: Path) -> str:
    """Store project-relative traces so moving the repository is harmless."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    questions = load_questions(args.vqa)
    slides = load_slides(args.manifest)
    tasks = build_tasks(
        questions, slides, patient_id=args.patient_id, qa_id=args.qa_id
    )
    if args.task_key:
        requested: set[tuple[str, str]] = set()
        for raw in args.task_key:
            if "/" not in raw:
                raise ValueError(f"invalid --task-key {raw!r}; expected PATIENT_ID/QA_ID")
            requested.add(tuple(raw.split("/", 1)))
        tasks = [
            item
            for item in tasks
            if (str(item[0]["patient_id"]), str(item[0]["qa_id"])) in requested
        ]
        found = {
            (str(question["patient_id"]), str(question["qa_id"]))
            for question, _ in tasks
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown --task-key values: {missing}")
    if bool(defaults_get(args.config, "prefer_complete_scan_source_tasks", False)):
        # A pN decision requires complete regional-node coverage.  Running it
        # before the paired existential yes/no question guarantees that the
        # latter can reuse a complete scan even in node-positive patients that
        # would otherwise stop early.
        tasks.sort(
            key=lambda item: (
                str(item[0]["patient_id"]),
                0 if _is_pn_question(item[0]) else 1,
                str(item[0]["qa_id"]),
            )
        )
    if args.shard_unit == "patient":
        patient_ids = sorted(
            {str(question["patient_id"]) for question, _ in tasks}
        )
        selected_patients = set(
            patient_ids[args.shard_index :: args.shard_count]
        )
        tasks = [
            item
            for item in tasks
            if str(item[0]["patient_id"]) in selected_patients
        ]
    else:
        tasks = tasks[args.shard_index :: args.shard_count]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    existing = read_jsonl(args.output) if args.output.is_file() else []
    active_method = str(defaults_get(args.config, "method", METHOD))
    completed = {
        (str(item.get("patient_id")), str(item.get("qa_id")))
        for item in existing
        if item.get("method") == active_method and not item.get("error")
    }
    agent = build_agent(args)
    records = list(existing)
    for index, (question, patient_slides) in enumerate(tasks, 1):
        key = (str(question["patient_id"]), str(question["qa_id"]))
        if key in completed and not args.overwrite:
            print(f"[{index}/{len(tasks)}] skip completed {key}", flush=True)
            continue
        trace_path = args.trace_root / key[0] / f"{key[1]}.json"
        record: dict[str, Any] | None = None
        last_error: Exception | None = None
        last_traceback = ""
        for task_attempt in range(args.task_retries + 1):
            try:
                trace = agent.run(
                    question,
                    patient_slides,
                    checkpoint=lambda value, path=trace_path: write_json_atomic(path, value),
                )
                record = prediction_record(question, trace, trace_path, active_method)
                record["task_attempts"] = task_attempt + 1
                print(
                    f"[{index}/{len(tasks)}] {key}: {record['prediction']} "
                    f"(attempt {task_attempt + 1})",
                    flush=True,
                )
                break
            except Exception as error:
                last_error = error
                last_traceback = traceback.format_exc()
                failed_trace = read_json(trace_path) if trace_path.is_file() else {}
                failed_trace.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    failure_phase=getattr(error, "phase", None),
                    failed_task_attempt=task_attempt + 1,
                    generation_audit=getattr(error, "audit", []),
                    last_response_excerpt=getattr(error, "last_response", "")[-2000:],
                )
                write_json_atomic(trace_path, failed_trace)
                if task_attempt < args.task_retries:
                    print(
                        f"[{index}/{len(tasks)}] {key}: retry after "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                if args.fail_fast:
                    raise
        if record is None:
            assert last_error is not None
            record = {
                "method": active_method,
                "patient_id": key[0],
                "qa_id": key[1],
                "question": question["question"],
                "reference_answer": question.get("reference_answer"),
                "prediction": None,
                "trace_path": portable_trace_path(trace_path),
                "error": f"{type(last_error).__name__}: {last_error}",
                "failure_phase": getattr(last_error, "phase", None),
                "task_attempts": args.task_retries + 1,
                "traceback": last_traceback,
            }
            print(f"[{index}/{len(tasks)}] {key}: ERROR {last_error}", flush=True)
        records = [
            item
            for item in records
            if (str(item.get("patient_id")), str(item.get("qa_id"))) != key
        ]
        records.append(record)
        records.sort(key=lambda item: (str(item.get("patient_id")), str(item.get("qa_id"))))
        write_jsonl_atomic(args.output, records)


if __name__ == "__main__":
    main()
