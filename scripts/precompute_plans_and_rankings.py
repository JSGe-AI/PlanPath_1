#!/usr/bin/env python3
"""Pre-generate fixed Qwen plans and complete TITAN rankings for later reuse."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.config import AgentConfig  # noqa: E402
from planpath.core.controller import QwenController, normalize_plan_for_question  # noqa: E402
from planpath.core.data import build_tasks, load_questions, load_slides  # noqa: E402
from planpath.core.io_utils import read_json, write_json_atomic  # noqa: E402
from planpath.core.llm_client import OpenAICompatClient  # noqa: E402
from planpath.core.precompute_cache import PrecomputedRetrievalCache  # noqa: E402
from planpath.core.structured import StructuredGenerator  # noqa: E402
from planpath.core.titan import LazyTitanTextEncoder, TitanSlideRetriever  # noqa: E402
from planpath.core.workflow import PlanPathCore  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, required=True)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = read_json(known.config)
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--manifest", type=Path, default=Path(defaults["manifest"]))
    parser.add_argument("--vqa", type=Path, default=Path(defaults["vqa"]))
    parser.add_argument("--cache-root", type=Path, default=Path(defaults["cache_root"]))
    parser.add_argument("--clinical-domain", default=defaults.get("clinical_domain"))
    parser.add_argument("--titan-model", type=Path, default=Path(defaults["titan_model"]))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument("--qwen-endpoint", default=defaults.get("qwen_endpoint", "http://127.0.0.1:7050/v1"))
    parser.add_argument("--qwen-model", default=defaults.get("qwen_model", "PlanPath-Qwen3.5-4B"))
    parser.add_argument("--generation-seed", type=int, default=int(defaults.get("generation_seed", 0)))
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--patient-id")
    parser.add_argument("--qa-id")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--regenerate-plans", action="store_true")
    parser.add_argument("--recompute-rankings", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    questions = load_questions(args.vqa)
    slides_by_patient = load_slides(args.manifest)
    tasks = build_tasks(
        questions,
        slides_by_patient,
        patient_id=args.patient_id,
        qa_id=args.qa_id,
    )[args.shard_index :: args.shard_count]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    config = AgentConfig(
        titan_model_path=args.titan_model,
        cache_root=args.cache_root,
        clinical_domain=args.clinical_domain,
        device=args.device,
        qwen_endpoint=args.qwen_endpoint,
        qwen_model=args.qwen_model,
        request_timeout=args.request_timeout,
        generation_seed=args.generation_seed,
    )
    qwen_client = OpenAICompatClient(
        config.qwen_endpoint,
        config.qwen_model,
        config.request_timeout,
        config.generation_seed,
    )
    controller = QwenController(StructuredGenerator(qwen_client), config)
    cache = PrecomputedRetrievalCache(
        config.cache_root,
        config.qwen_model,
        config.titan_model_path,
        config.generation_seed,
    )
    retriever = TitanSlideRetriever(
        LazyTitanTextEncoder(config.titan_model_path, config.device)
    )
    health_checked = False
    summary = {
        "tasks": len(tasks),
        "completed": 0,
        "errors": 0,
        "generated_plans": 0,
        "reused_plans": 0,
        "computed_rankings": 0,
        "reused_rankings": 0,
        "error_records": [],
    }

    for index, (question, slides) in enumerate(tasks, 1):
        clean_question = PlanPathCore.clean_question(
            question, config.clinical_domain
        )
        label = f"{clean_question['patient_id']}/{clean_question['qa_id']}"
        try:
            plan = None if args.regenerate_plans else cache.load_plan(clean_question)
            if plan is None:
                if not health_checked and not args.skip_health_check:
                    print(f"Qwen service: {qwen_client.health()}", flush=True)
                    health_checked = True
                audit_start = len(controller.audit)
                plan = controller.plan(clean_question)
                cache.save_plan(
                    clean_question, plan, controller.audit[audit_start:]
                )
                summary["generated_plans"] += 1
            else:
                summary["reused_plans"] += 1

            # Runtime applies deterministic task semantics after loading the
            # fixed Qwen plan. Precompute the TITAN vector from that same
            # normalized plan; otherwise the saved ranking key cannot be hit
            # by the actual workflow whenever a task-specific query override
            # (for example renal grading or gastric depth) is active.
            plan = normalize_plan_for_question(plan, clean_question)

            cached = (
                None
                if args.recompute_rankings
                else cache.load_ranking(clean_question, slides, plan)
            )
            if cached is None:
                ranking = retriever.rank(
                    slides, plan.slide_retrieval_descriptions
                )
                path = cache.save_ranking(
                    clean_question,
                    slides,
                    plan,
                    ranking,
                    retriever.last_skipped_slide_ids,
                )
                summary["computed_rankings"] += 1
                action = "ranking-generated"
            else:
                ranking, _, path = cached
                summary["reused_rankings"] += 1
                action = "ranking-reused"
            summary["completed"] += 1
            print(
                f"[{index}/{len(tasks)}] {label}: {action}, slides={len(ranking)}, "
                f"cache={path}",
                flush=True,
            )
        except Exception as error:
            summary["errors"] += 1
            summary["error_records"].append(
                {"patient_id": label.split("/", 1)[0], "qa_id": label.split("/", 1)[1], "error": f"{type(error).__name__}: {error}"}
            )
            print(f"[{index}/{len(tasks)}] {label}: ERROR {error}", flush=True)
            if args.fail_fast:
                raise

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["cache_root"] = str(cache.root)
    summary["cache_metrics"] = dict(cache.metrics)
    summary["titan_metrics"] = dict(retriever.metrics)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = cache.root / "precompute_runs" / f"{run_name}.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
