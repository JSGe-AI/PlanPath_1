"""Versioned persistent cache for fixed plans and TITAN slide rankings."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic
from .prompts import PLAN_PROMPT_VERSION
from .schemas import EvidencePlan, SlideScore


CACHE_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_CONTRACT_VERSION = "1.0"
RANKING_ALGORITHM_VERSION = "titan-aligned-cosine-max-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:100] or _sha256(value)[:16]


def _path_state(path_value: str | Path | None) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    try:
        stat = path.stat()
    except (FileNotFoundError, OSError):
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class PrecomputedRetrievalCache:
    """Persist immutable plan and full per-patient TITAN score vectors."""

    def __init__(
        self,
        root: Path,
        qwen_model: str,
        titan_model_path: Path,
        generation_seed: int = 0,
    ) -> None:
        self.root = root / "planpath.quantitative"
        self.qwen_model = qwen_model
        self.titan_model_path = titan_model_path.resolve()
        self.generation_seed = generation_seed
        self.metrics = {
            "plan_hits": 0,
            "plan_misses": 0,
            "plan_writes": 0,
            "ranking_hits": 0,
            "ranking_misses": 0,
            "ranking_writes": 0,
        }

    @staticmethod
    def clean_question(question: dict[str, Any]) -> dict[str, Any]:
        value = {
            "question_type": str(question["question_type"]),
            "question": str(question["question"]),
            "options": [str(item) for item in question.get("options", [])],
        }
        if question.get("clinical_domain"):
            value["clinical_domain"] = str(question["clinical_domain"])
        return value

    def question_signature(self, question: dict[str, Any]) -> str:
        return _sha256(self.clean_question(question))

    def planner_signature(self) -> str:
        return _sha256(
            {
                "qwen_model": self.qwen_model,
                "plan_prompt_version": PLAN_PROMPT_VERSION,
                # Use an explicit contract version instead of serialized
                # Pydantic JSON Schema, whose formatting varies by release.
                "plan_schema_contract_version": PLAN_SCHEMA_CONTRACT_VERSION,
                "generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "thinking": False,
                    "seed": self.generation_seed,
                },
            }
        )

    def titan_signature(self) -> str:
        return _sha256(
            {
                "model_root": str(self.titan_model_path),
                "files": [
                    _path_state(self.titan_model_path / name)
                    for name in (
                        "config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "model.safetensors",
                        "text_transformer.py",
                        "conch_tokenizer.py",
                    )
                ],
            }
        )

    @staticmethod
    def plan_signature(plan: EvidencePlan) -> str:
        return _sha256(plan.model_dump(mode="json"))

    @staticmethod
    def slide_signature(slides: list[dict[str, Any]]) -> str:
        records = []
        for slide in sorted(slides, key=lambda item: str(item["slide_id"])):
            records.append(
                {
                    "slide_id": str(slide["slide_id"]),
                    "titan": _path_state(slide.get("titan_aligned_feature")),
                    "plip": _path_state(slide.get("plip_feature_cache")),
                    "wsi": _path_state(slide.get("raw_slide_path")),
                }
            )
        return _sha256(records)

    def _plan_path(self, question: dict[str, Any]) -> Path:
        return (
            self.root
            / "plans"
            / self.planner_signature()[:16]
            / f"{self.question_signature(question)}.json"
        )

    def plan_path(self, question: dict[str, Any]) -> Path:
        return self._plan_path(question)

    def load_plan(self, question: dict[str, Any]) -> EvidencePlan | None:
        path = self._plan_path(question)
        if not path.is_file():
            self.metrics["plan_misses"] += 1
            return None
        record = read_json(path)
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "question_signature": self.question_signature(question),
            "planner_signature": self.planner_signature(),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            self.metrics["plan_misses"] += 1
            return None
        plan = EvidencePlan.model_validate(record["plan"])
        if record.get("plan_signature") != self.plan_signature(plan):
            raise ValueError(f"cached plan content hash mismatch: {path}")
        self.metrics["plan_hits"] += 1
        return plan

    def save_plan(
        self,
        question: dict[str, Any],
        plan: EvidencePlan,
        generation_audit: list[dict[str, Any]],
    ) -> Path:
        path = self._plan_path(question)
        record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "record_type": "fixed_evidence_plan",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_signature": self.question_signature(question),
            "planner_signature": self.planner_signature(),
            "qwen_model": self.qwen_model,
            "plan_prompt_version": PLAN_PROMPT_VERSION,
            "question": self.clean_question(question),
            "plan_signature": self.plan_signature(plan),
            "plan": plan.model_dump(mode="json"),
            "slide_retrieval_descriptions": list(
                plan.slide_retrieval_descriptions
            ),
            "generation_audit": generation_audit,
        }
        write_json_atomic(path, record)
        self.metrics["plan_writes"] += 1
        return path

    def ranking_key(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        plan: EvidencePlan,
    ) -> str:
        return _sha256(
            {
                "question_signature": self.question_signature(question),
                "planner_signature": self.planner_signature(),
                "plan_signature": self.plan_signature(plan),
                "titan_signature": self.titan_signature(),
                "slide_signature": self.slide_signature(slides),
                "ranking_algorithm": RANKING_ALGORITHM_VERSION,
            }
        )

    def _ranking_path(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        plan: EvidencePlan,
    ) -> Path:
        cancer = str(slides[0].get("cancer") or "unknown") if slides else "unknown"
        patient_id = str(question["patient_id"])
        qa_id = str(question["qa_id"])
        key = self.ranking_key(question, slides, plan)
        return (
            self.root
            / "rankings"
            / _safe_component(cancer)
            / _safe_component(patient_id)
            / _safe_component(qa_id)
            / f"{key}.json"
        )

    def load_ranking(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        plan: EvidencePlan,
    ) -> tuple[list[SlideScore], list[str], Path] | None:
        path = self._ranking_path(question, slides, plan)
        if not path.is_file():
            self.metrics["ranking_misses"] += 1
            return None
        record = read_json(path)
        expected_key = self.ranking_key(question, slides, plan)
        if (
            record.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or record.get("ranking_key") != expected_key
        ):
            self.metrics["ranking_misses"] += 1
            return None
        ranking = [SlideScore.model_validate(item) for item in record["ranking"]]
        self.metrics["ranking_hits"] += 1
        return ranking, [str(item) for item in record["unavailable_slide_ids"]], path

    def save_ranking(
        self,
        question: dict[str, Any],
        slides: list[dict[str, Any]],
        plan: EvidencePlan,
        ranking: list[SlideScore],
        unavailable_slide_ids: list[str],
    ) -> Path:
        path = self._ranking_path(question, slides, plan)
        record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "record_type": "fixed_titan_slide_ranking",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ranking_key": self.ranking_key(question, slides, plan),
            "question_signature": self.question_signature(question),
            "planner_signature": self.planner_signature(),
            "plan_signature": self.plan_signature(plan),
            "titan_signature": self.titan_signature(),
            "slide_signature": self.slide_signature(slides),
            "ranking_algorithm": RANKING_ALGORITHM_VERSION,
            "question": self.clean_question(question),
            "plan": plan.model_dump(mode="json"),
            "slide_retrieval_descriptions": list(
                plan.slide_retrieval_descriptions
            ),
            "aggregation": "maximum cosine similarity across descriptions",
            "score_semantics": "retrieval priority only; never diagnostic evidence",
            "ranking": [item.model_dump(mode="json") for item in ranking],
            "unavailable_slide_ids": list(unavailable_slide_ids),
        }
        write_json_atomic(path, record)
        self.metrics["ranking_writes"] += 1
        return path
