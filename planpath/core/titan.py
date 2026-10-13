"""TITAN text-to-slide retrieval over precomputed aligned embeddings."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import SlideScore


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalTitanTextEncoder:
    """Load only TITAN's text tower, weights, and tokenizer from local files."""

    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from safetensors import safe_open
            from transformers import PreTrainedTokenizerFast
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TITAN text retrieval requires torch, transformers, and safetensors"
            ) from error
        self._torch = torch
        self.device = torch.device(device)
        model_path = model_path.resolve()
        # Direct submodule imports avoid stale Hugging Face dynamic-module caches
        # and do not import TITAN's unused timm-based vision tower.
        parent = str(model_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            config_module = importlib.import_module(
                f"{model_path.name}.configuration_titan"
            )
            text_module = importlib.import_module(
                f"{model_path.name}.text_transformer"
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"cannot import TITAN text modules at {model_path}; install einops "
                "and transformers"
            ) from error

        class LocalConchTokenizer:
            """Equivalent tokenizer that never resolves the hard-coded Hub ID."""

            def __init__(self, context_length: int = 128) -> None:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    str(model_path), local_files_only=True
                )
                self.context_length = context_length

            def __call__(self, texts: list[str] | str) -> Any:
                if isinstance(texts, str):
                    texts = [texts]
                tokens = self.tokenizer(
                    texts,
                    max_length=self.context_length - 1,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )["input_ids"]
                return functional.pad(
                    tokens, (0, 1), value=self.tokenizer.pad_token_id
                )

        # build_text_tower resolves this module global at call time.
        text_module.Conch_Tokenizer = LocalConchTokenizer
        titan_config = config_module.TitanConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.text_encoder = text_module.build_text_tower(titan_config.text_config)
        weight_path = model_path / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"TITAN safetensors weights not found: {weight_path}")
        state: dict[str, Any] = {}
        prefix = "text_encoder."
        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
        if not state:
            raise ValueError(f"TITAN weights contain no {prefix} parameters")
        self.text_encoder.load_state_dict(state, strict=True)
        self.text_encoder = self.text_encoder.to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("TITAN cannot encode an empty text list")
        token_ids = self.text_encoder.tokenizer(texts).to(self.device)
        autocast_enabled = self.device.type == "cuda"
        with self._torch.inference_mode(), self._torch.autocast(
            device_type=self.device.type,
            dtype=self._torch.float16,
            enabled=autocast_enabled,
        ):
            # TITAN reserves the last tokenizer position for its appended CLS token.
            features, _ = self.text_encoder(token_ids[:, :-1])
            features = self._torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy()


class LazyTitanTextEncoder:
    """Defer weight loading until a persistent ranking cache actually misses."""

    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self._encoder: LocalTitanTextEncoder | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            self._encoder = LocalTitanTextEncoder(self.model_path, self.device)
        return self._encoder.encode(texts)


class TitanSlideRetriever:
    """Rank a patient's feature-backed slides by maximum query cosine score."""

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self._feature_cache: dict[str, np.ndarray] = {}
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}
        self.metrics = {
            "text_encoder_calls": 0,
            "text_cache_hits": 0,
            "slide_feature_loads": 0,
            "slide_feature_cache_hits": 0,
            "slides_without_titan_feature": 0,
            "slides_without_plip_feature": 0,
            "slides_without_wsi": 0,
        }
        self.last_skipped_slide_ids: list[str] = []

    def _text_features(self, descriptions: list[str]) -> np.ndarray:
        key = tuple(" ".join(item.split()).casefold() for item in descriptions)
        if key in self._text_cache:
            self.metrics["text_cache_hits"] += 1
            return self._text_cache[key]
        features = np.asarray(self.encoder.encode(descriptions), dtype=np.float32)
        if features.ndim != 2 or len(features) != len(descriptions):
            raise ValueError(f"invalid TITAN text feature shape: {features.shape}")
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        self._text_cache[key] = features
        self.metrics["text_encoder_calls"] += 1
        return features

    def _slide_feature(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key in self._feature_cache:
            self.metrics["slide_feature_cache_hits"] += 1
            return self._feature_cache[key]
        value = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError(f"invalid TITAN aligned feature: {path}")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError(f"zero TITAN aligned feature: {path}")
        value = value / norm
        self._feature_cache[key] = value
        self.metrics["slide_feature_loads"] += 1
        return value

    def rank(
        self, slides: list[dict[str, Any]], descriptions: list[str]
    ) -> list[SlideScore]:
        text_features = self._text_features(descriptions)
        eligible: list[tuple[str, np.ndarray]] = []
        skipped: list[str] = []
        for slide in slides:
            feature_path = slide.get("titan_aligned_feature")
            missing_titan = not feature_path or not Path(str(feature_path)).is_file()
            missing_plip = not Path(str(slide.get("plip_feature_cache") or "")).is_file()
            missing_wsi = not Path(str(slide.get("raw_slide_path") or "")).is_file()
            if missing_titan or missing_plip or missing_wsi:
                skipped.append(str(slide["slide_id"]))
                self.metrics["slides_without_titan_feature"] += int(missing_titan)
                self.metrics["slides_without_plip_feature"] += int(missing_plip)
                self.metrics["slides_without_wsi"] += int(missing_wsi)
            # PLIP/WSI availability controls later ROI analysis, not TITAN
            # scoring. Preserve a score for every slide with an aligned vector.
            if missing_titan:
                continue
            feature = self._slide_feature(Path(str(feature_path)))
            if feature.shape[0] != text_features.shape[1]:
                raise ValueError(
                    f"TITAN dimension mismatch for {slide['slide_id']}: "
                    f"slide={feature.shape[0]}, text={text_features.shape[1]}"
                )
            eligible.append((str(slide["slide_id"]), feature))
        self.last_skipped_slide_ids = skipped
        if not eligible:
            raise ValueError("patient has no slide with an accessible TITAN aligned feature")
        matrix = np.stack([item[1] for item in eligible], axis=0)
        per_query = matrix @ text_features.T
        aggregate = per_query.max(axis=1)
        # Stable tie-breaking by slide_id makes retries reproducible.
        order = sorted(
            range(len(eligible)),
            key=lambda index: (-float(aggregate[index]), eligible[index][0]),
        )
        output: list[SlideScore] = []
        for rank, index in enumerate(order, 1):
            scores = per_query[index]
            query_index = int(np.argmax(scores))
            output.append(
                SlideScore(
                    slide_id=eligible[index][0],
                    rank=rank,
                    score=float(aggregate[index]),
                    matched_description=descriptions[query_index],
                    per_description_scores=[float(value) for value in scores],
                )
            )
        return output
