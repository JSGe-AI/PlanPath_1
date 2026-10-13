#!/usr/bin/env python3
"""Explicitly download the gated official HistoPLUS checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "histoplus",
    )
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HistoPLUS weights are gated. Request access at "
            "https://huggingface.co/Owkin-Bioptimus/histoplus and rerun with "
            "HF_TOKEN set in the environment. The token is never written to disk."
        )
    args.model_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "histoplus_cellvit_segmentor_20x.pt",
        "histoplus_cellvit_segmentor_40x.pt",
    ):
        path = hf_hub_download(
            repo_id="Owkin-Bioptimus/histoplus",
            filename=filename,
            token=token,
            cache_dir=str(args.model_root / "hf_cache"),
        )
        print(f"downloaded {filename}: {path}", flush=True)


if __name__ == "__main__":
    main()
