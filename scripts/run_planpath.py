#!/usr/bin/env python3
"""Run the final PlanPath workflow."""

from __future__ import annotations

import copy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_backend as base_runner  # noqa: E402
import quantitative_backend  # noqa: E402
from planpath.measurement.controller import (  # noqa: E402
    R24CoreController as MeasurementController,
)
from planpath.model import (  # noqa: E402
    PlanPath,
    PlanPathHistoPLUSAnalyzer,
    PlanPathQuantifier,
    PlanPathQuantitativeOverlay,
    PlanPathROISelector,
    patch_final_component_hooks,
)


_build_base_agent = base_runner.build_agent


def build_final_agent(args):
    """Build one PlanPath agent with automatic quantitative dispatch."""

    patch_final_component_hooks()
    base_runner.ROISelector = PlanPathROISelector
    base_runner.HistoPLUSAnalyzer = PlanPathHistoPLUSAnalyzer
    base = _build_base_agent(args)
    selector = base.visual.selector
    histoplus = base.visual.histoplus
    if isinstance(selector, PlanPathROISelector) and isinstance(
        histoplus,
        PlanPathHistoPLUSAnalyzer,
    ):
        selector.attach_histoplus_hotspots(histoplus.hotspots)
    quantifier = PlanPathQuantifier(
        args.cell_seg_root,
        tissue_navigation_enabled=base.config.tissue_navigation_enabled,
        histoplus_enabled=base.config.histoplus_enabled,
    )
    main_agent = PlanPath(base, MeasurementController(), quantifier)

    # The quantitative backend uses the same command-line/configuration inputs
    # as the main workflow. Reuse the already loaded local retrieval encoders
    # and skip the duplicate endpoint health check.
    quantitative_args = copy.copy(args)
    quantitative_args.skip_health_check = True
    quantitative_agent = quantitative_backend.build_agent(
        quantitative_args,
        titan_encoder=base.retriever.encoder,
        plip_encoder=base.visual.selector.encoder,
    )
    return PlanPathQuantitativeOverlay(main_agent, quantitative_agent)


def main() -> None:
    base_runner.build_agent = build_final_agent
    base_runner.main()


if __name__ == "__main__":
    main()
