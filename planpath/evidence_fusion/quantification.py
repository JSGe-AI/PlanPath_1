"""r25 quantitative adapter with explicit component switches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from planpath.measurement.quantification import (
    R24Quantifier,
    _load_histo_fields,
    _read_tissue_map,
)


class R25Quantifier(R24Quantifier):
    """Retain r24 measurement logic while honoring component switches.

    Disabled tools are not loaded and cannot contribute to a measurement.
    The remaining tool still runs, so component-disabled configurations fail
    only the actual coverage or specificity checks.
    """

    def __init__(
        self,
        cell_seg_root: Path,
        *,
        tissue_navigation_enabled: bool,
        histoplus_enabled: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(cell_seg_root, **kwargs)
        self.tissue_navigation_enabled = bool(tissue_navigation_enabled)
        self.histoplus_enabled = bool(histoplus_enabled)

    def _base_measurements(self, slides):  # type: ignore[override]
        tissue = (
            [item for slide in slides if (item := _read_tissue_map(slide)) is not None]
            if self.tissue_navigation_enabled
            else []
        )
        cancer = str(slides[0].get("cancer") or "unknown") if slides else "unknown"
        patient_id = (
            str(slides[0].get("patient_id") or "unknown") if slides else "unknown"
        )
        slide_ids = {str(slide["slide_id"]) for slide in slides}
        histo = (
            _load_histo_fields(self.cell_seg_root, cancer, patient_id, slide_ids)
            if self.histoplus_enabled
            else []
        )
        adequate = [item for item in histo if item.quality == "adequate"]
        map_slide_coverage = len(tissue) / max(len(slides), 1)
        map_known_area_coverage = (
            sum(item.known_area_px2 for item in tissue)
            / max(sum(item.total_area_px2 for item in tissue), 1e-12)
            if tissue
            else 0.0
        )
        summary = {
            "map_slide_count": len(tissue),
            "expected_slide_count": len(slides),
            "map_slide_coverage": map_slide_coverage,
            "map_known_area_coverage": map_known_area_coverage,
            "histoplus_adequate_roi_count": len(adequate),
            "histoplus_valid_area_mm2": sum(item.valid_area_mm2 for item in adequate),
            "histoplus_tumor_like_cell_count": sum(
                item.tumor_like_count for item in adequate
            ),
            "histoplus_analyzed_slide_count": len(
                {item.slide_id for item in adequate}
            ),
            "tissue_component_enabled": self.tissue_navigation_enabled,
            "histoplus_component_enabled": self.histoplus_enabled,
        }
        summary["tissue_coverage_fraction"] = (
            map_slide_coverage * map_known_area_coverage
        )
        return tissue, adequate, summary
