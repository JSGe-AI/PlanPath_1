"""Drop-in r25 adapters for r23_5 ROI selection and HistoPLUS analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from planpath.core.cell_store import PrecomputedCellStore
from planpath.core.histoplus import HistoPLUSAnalyzer
from planpath.core.image_io import field_size_level0, read_field
from planpath.core.plip import ROISelector, SelectedROI
from planpath.core.schemas import (
    CellularAnalysis,
    EvidencePlan,
    TissueNavigation,
)

from .routing import (
    component_route,
    histoplus_applicable,
    routing_policy,
    tissue_navigation_applicable,
)
from .context import current_question, register_roi_stratum


@dataclass(frozen=True)
class CellularHotspot:
    """A HistoPLUS focus proposal; never a diagnostic conclusion."""

    center_level0: tuple[int, int]
    score: float
    route: str
    tissue_qualified: bool
    tissue_stratum: str | None = None


class R25ROISelector(ROISelector):
    """Use Tissue Map as a recall tranche while preserving PLIP anchors."""

    algorithm_version = "r25-stratified-tissue-recall-v2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._histoplus_hotspots: dict[str, CellularHotspot] = {}
        self._active_low_power_strata: dict[str, np.ndarray] = {}
        self._active_low_power_stratum_centers: dict[
            str, dict[tuple[int, int], tuple[int, int]]
        ] = {}
        self.metrics["r25_histoplus_hotspot_recenters"] = 0
        self.metrics["r25_joint_tissue_histoplus_recenters"] = 0
        self.metrics["r25_equalized_low_power_budget_uses"] = 0
        self.metrics["r25_equalized_high_power_budget_uses"] = 0
        self.metrics["r25_tissue_stratified_selections"] = 0
        self.metrics["r25_tissue_stratum_descendants"] = 0
        self.metrics["r25_tissue_stratum_recenters"] = 0

    def attach_histoplus_hotspots(
        self, hotspots: dict[str, CellularHotspot]
    ) -> None:
        """Share the analyzer's per-parent focus cache without global state."""

        self._histoplus_hotspots = hotspots

    @staticmethod
    def _low_power_budget(route: str) -> int:
        if route in {
            "renal_boundary",
            "vascular_interface",
            "neural_interface",
            "nodal",
            "depth_interface",
        }:
            return 12
        if route in {
            "margin_interface",
            "collecting_system_interface",
            "cellular_extremum",
            "tissue_composition",
            "tumor_bed_response",
        }:
            return 8
        if route == "tumor_phenotype":
            domain = str((current_question() or {}).get("clinical_domain") or "")
            return 8 if "gastric" in domain.casefold() else 5
        if route == "tumor_distribution":
            return 8
        return 5

    @staticmethod
    def _high_power_budget(route: str, magnification: str) -> int:
        if route in {
            "renal_boundary",
            "vascular_interface",
            "neural_interface",
            "nodal",
            "depth_interface",
        }:
            return 8 if magnification == "40x" else 16
        if route in {
            "margin_interface",
            "collecting_system_interface",
            "cellular_extremum",
            "tissue_composition",
            "tumor_bed_response",
        }:
            return 8
        if route == "tumor_phenotype":
            domain = str((current_question() or {}).get("clinical_domain") or "")
            return 8 if "gastric" in domain.casefold() else 5
        if route == "tumor_distribution":
            return 8
        return 5

    @staticmethod
    def _stratum_labels(cancer: str, route: str) -> tuple[str, ...]:
        """Return independent navigation hypotheses for heterogeneous tasks."""

        if route == "tumor_phenotype" and cancer == "gastric":
            return (
                "diffuse_or_poorly_cohesive_tumor",
                "gland_forming_tumor",
                "mucin_pool",
            )
        if route == "tumor_phenotype" and cancer == "renal":
            # The renal map has one broad tumour class and cannot separate
            # renal epithelial subtypes. Keep the validated PLIP budget and
            # use HistoPLUS only to prioritize a representative tumour field.
            return ()
        if route == "tumor_bed_response":
            return (
                "viable_renal_tumor" if cancer == "renal" else "gland_forming_tumor",
                "interstitium_or_fibrosis"
                if cancer == "renal"
                else "desmoplastic_stroma_or_tumor_bed",
                "immune_or_lymphoid",
            )
        if route == "tissue_composition":
            return (
                "viable_renal_tumor" if cancer == "renal" else "gland_forming_tumor",
                "necrosis_or_debris"
                if cancer == "renal"
                else "necrosis_or_ulcer_debris",
            )
        if route == "nodal":
            return (
                "immune_or_lymphoid",
                "viable_renal_tumor"
                if cancer == "renal"
                else "diffuse_or_poorly_cohesive_tumor",
                "gland_forming_tumor" if cancer == "gastric" else "",
            )
        if route == "tumor_distribution":
            return (
                "viable_renal_tumor"
                if cancer == "renal"
                else "gland_forming_tumor",
            )
        return ()

    @staticmethod
    def _navigation_stratum(navigation: TissueNavigation | None) -> str | None:
        if navigation is None:
            return None
        route = str(navigation.selection_route or "")
        for prefix in (
            "r25_tissue_stratum:",
            "r25_tissue_stratum_descendant:",
        ):
            if route.startswith(prefix):
                return route.removeprefix(prefix)
        return None

    @staticmethod
    def _stratum_scores_and_centers(
        tissue_map: Any,
        coords: np.ndarray,
        label: str,
    ) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
        """Score focal map classes without diluting them over a 4096px box."""

        label_id = tissue_map.label_id(label)
        scores = np.zeros(len(coords), dtype=np.float32)
        centers: dict[tuple[int, int], tuple[int, int]] = {}
        if label_id is None:
            return scores, centers
        for index, (raw_x, raw_y) in enumerate(coords):
            x, y = int(raw_x), int(raw_y)
            indices = tissue_map._overlap_indices((x, y, x + 4096, y + 4096))
            if not len(indices):
                continue
            probabilities = tissue_map.class_probs[indices, label_id]
            known = ~tissue_map.unknown[indices]
            usable = probabilities[known] if bool(known.any()) else probabilities
            if not len(usable):
                continue
            # A focal component should not disappear into a window mean, but
            # an isolated uncertain patch should not completely control the
            # field either. Blend focal recall with regional support.
            scores[index] = 0.7 * float(usable.max()) + 0.3 * float(usable.mean())
            local = int(np.argmax(probabilities * (0.5 + 0.5 * known.astype(np.float32))))
            patch_index = int(indices[local])
            centers[(x, y)] = (
                int(tissue_map.x[patch_index] + tissue_map.patch_size[patch_index] // 2),
                int(tissue_map.y[patch_index] + tissue_map.patch_size[patch_index] // 2),
            )
        return scores, centers

    def select_low(
        self,
        slide: dict[str, Any],
        query: str,
        plan: EvidencePlan | None = None,
    ) -> list[SelectedROI]:
        """Keep the ROI budget identical when Tissue Map is disabled.

        Otherwise a no-Tissue run receives only five PLIP fields while the
        full interface route receives up to twelve fields.  Equalizing the
        budget isolates missing navigation information,
        rather than missing compute.
        """

        budget = self._low_power_budget(component_route()) if plan is not None else 5
        previous = self.low_power_top_k
        self.low_power_top_k = max(previous, budget)
        if self.low_power_top_k != previous:
            self.metrics["r25_equalized_low_power_budget_uses"] += 1
        self._active_low_power_strata = {}
        self._active_low_power_stratum_centers = {}
        tissue_map = None
        if plan is not None and tissue_navigation_applicable():
            tissue_map = self._tissue_map(slide)
            labels = tuple(
                label
                for label in self._stratum_labels(
                    str(slide.get("cancer") or ""), component_route()
                )
                if label
            )
            if tissue_map is not None and labels:
                coords, _ = self._load_low(slide)
                for label in labels:
                    scores, centers = self._stratum_scores_and_centers(
                        tissue_map, coords, label
                    )
                    self._active_low_power_strata[label] = scores
                    self._active_low_power_stratum_centers[label] = centers
        try:
            output = super().select_low(slide, query, plan)
            recenterable = [
                item
                for item in output
                if self._navigation_stratum(item.roi.tissue_navigation) is not None
            ]
            if tissue_map is not None and plan is not None and recenterable:
                policy = routing_policy(str(slide.get("cancer") or ""), plan)
                field_size = field_size_level0(5)
                with self._open_slide(str(slide["raw_slide_path"])) as wsi:
                    for index, item in enumerate(output):
                        stratum = self._navigation_stratum(item.roi.tissue_navigation)
                        if stratum is None:
                            continue
                        x1, y1, _, _ = item.roi.coordinates
                        center = self._active_low_power_stratum_centers.get(
                            stratum, {}
                        ).get((x1, y1))
                        if center is None:
                            continue
                        x = min(
                            max(0, center[0] - field_size // 2),
                            max(0, int(wsi.dimensions[0]) - field_size),
                        )
                        y = min(
                            max(0, center[1] - field_size // 2),
                            max(0, int(wsi.dimensions[1]) - field_size),
                        )
                        coordinates = (x, y, x + field_size, y + field_size)
                        if coordinates == item.roi.coordinates:
                            continue
                        summary = tissue_map.window_summary(
                            coordinates,
                            policy.target_labels,
                            policy.interface_pairs,
                        )
                        old_navigation = item.roi.tissue_navigation
                        assert old_navigation is not None
                        navigation = old_navigation.model_copy(
                            update={
                                "target_score": summary["target_score"],
                                "interface_score": summary["interface_score"],
                                "heterogeneity": summary["heterogeneity"],
                                "map_confidence": summary["map_confidence"],
                                "unknown_fraction": summary.get(
                                    "unknown_fraction", 0.0
                                ),
                                "class_fractions": summary["class_fractions"],
                            }
                        )
                        output[index] = SelectedROI(
                            item.roi.model_copy(
                                update={
                                    "coordinates": coordinates,
                                    "tissue_navigation": navigation,
                                }
                            ),
                            read_field(wsi, x, y, 5),
                        )
                        self.metrics["r25_tissue_stratum_recenters"] += 1
            for item in output:
                stratum = self._navigation_stratum(item.roi.tissue_navigation)
                if stratum is not None:
                    register_roi_stratum(item.roi.roi_id, stratum)
            return output
        finally:
            self.low_power_top_k = previous
            self._active_low_power_strata = {}
            self._active_low_power_stratum_centers = {}

    def _tissue_map(self, slide: dict[str, Any]):  # type: ignore[override]
        if not tissue_navigation_applicable():
            self.metrics["tissue_map_missing"] += 1
            return None
        return super()._tissue_map(slide)

    def _hybrid_indices(
        self,
        coords: np.ndarray,
        plip_scores: np.ndarray,
        fusion_scores: np.ndarray,
        tissue_scores: np.ndarray,
        interface_scores: np.ndarray,
        heterogeneity_scores: np.ndarray,
        count: int,
        **kwargs: Any,
    ) -> tuple[list[int], dict[int, str]]:
        target = min(count, len(coords))
        # The strongest semantic fields remain invariant between full and
        # no-Tissue runs. Tissue Map spends only the remaining tranche on a
        # complementary target/interface hypothesis.
        semantic_count = min(target, max(2, (target + 1) // 2))
        semantic = self._diverse_indices(coords, plip_scores, semantic_count)
        routed, routed_labels = super()._hybrid_indices(
            coords,
            plip_scores,
            fusion_scores,
            tissue_scores,
            interface_scores,
            heterogeneity_scores,
            target,
            **kwargs,
        )
        selected = list(semantic)
        routes = {index: "r25_plip_anchor" for index in semantic}

        # Reserve one field per independently mapped phenotype/composition
        # hypothesis before allowing the aggregate tissue score to fill the
        # portfolio.  Without this step, the common gland-forming class can
        # consume every Tissue Map slot and hide a small diffuse or mucinous
        # component. A minimum absolute soft score avoids reserving a field
        # for numerical noise when that class is genuinely absent.
        for label, values in getattr(self, "_active_low_power_strata", {}).items():
            if len(values) != len(coords) or float(values.max(initial=0.0)) < 0.02:
                continue
            for index in self._diverse_indices(coords, values, len(coords)):
                if index in selected:
                    continue
                selected.append(index)
                routes[index] = f"r25_tissue_stratum:{label}"
                if hasattr(self, "metrics"):
                    self.metrics["r25_tissue_stratified_selections"] += 1
                break
            if len(selected) == target:
                break
        for index in routed:
            if index in selected:
                continue
            selected.append(index)
            routes[index] = routed_labels.get(index, "r25_tissue_recall")
            if len(selected) == target:
                break
        if len(selected) < target:
            for index in self._diverse_indices(coords, fusion_scores, target):
                if index not in selected:
                    selected.append(index)
                    routes[index] = "r25_fusion_fill"
                if len(selected) == target:
                    break
        return selected, routes

    def select_high(
        self,
        slide: dict[str, Any],
        parents: list[Any],
        magnification: Literal["10x", "20x", "40x"],
        query: str,
        *,
        ensure_parent_diversity: bool = True,
        plan: EvidencePlan | None = None,
    ) -> list[SelectedROI]:
        """Recenter a bounded 40x tranche on measured cellular hotspots.

        HistoPLUS is run on the preceding localization level.  Its instance
        coordinates nominate at most two child fields; Patho-R1 still has to
        transcribe the pixels in the recentered field before the controller
        may use them.  All remaining children retain the ordinary PLIP/Tissue
        selection, preventing a noisy cell classifier from taking over the
        search.
        """

        budget = (
            self._high_power_budget(component_route(), magnification)
            if plan is not None
            else self.high_power_top_k
        )
        previous = self.high_power_top_k
        self.high_power_top_k = max(previous, budget)
        if self.high_power_top_k != previous:
            self.metrics["r25_equalized_high_power_budget_uses"] += 1
        try:
            output = super().select_high(
                slide,
                parents,
                magnification,
                query,
                ensure_parent_diversity=ensure_parent_diversity,
                plan=plan,
            )
        finally:
            self.high_power_top_k = previous

        # Keep the low-power stratum identity through both zoom transitions.
        # It is used only to balance HistoPLUS screening and never reaches the
        # diagnostic answer as a class label.
        parent_by_id = {str(item.roi_id): item for item in parents}
        for index, selected in enumerate(output):
            parent = parent_by_id.get(str(selected.roi.parent_roi_id or ""))
            stratum = self._navigation_stratum(
                parent.tissue_navigation if parent is not None else None
            )
            navigation = selected.roi.tissue_navigation
            if stratum is None or navigation is None:
                continue
            output[index] = SelectedROI(
                selected.roi.model_copy(
                    update={
                        "tissue_navigation": navigation.model_copy(
                            update={
                                "selection_route": (
                                    f"r25_tissue_stratum_descendant:{stratum}"
                                )
                            }
                        )
                    }
                ),
                selected.image,
            )
            register_roi_stratum(output[index].roi.roi_id, stratum)
            self.metrics["r25_tissue_stratum_descendants"] += 1
        if magnification != "40x" or not output or not self._histoplus_hotspots:
            return output

        route = component_route()
        candidates: list[tuple[float, int, CellularHotspot]] = []
        represented_parents: set[str] = set()
        # Replace the lowest-ranked redundant child for a parent. The best
        # ordinary PLIP/Tissue crop remains unchanged, so HistoPLUS adds a
        # complementary focus instead of deleting the strongest baseline
        # field.
        for index in reversed(range(len(output))):
            item = output[index]
            parent_id = str(item.roi.parent_roi_id or "")
            if parent_id in represented_parents:
                continue
            hotspot = self._histoplus_hotspots.get(parent_id)
            if hotspot is None or hotspot.route != route:
                continue
            parent = next(
                (candidate for candidate in parents if candidate.roi_id == parent_id),
                None,
            )
            if parent is None:
                continue
            cx, cy = hotspot.center_level0
            x1, y1, x2, y2 = parent.coordinates
            if not (x1 <= cx < x2 and y1 <= cy < y2):
                continue
            # A Tissue-qualified hotspot wins a tie, while the measured
            # cellular score remains the primary ranking signal.
            candidates.append(
                (hotspot.score + (0.01 if hotspot.tissue_qualified else 0.0), index, hotspot)
            )
            represented_parents.add(parent_id)
        if not candidates:
            return output
        candidates.sort(reverse=True, key=lambda item: item[0])
        limit = min(2, max(1, len(output) // 4))
        chosen: list[tuple[float, int, CellularHotspot]] = []
        if route == "tumor_phenotype":
            used_strata: set[str] = set()
            for candidate in candidates:
                stratum = candidate[2].tissue_stratum
                if stratum is None or stratum in used_strata:
                    continue
                chosen.append(candidate)
                used_strata.add(stratum)
                if len(chosen) == limit:
                    break
        for candidate in candidates:
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == limit:
                break
        target_size = field_size_level0(40)
        tissue_map = self._tissue_map(slide) if plan is not None else None
        policy = (
            routing_policy(str(slide.get("cancer") or ""), plan)
            if plan is not None
            else None
        )
        with self._open_slide(str(slide["raw_slide_path"])) as wsi:
            for _, index, hotspot in chosen:
                old = output[index]
                parent = parent_by_id[str(old.roi.parent_roi_id)]
                px1, py1, px2, py2 = parent.coordinates
                cx, cy = hotspot.center_level0
                x = min(max(cx - target_size // 2, px1), px2 - target_size)
                y = min(max(cy - target_size // 2, py1), py2 - target_size)
                x, y = max(0, int(x)), max(0, int(y))
                coordinates = (x, y, x + target_size, y + target_size)
                if coordinates == old.roi.coordinates:
                    continue
                navigation = None
                if tissue_map is not None and policy is not None:
                    summary = tissue_map.window_summary(
                        coordinates,
                        policy.target_labels,
                        policy.interface_pairs,
                    )
                    old_navigation = old.roi.tissue_navigation
                    navigation = TissueNavigation(
                        map_available=True,
                        target_score=summary["target_score"],
                        interface_score=summary["interface_score"],
                        heterogeneity=summary["heterogeneity"],
                        map_confidence=summary["map_confidence"],
                        unknown_fraction=summary.get("unknown_fraction", 0.0),
                        fusion_score=(
                            old_navigation.fusion_score
                            if old_navigation is not None
                            else 0.0
                        ),
                        selection_route="r25_joint_tissue_histoplus_hotspot",
                        target_labels=list(policy.target_labels),
                        class_fractions=summary["class_fractions"],
                    )
                new_roi = old.roi.model_copy(
                    update={
                        "coordinates": coordinates,
                        "tissue_navigation": navigation,
                    }
                )
                output[index] = SelectedROI(new_roi, read_field(wsi, x, y, 40))
                self.metrics["r25_histoplus_hotspot_recenters"] += 1
                if hotspot.tissue_qualified:
                    self.metrics["r25_joint_tissue_histoplus_recenters"] += 1
        return output


class R25HistoPLUSAnalyzer(HistoPLUSAnalyzer):
    """Expose cellular measurements only for a supported clinical endpoint.

    When Tissue Navigation is present, HistoPLUS fields must also lie in a
    compatible mapped context. Histo-only operation retains its independent
    measurements instead of forcing an artificial refusal.
    """

    algorithm_version = "r25-stratum-balanced-histoplus-v2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hotspots: dict[str, CellularHotspot] = {}

    @staticmethod
    def _grade_screening_proxy(selected: SelectedROI) -> SelectedROI:
        """Run the 20x cellular model inside a true-10x grade localization."""

        x1, y1, x2, y2 = selected.roi.coordinates
        size = field_size_level0(20)
        x = (x1 + x2 - size) // 2
        y = (y1 + y2 - size) // 2
        roi = selected.roi.model_copy(
            update={
                "coordinates": (x, y, x + size, y + size),
                "magnification": "20x",
            }
        )
        return SelectedROI(roi, selected.image)

    def _instances(
        self,
        selected: SelectedROI,
        analysis: CellularAnalysis,
    ) -> list[dict[str, Any]]:
        raw = str(analysis.annotation_path or "")
        if not raw:
            return []
        path = Path(raw)
        try:
            if path.name.endswith(".json.gz"):
                payload = self._read_patch_cache(path)
                return list(payload.get("instances") or [])
            if path.suffix == ".h5":
                query = PrecomputedCellStore(path).query(
                    selected.roi.coordinates,
                    expected_model_signature=analysis.model_signature,
                )
                return list(query.instances) if query is not None else []
        except (OSError, ValueError, KeyError, TypeError):
            return []
        return []

    @staticmethod
    def _hotspot_from_instances(
        instances: list[dict[str, Any]],
        *,
        route: str,
        tissue_qualified: bool,
        tissue_stratum: str | None = None,
    ) -> CellularHotspot | None:
        """Find a dense route-specific cellular neighborhood."""

        weights_by_route = {
            "cellular_extremum": {
                "Cancer cell": 3.0,
                "Epithelial": 1.0,
                "Mitotic Figures": 8.0,
                "Apoptotic Body": 1.0,
            },
            "nodal": {
                "Cancer cell": 5.0,
                "Epithelial": 2.0,
                "Lymphocytes": 0.35,
                "Plasmocytes": 0.25,
            },
            "vascular_interface": {
                "Cancer cell": 4.0,
                "Epithelial": 1.5,
                "Endothelial Cell": 3.0,
            },
            "neural_interface": {"Cancer cell": 4.0, "Epithelial": 1.5},
            "renal_boundary": {
                "Cancer cell": 4.0,
                "Epithelial": 1.5,
                "Fibroblasts": 0.5,
            },
            "margin_interface": {
                "Cancer cell": 4.0,
                "Epithelial": 1.5,
                "Fibroblasts": 0.5,
            },
            "depth_interface": {
                "Cancer cell": 4.0,
                "Epithelial": 1.5,
                "Fibroblasts": 0.5,
                "Muscle Cell": 0.5,
            },
            "tumor_phenotype": {"Cancer cell": 4.0, "Epithelial": 2.0},
            "tumor_bed_response": {
                "Cancer cell": 5.0,
                "Epithelial": 2.0,
                "Fibroblasts": 0.5,
                "Lymphocytes": 0.25,
                "Macrophages": 0.25,
            },
            "immune_context": {
                "Lymphocytes": 2.0,
                "Plasmocytes": 2.0,
                "Eosinophils": 2.0,
                "Neutrophils": 2.0,
                "Macrophages": 2.0,
            },
        }
        weights = weights_by_route.get(route)
        if weights is None:
            return None
        accepted = [
            item
            for item in instances
            if float(item.get("confidence", 0.0)) >= 0.5
            and str(item.get("cell_type") or "") in weights
            and len(item.get("centroid_level0") or []) == 2
        ]
        primary_names = (
            {"Cancer cell", "Epithelial"}
            if route != "immune_context"
            else set(weights)
        )
        candidates = [
            item for item in accepted if str(item.get("cell_type")) in primary_names
        ]
        if not candidates:
            return None
        coordinates = np.asarray(
            [item["centroid_level0"] for item in accepted], dtype=np.float32
        )
        cell_weights = np.asarray(
            [weights[str(item["cell_type"])] for item in accepted], dtype=np.float32
        )
        areas = np.asarray(
            [float(item.get("area_um2", 0.0)) for item in accepted], dtype=np.float32
        )
        best: tuple[float, tuple[int, int]] | None = None
        half_field = field_size_level0(40) / 2
        for item in candidates:
            center = np.asarray(item["centroid_level0"], dtype=np.float32)
            inside = np.max(np.abs(coordinates - center), axis=1) <= half_field
            score = float(cell_weights[inside].sum())
            if route == "cellular_extremum" and bool(inside.any()):
                # Nuclear size is used only to choose a field for Patho-R1;
                # it is never translated into a grade.
                score += float(np.quantile(areas[inside], 0.95)) / 25.0
            value = (int(round(float(center[0]))), int(round(float(center[1]))))
            if best is None or score > best[0]:
                best = (score, value)
        if best is None:
            return None
        return CellularHotspot(
            center_level0=best[1],
            score=best[0],
            route=route,
            tissue_qualified=tissue_qualified,
            tissue_stratum=tissue_stratum,
        )

    @staticmethod
    def _navigation_stratum(selected: SelectedROI) -> str | None:
        return R25ROISelector._navigation_stratum(
            selected.roi.tissue_navigation
        )

    def _balanced_analysis_selection(
        self,
        selected: list[SelectedROI],
        *,
        limit: int,
    ) -> list[SelectedROI]:
        """Keep one measurable field per Tissue Map hypothesis when possible."""

        ranked = sorted(
            (item for item in selected if self._context_supported(item)),
            key=lambda item: (
                item.roi.tissue_navigation.fusion_score
                if item.roi.tissue_navigation
                else 0.0,
                item.roi.retrieval_score,
            ),
            reverse=True,
        )
        if component_route() not in {
            "tumor_phenotype",
            "tumor_bed_response",
            "tissue_composition",
            "nodal",
        }:
            return ranked[:limit]
        output: list[SelectedROI] = []
        used: set[str] = set()
        for item in ranked:
            stratum = self._navigation_stratum(item)
            if stratum is None or stratum in used:
                continue
            output.append(item)
            used.add(stratum)
            if len(output) == limit:
                return output
        for item in ranked:
            if item not in output:
                output.append(item)
            if len(output) == limit:
                break
        return output

    @staticmethod
    def _context_supported(selected: Any) -> bool:
        navigation = selected.roi.tissue_navigation
        if navigation is None:
            return True
        fractions = navigation.class_fractions
        stratum = R25ROISelector._navigation_stratum(navigation)
        if stratum is not None:
            return (
                float(fractions.get(stratum, 0.0)) >= 0.01
                and navigation.unknown_fraction <= 0.60
            )
        target_fraction = sum(
            float(fractions.get(label, 0.0)) for label in navigation.target_labels
        )
        route = component_route()
        if route == "nodal":
            return (
                float(fractions.get("immune_or_lymphoid", 0.0)) >= 0.01
                or navigation.interface_score > 0.0
            )
        if route.endswith("_interface") or route == "renal_boundary":
            return navigation.interface_score > 0.0 or target_fraction >= 0.02
        return target_fraction >= 0.02 and navigation.unknown_fraction <= 0.60

    def analyze_selected(
        self,
        slide: dict[str, Any],
        selected: list[Any],
        plan: EvidencePlan,
    ) -> dict[str, CellularAnalysis]:
        for item in selected:
            self.hotspots.pop(item.roi.roi_id, None)
        if not self.config.histoplus_enabled or not selected:
            self.skipped += len(selected)
            return {}
        if not histoplus_applicable():
            self.skipped += len(selected)
            self.audit.append(
                {
                    "validation": "r25_question_route_skipped",
                    "route": component_route(),
                    "roi_count": len(selected),
                }
            )
            return {}
        policy = routing_policy(str(slide.get("cancer") or ""), plan)
        if policy.histoplus_policy == "off":
            self.skipped += len(selected)
            return {}
        limit = (
            self.config.histoplus_required_top_k
            if policy.histoplus_policy == "required"
            else self.config.histoplus_conditional_top_k
        )
        analysis_selected = self._balanced_analysis_selection(
            selected,
            limit=limit,
        )
        self.skipped += max(0, len(selected) - len(analysis_selected))
        grade_screening = bool(
            component_route() == "cellular_extremum"
            and analysis_selected
            and all(item.roi.magnification == "10x" for item in analysis_selected)
        )
        if grade_screening:
            analysis_selected = [
                self._grade_screening_proxy(item) for item in analysis_selected
            ]
        measured = self.analyze_batch(slide, analysis_selected)
        selected_by_id = {item.roi.roi_id: item for item in selected}
        analysis_selected_by_id = {
            item.roi.roi_id: item for item in analysis_selected
        }
        output: dict[str, CellularAnalysis] = {}
        for roi_id, analysis in measured.items():
            source = selected_by_id.get(roi_id)
            if source is None or not self._context_supported(source):
                self.audit.append(
                    {
                        "validation": "r25_tissue_context_rejected",
                        "route": component_route(),
                        "roi_id": roi_id,
                    }
                )
                continue
            limitations = list(analysis.limitations)
            if source.roi.tissue_navigation is not None:
                limitations.append(
                    "r25: cellular measurement passed the independent Tissue Map context check."
                )
            if grade_screening:
                limitations.append(
                    "r25: 20x cellular screening nominated a 40x focus inside this true-10x field."
                )
            output[roi_id] = analysis.model_copy(
                update={"limitations": limitations[:4]}
            )
            analysis_source = analysis_selected_by_id.get(roi_id)
            if analysis_source is not None and analysis.quality == "adequate":
                hotspot = self._hotspot_from_instances(
                    self._instances(analysis_source, analysis),
                    route=component_route(),
                    tissue_qualified=source.roi.tissue_navigation is not None,
                    tissue_stratum=self._navigation_stratum(source),
                )
                if hotspot is not None:
                    self.hotspots[roi_id] = hotspot
        return output


def patch_r23_5_component_hooks() -> None:
    """Install r25 policies in one r25 runner process only."""

    import planpath.core.controller as controller_module
    import planpath.core.histoplus as histoplus_module
    import planpath.core.observer as observer_module
    import planpath.core.plip as plip_module
    import planpath.core.workflow as workflow_module

    from .evidence import (
        canonical_answer_format,
        gastric_invasion_layer,
        question_specific_focus_roi_ids,
        question_specific_preference,
        renal_grade_bounds,
    )
    from .prompts import final_messages, review_messages

    plip_module.routing_policy = routing_policy
    observer_module.routing_policy = routing_policy
    histoplus_module.routing_policy = routing_policy
    # workflow.py imports the policy into its own module namespace.  Patch that
    # binding as well so the recorded modality route and the candidate-pool
    # protection use the same question-anchored decision as the two tools.
    workflow_module.routing_policy = routing_policy
    # The controller imports prompt functions and grade aggregation into its
    # own namespace.  r25 exposes bounded cross-scale focus support at both
    # review and final integration, and fixes the old interpretation where a
    # phrase such as "lacks extreme pleomorphism" could itself trigger Grade 4.
    controller_module.review_messages = review_messages
    controller_module.final_messages = final_messages
    controller_module._renal_grade_bounds = renal_grade_bounds
    controller_module._gastric_invasion_layer = gastric_invasion_layer
    controller_module._canonical_answer_format = canonical_answer_format
    controller_module._question_specific_preference = question_specific_preference
    controller_module.question_specific_focus_roi_ids = (
        question_specific_focus_roi_ids
    )
    workflow_module.question_specific_focus_roi_ids = question_specific_focus_roi_ids
