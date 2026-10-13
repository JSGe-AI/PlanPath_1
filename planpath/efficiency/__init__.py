"""PlanPath r26 efficiency extension."""

from planpath.evidence_fusion.components import patch_r23_5_component_hooks
from planpath.evidence_fusion.quantification import R25Quantifier

from .components import (
    R26HistoPLUSAnalyzer,
    R26ROISelector,
    RouteBudget,
    route_budget,
)
from .workflow import METHOD, PlanPathEfficientFusion


# The quantitative algorithm is intentionally unchanged; the alias makes the
# r26 runner's retained branch explicit without duplicating its implementation.
R26Quantifier = R25Quantifier

__all__ = [
    "METHOD",
    "PlanPathEfficientFusion",
    "R26HistoPLUSAnalyzer",
    "R26Quantifier",
    "R26ROISelector",
    "RouteBudget",
    "patch_r23_5_component_hooks",
    "route_budget",
]

