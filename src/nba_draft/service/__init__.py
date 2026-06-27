"""Service layer (Phase 11): the logic the API and dashboard both call.

A :class:`DraftBoardService` bundles a trained impact model, an uncertainty ensemble, the
fold preprocessor, and (lazily) a SHAP explainer, exposing the operations a GM needs:
rank prospects with projections + intervals + scenario distribution, explain a prospect, and
score a prospect's fit with a specific roster. Kept framework-free so it is unit-tested directly.
"""

from nba_draft.service.board import (
    CounterfactualChange,
    CounterfactualResult,
    DraftBoardService,
    build_demo_service,
    build_service_from_master,
    build_service_from_table,
    prospect_to_player,
)

__all__ = [
    "CounterfactualChange",
    "CounterfactualResult",
    "DraftBoardService",
    "build_demo_service",
    "build_service_from_master",
    "build_service_from_table",
    "prospect_to_player",
]
