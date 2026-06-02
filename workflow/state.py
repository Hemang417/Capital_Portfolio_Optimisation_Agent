"""
workflow/state.py — Shared TypedDict passed between every LangGraph node.

All keys are Optional except user_intent, budget_gbp, and mc_iterations which
are required in the initial state.  Each node reads only the keys it needs and
writes only the keys it produces — state is merged automatically by LangGraph.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class WorkflowState(TypedDict):
    # ── Required initial inputs ───────────────────────────────────────────────
    user_intent:    str    # "run_all"  or  natural-language query string
    budget_gbp:     float  # capital budget in GBP  (e.g. 45_000_000.0)
    mc_iterations:  int    # Monte Carlo iteration count  (e.g. 10_000)

    # ── Written by data_validation_node ──────────────────────────────────────
    validated_assets:   Optional[list]   # List[Asset] that passed validation
    validation_errors:  Optional[list]   # List[str] human-readable error messages

    # ── Written by scenario_retrieval_node ───────────────────────────────────
    # Single Scenario for NL-query mode; List[Scenario] for run_all mode.
    # Typed as Any to accommodate both at runtime — TypedDict is a hint only.
    scenario:        Optional[Any]   # Scenario | List[Scenario]
    rag_confidence:  Optional[float] # AgenticRAG confidence score (None for run_all)

    # ── Written by financial_engine_node ─────────────────────────────────────
    # Single ScenarioRunResult for NL-query; List[ScenarioRunResult] for run_all.
    engine_result:   Optional[Any]   # ScenarioRunResult | List[ScenarioRunResult]

    # ── Written by commentary_node (parallel with risk_flag_node) ────────────
    commentary:      Optional[dict]  # {"chart_explanations": {...}, "suggestions": [...]}

    # ── Written by risk_flag_node (parallel with commentary_node) ────────────
    risk_flags:      Optional[list]  # List[{"severity", "description", "affected_metric"}]

    # ── Written by human_approval_node ───────────────────────────────────────
    human_approved:  Optional[bool]  # True → proceed to report generation

    # ── Written by report_generation_node ────────────────────────────────────
    report_paths:    Optional[dict]  # {"json": path, "csv": path, "pdf": path}
