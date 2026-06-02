"""
workflow/nodes.py — One function per LangGraph node.

CONSTRAINTS
-----------
* No file inside engine/, retrieval/, llm/, data/, or reporting/ is modified.
  Every node wraps the existing modules exactly as they are.
* All LLM calls route through llm/claude_client.ClaudeClient — never direct
  groq/anthropic imports in this file.
* Each function receives WorkflowState and returns a *partial* dict containing
  only the keys it writes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Sectors that data/scenarios.py and data/assets.py recognise
_VALID_SECTORS = frozenset(
    {"Infrastructure", "Renewables", "Technology", "Real Estate", "M&A"}
)

# LLM model used for commentary (highest-quality Groq model already in project)
_REASONING_MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — Data Validation
# ══════════════════════════════════════════════════════════════════════════════

def data_validation_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Load investment_pipeline.xlsx via data/assets.py and validate every
        asset against business rules.  Invalid assets are logged and excluded;
        valid ones are forwarded.  The node never aborts on validation failure —
        it continues with the clean subset.

    Inputs read from state:
        (none — reads directly from the file system via data/assets.py)

    Outputs written to state:
        validated_assets   : List[Asset] — assets that passed all checks
        validation_errors  : List[str]   — human-readable error messages
    """
    from data.assets import get_asset_pool

    errors: List[str] = []
    valid_assets: List[Any] = []

    try:
        all_assets = get_asset_pool()
    except (FileNotFoundError, ValueError) as exc:
        msg = str(exc)
        logger.error("data_validation_node: cannot load assets — %s", msg)
        print(f"\n  [Validation] FATAL: {msg}")
        return {"validated_assets": [], "validation_errors": [msg]}

    for asset in all_assets:
        asset_errors: List[str] = []

        # ── Cash flows: exactly 8 years, no NaN ──────────────────────────────
        cfs = asset.annual_cash_flows
        if len(cfs) != 8:
            asset_errors.append(
                f"[{asset.id}] Expected 8 annual cash flows (Y1-Y8), "
                f"found {len(cfs)}."
            )
        elif any(cf != cf for cf in cfs):          # NaN test: NaN != NaN
            asset_errors.append(
                f"[{asset.id}] NaN value in annual_cash_flows."
            )

        # ── Sector name ───────────────────────────────────────────────────────
        if asset.sector not in _VALID_SECTORS:
            asset_errors.append(
                f"[{asset.id}] Invalid sector '{asset.sector}'. "
                f"Must be one of: {sorted(_VALID_SECTORS)}."
            )

        # ── Discount rate: [0.01, 0.30] ───────────────────────────────────────
        if not (0.01 <= asset.discount_rate <= 0.30):
            asset_errors.append(
                f"[{asset.id}] discount_rate {asset.discount_rate:.4f} "
                f"outside [0.01, 0.30]."
            )

        # ── Risk sigma: [0.05, 0.30] ──────────────────────────────────────────
        if not (0.05 <= asset.risk_sigma <= 0.30):
            asset_errors.append(
                f"[{asset.id}] risk_sigma {asset.risk_sigma:.4f} "
                f"outside [0.05, 0.30]."
            )

        if asset_errors:
            for e in asset_errors:
                logger.warning("Validation: %s", e)
                errors.append(e)
        else:
            valid_assets.append(asset)

    if errors:
        print(
            f"\n  [Validation] {len(errors)} issue(s) found — "
            f"continuing with {len(valid_assets)} valid asset(s)."
        )
        for e in errors:
            print(f"    ! {e}")
    else:
        print(f"\n  [Validation] All {len(valid_assets)} assets passed.")

    return {
        "validated_assets":  valid_assets,
        "validation_errors": errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 — Scenario Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def scenario_retrieval_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Resolve user_intent into one or more Scenario objects.
        • "run_all"           → all 22 scenarios from data/scenarios.py
        • any other string    → agentic RAG pipeline (retrieval/agentic_rag.py)
          produces a single composed Scenario and a confidence score.

    Inputs read from state:
        user_intent : str

    Outputs written to state:
        scenario        : List[Scenario]  (run_all)  or  Scenario  (NL query)
        rag_confidence  : None            (run_all)  or  float     (NL query)
    """
    user_intent: str = state["user_intent"]

    if user_intent == "run_all":
        from data.scenarios import get_all_scenarios
        scenarios = get_all_scenarios()
        print(f"  [Scenario] Loaded {len(scenarios)} scenarios for batch run.")
        return {"scenario": scenarios, "rag_confidence": None}

    # ── Natural language query → Agentic RAG ──────────────────────────────────
    from data.scenarios import get_all_scenarios
    from retrieval.scenario_store import ScenarioStore
    from retrieval.agentic_rag import AgenticRAG
    from llm.claude_client import ClaudeClient

    scenarios = get_all_scenarios()
    store = ScenarioStore(scenarios)
    client = ClaudeClient()

    print(f"  [Scenario] Agentic RAG — query: \"{user_intent}\"")
    rag_result = AgenticRAG(store, client).retrieve(user_intent)

    print(
        f"  [Scenario] Composed: '{rag_result.composed_scenario.name}' "
        f"(confidence={rag_result.confidence:.2f}, "
        f"attempts={rag_result.attempts})"
    )
    return {
        "scenario":       rag_result.composed_scenario,
        "rag_confidence": rag_result.confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 — Financial Engine
# ══════════════════════════════════════════════════════════════════════════════

def financial_engine_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Apply scenario modifiers, run the greedy NPV optimiser, and stress-test
        the selected portfolio with Monte Carlo.  Wraps engine/ modules without
        modifying them.

    Inputs read from state:
        validated_assets : List[Asset]
        scenario         : Scenario  or  List[Scenario]
        budget_gbp       : float
        mc_iterations    : int

    Outputs written to state:
        engine_result : ScenarioRunResult  (single)
                      | List[ScenarioRunResult]  (run_all)
    """
    from engine.npv_engine import calculate_npv
    from engine.optimizer import greedy_knapsack_optimizer
    from engine.monte_carlo import run_monte_carlo
    from models import ScenarioRunResult

    assets: List[Any] = state["validated_assets"] or []
    scenario_input: Any = state["scenario"]
    budget: float = state["budget_gbp"]
    mc_iters: int = state["mc_iterations"]

    def _run_one(scenario: Any, seed_offset: int = 0) -> "ScenarioRunResult":
        # Filter by sector mandate (empty list = all sectors)
        candidates = (
            assets
            if not scenario.eligible_sectors
            else [a for a in assets if a.sector in scenario.eligible_sectors]
        )

        # Apply scenario modifiers
        adj_npvs: Dict[str, float] = {}
        adj_capex: Dict[str, float] = {}
        adj_cfs: Dict[str, List[float]] = {}
        adj_dr: Dict[str, float] = {}

        for asset in candidates:
            cfs   = [cf * scenario.cash_flow_modifier for cf in asset.annual_cash_flows]
            capex = asset.capital_required * scenario.capex_modifier
            dr    = asset.discount_rate + scenario.discount_rate_delta
            adj_cfs[asset.id]   = cfs
            adj_capex[asset.id] = capex
            adj_dr[asset.id]    = dr
            adj_npvs[asset.id]  = calculate_npv(cfs, capex, dr)

        # Greedy budget-constrained optimiser
        opt = greedy_knapsack_optimizer(candidates, adj_npvs, adj_capex, budget)

        # Monte Carlo stress test on selected portfolio only
        selected = opt.selected_assets
        mc = run_monte_carlo(
            selected_assets=selected,
            adjusted_cfs={a.id: adj_cfs[a.id] for a in selected},
            adjusted_capex={a.id: adj_capex[a.id] for a in selected},
            adjusted_discount_rates={a.id: adj_dr[a.id] for a in selected},
            sigma_multiplier=scenario.risk_sigma_multiplier,
            n_iterations=mc_iters,
            random_seed=42 + seed_offset,
        )

        return ScenarioRunResult(scenario=scenario, optimization=opt, monte_carlo=mc)

    if isinstance(scenario_input, list):
        results = []
        total = len(scenario_input)
        for i, sc in enumerate(scenario_input):
            print(f"  [Engine] {i+1}/{total}: {sc.name}")
            results.append(_run_one(sc, seed_offset=i))
        return {"engine_result": results}

    print(f"  [Engine] Running: {scenario_input.name}")
    return {"engine_result": _run_one(scenario_input)}


# ══════════════════════════════════════════════════════════════════════════════
# Node 4a — Commentary  (runs in parallel with risk_flag_node)
# ══════════════════════════════════════════════════════════════════════════════

_COMMENTARY_SYSTEM = (
    "You are a senior financial analyst presenting portfolio results to a CFO "
    "or board member. Explain all results in plain English — no jargon, no "
    "formulas. Reference specific numbers from the data provided. "
    "Return ONLY valid JSON matching the requested schema. No markdown fences."
)


def commentary_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Generate plain-English chart explanations and ranked strategic
        suggestions using the Groq LLM via llm/claude_client.ClaudeClient.
        Uses the base case (scenario id=1) when engine_result is a list.

    Inputs read from state:
        engine_result : ScenarioRunResult | List[ScenarioRunResult]

    Outputs written to state:
        commentary : dict  {
            "chart_explanations": {
                "npv_waterfall": str,
                "monte_carlo":   str,
                "sector_allocation": str,
                "cashflow_projection": str,
            },
            "suggestions": [
                {"rank": int, "urgency": "High|Medium|Low",
                 "action": str, "rationale": str},
                ...
            ]
        }
    """
    from llm.claude_client import ClaudeClient

    engine_result = state["engine_result"]

    # Resolve to a single result for commentary
    if isinstance(engine_result, list):
        result = next(
            (r for r in engine_result if r.scenario.id == 1), engine_result[0]
        )
    else:
        result = engine_result

    opt = result.optimization
    mc  = result.monte_carlo
    mult = opt.total_npv / max(opt.total_capital_deployed, 1.0)

    # Build sector allocation summary
    sector_capex: Dict[str, float] = {}
    for asset in opt.selected_assets:
        sector_capex[asset.sector] = (
            sector_capex.get(asset.sector, 0.0) + asset.capital_required
        )
    sector_lines = "\n".join(
        f"  {s}: £{v/1e6:.1f}M ({v/opt.total_capital_deployed*100:.0f}%)"
        for s, v in sector_capex.items()
    ) if opt.total_capital_deployed else "  No assets selected."

    selected_lines = "\n".join(
        f"  {a.name} ({a.sector}): capex=£{a.capital_required/1e6:.1f}M  "
        f"PI={opt.profitability_indices.get(a.id, 0):.1f}x"
        for a in opt.selected_assets
    ) or "  None"

    prompt = (
        f"PORTFOLIO RESULTS:\n"
        f"Total NPV: £{opt.total_npv/1e6:.1f}M | "
        f"Capital: £{opt.total_capital_deployed/1e6:.1f}M | "
        f"Return: {mult:.2f}x | "
        f"Remaining budget: £{opt.remaining_budget/1e6:.1f}M\n\n"
        f"Projects ({len(opt.selected_assets)}):\n{selected_lines}\n\n"
        f"Sector allocation:\n{sector_lines}\n\n"
        f"Monte Carlo (10,000 iter): "
        f"mean=£{mc.mean_total_npv/1e6:.1f}M  "
        f"std=£{mc.std_total_npv/1e6:.1f}M  "
        f"P05=£{mc.p5_total_npv/1e6:.1f}M  "
        f"P95=£{mc.p95_total_npv/1e6:.1f}M  "
        f"deficit={mc.deficit_probability*100:.4f}%\n\n"
        'Return this JSON (no markdown):\n'
        '{"chart_explanations":{'
        '"npv_waterfall":"<1-2 sentences>",'
        '"monte_carlo":"<1-2 sentences>",'
        '"sector_allocation":"<1-2 sentences>",'
        '"cashflow_projection":"<1-2 sentences>"},'
        '"suggestions":['
        '{"rank":1,"urgency":"High","action":"<action>","rationale":"<with numbers>"},'
        '{"rank":2,"urgency":"Medium","action":"<action>","rationale":"<with numbers>"},'
        '{"rank":3,"urgency":"Low","action":"<action>","rationale":"<with numbers>"}'
        "]}"
    )

    client = ClaudeClient()
    raw = client._call(
        model=_REASONING_MODEL,
        system=_COMMENTARY_SYSTEM,
        user_text=prompt,
    )

    try:
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        commentary = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        logger.warning("commentary_node: LLM returned non-JSON; using fallback.")
        commentary = {
            "chart_explanations": {
                "npv_waterfall": (
                    f"The portfolio generates £{opt.total_npv/1e6:.0f}M NPV "
                    f"from £{opt.total_capital_deployed/1e6:.0f}M invested "
                    f"across {len(opt.selected_assets)} projects."
                ),
                "monte_carlo": (
                    f"Across 10,000 simulated futures the mean NPV is "
                    f"£{mc.mean_total_npv/1e6:.0f}M with "
                    f"{mc.deficit_probability*100:.4f}% probability of loss."
                ),
                "sector_allocation": (
                    f"Capital is distributed across {len(sector_capex)} sectors, "
                    "reducing single-sector concentration risk."
                ),
                "cashflow_projection": (
                    "The portfolio delivers progressive cash flows over "
                    "the 8-year investment horizon."
                ),
            },
            "suggestions": [
                {
                    "rank": 1,
                    "urgency": "High",
                    "action": "Proceed with Wave-1 portfolio deployment.",
                    "rationale": (
                        f"£{opt.total_npv/1e6:.0f}M NPV at {mult:.1f}x return "
                        f"with {mc.deficit_probability*100:.4f}% deficit probability."
                    ),
                },
            ],
        }

    print(
        f"  [Commentary] Generated "
        f"{len(commentary.get('suggestions', []))} suggestion(s)."
    )
    return {"commentary": commentary}


# ══════════════════════════════════════════════════════════════════════════════
# Node 4b — Risk Flags  (runs in parallel with commentary_node)
# ══════════════════════════════════════════════════════════════════════════════

def risk_flag_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Evaluate the engine_result for financial and operational risk conditions.
        Each flag is classified HIGH / MEDIUM / LOW with a plain-English
        description and the specific metric that triggered it.
        Uses base case (scenario id=1) when engine_result is a list.

    Inputs read from state:
        engine_result   : ScenarioRunResult | List[ScenarioRunResult]
        rag_confidence  : float | None

    Outputs written to state:
        risk_flags : List[dict]  — each dict has:
            severity        : "HIGH" | "MEDIUM" | "LOW"
            description     : str   (plain English, actionable)
            affected_metric : str   (specific number that triggered the flag)
    """
    engine_result  = state["engine_result"]
    rag_confidence = state.get("rag_confidence")

    if isinstance(engine_result, list):
        result = next(
            (r for r in engine_result if r.scenario.id == 1), engine_result[0]
        )
    else:
        result = engine_result

    opt   = result.optimization
    mc    = result.monte_carlo
    flags: List[dict] = []

    total_deployed = opt.total_capital_deployed

    # ── 1. Sector concentration > 60% ────────────────────────────────────────
    if total_deployed > 0:
        sector_capex: Dict[str, float] = {}
        for asset in opt.selected_assets:
            sector_capex[asset.sector] = (
                sector_capex.get(asset.sector, 0.0) + asset.capital_required
            )
        for sector, capex in sector_capex.items():
            pct = capex / total_deployed
            if pct > 0.60:
                flags.append({
                    "severity": "HIGH",
                    "description": (
                        f"Portfolio is {pct*100:.1f}% concentrated in "
                        f"{sector}. Diversify across additional sectors to "
                        "reduce single-sector exposure."
                    ),
                    "affected_metric": (
                        f"Sector concentration: {sector} = "
                        f"{pct*100:.1f}% of £{total_deployed/1e6:.1f}M deployed"
                    ),
                })

    # ── 2. Deficit probability > 0% ──────────────────────────────────────────
    if mc.deficit_probability > 0.0:
        flags.append({
            "severity": "HIGH",
            "description": (
                f"Monte Carlo shows {mc.deficit_probability*100:.4f}% probability "
                "of negative portfolio NPV. Immediate review of high-risk assets "
                "is recommended."
            ),
            "affected_metric": (
                f"deficit_probability = {mc.deficit_probability*100:.4f}% "
                f"({mc.deficit_count} of {mc.n_iterations:,} simulations)"
            ),
        })

    # ── 3. MC std dev > 10% of mean NPV ──────────────────────────────────────
    if mc.mean_total_npv > 0 and mc.std_total_npv / mc.mean_total_npv > 0.10:
        ratio = mc.std_total_npv / mc.mean_total_npv
        flags.append({
            "severity": "MEDIUM",
            "description": (
                f"High NPV volatility: standard deviation is "
                f"{ratio*100:.1f}% of mean NPV "
                f"(£{mc.std_total_npv/1e6:.1f}M on £{mc.mean_total_npv/1e6:.1f}M). "
                "Consider reducing exposure to high-sigma assets."
            ),
            "affected_metric": (
                f"std_dev / mean_npv = {ratio*100:.1f}%  (threshold: 10%)"
            ),
        })

    # ── 4. RAG confidence < 0.70 ──────────────────────────────────────────────
    if rag_confidence is not None and rag_confidence < 0.70:
        flags.append({
            "severity": "MEDIUM",
            "description": (
                f"Agentic RAG confidence is {rag_confidence:.2f} (threshold 0.70). "
                "The composed scenario may not fully capture all economic conditions. "
                "Review scenario parameters before proceeding."
            ),
            "affected_metric": (
                f"rag_confidence = {rag_confidence:.2f}  (threshold: 0.70)"
            ),
        })

    # ── 5. Any scenario NPV < 0 ───────────────────────────────────────────────
    if isinstance(engine_result, list):
        negative = [r for r in engine_result if r.optimization.total_npv < 0]
        if negative:
            names = ", ".join(r.scenario.name for r in negative[:3])
            more = f" (+{len(negative)-3} more)" if len(negative) > 3 else ""
            flags.append({
                "severity": "MEDIUM",
                "description": (
                    f"{len(negative)} scenario(s) produce negative portfolio NPV: "
                    f"{names}{more}. Review portfolio composition under stress."
                ),
                "affected_metric": (
                    f"{len(negative)} of {len(engine_result)} scenarios have NPV < 0"
                ),
            })
    else:
        if result.optimization.total_npv < 0:
            flags.append({
                "severity": "HIGH",
                "description": (
                    f"Portfolio NPV is negative "
                    f"(£{result.optimization.total_npv/1e6:.1f}M). "
                    "This portfolio destroys value under current parameters."
                ),
                "affected_metric": (
                    f"total_npv = £{result.optimization.total_npv/1e6:.1f}M"
                ),
            })

    # Sort: HIGH first, then MEDIUM, then LOW
    _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: _order.get(f["severity"], 3))

    print(f"  [Risk] {len(flags)} flag(s): {[f['severity'] for f in flags]}")
    return {"risk_flags": flags}


# ══════════════════════════════════════════════════════════════════════════════
# Node 5 — Human Approval  (uses LangGraph interrupt)
# ══════════════════════════════════════════════════════════════════════════════

def human_approval_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Pause workflow execution via LangGraph interrupt() and wait for a
        human yes/no decision before report generation proceeds.
        The interrupt payload contains a structured summary that callers
        (CLI or Streamlit) can render for the user.

    Inputs read from state:
        engine_result : ScenarioRunResult | List[ScenarioRunResult]
        risk_flags    : List[dict]
        commentary    : dict

    Outputs written to state:
        human_approved : bool  — True → proceed to report_generation_node
    """
    from langgraph.types import interrupt

    engine_result = state["engine_result"]
    risk_flags    = state.get("risk_flags") or []
    commentary    = state.get("commentary") or {}

    if isinstance(engine_result, list):
        result = next(
            (r for r in engine_result if r.scenario.id == 1), engine_result[0]
        )
    else:
        result = engine_result

    opt  = result.optimization
    mc   = result.monte_carlo
    mult = opt.total_npv / max(opt.total_capital_deployed, 1.0)

    suggestions = commentary.get("suggestions", [])
    first_suggestion = suggestions[0] if suggestions else None

    # Build interrupt payload — callers render this however they like
    payload = {
        "results": {
            "total_npv_gbp":          round(opt.total_npv, 0),
            "total_npv_m":            round(opt.total_npv / 1e6, 1),
            "return_multiplier":      round(mult, 2),
            "capital_deployed_gbp":   round(opt.total_capital_deployed, 0),
            "deficit_probability_pct": round(mc.deficit_probability * 100, 4),
        },
        "risk_flags":     risk_flags,
        "top_suggestion": first_suggestion,
        "prompt":         "Do you want to proceed to report generation? (yes/no)",
    }

    # Pause here — resumes when caller calls graph.stream(Command(resume=...))
    response = interrupt(payload)

    approved = str(response).lower().strip() in ("yes", "y", "true", "1")
    return {"human_approved": approved}


# ══════════════════════════════════════════════════════════════════════════════
# Node 6 — Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def report_generation_node(state: WorkflowState) -> dict:
    """
    Purpose:
        Generate three output artefacts — JSON, CSV, and PDF board summary.
        JSON and CSV use the existing reporting/file_reporter.py functions.
        PDF is built with reportlab: three pages (executive summary,
        portfolio table, Monte Carlo analysis).

    Inputs read from state:
        engine_result  : ScenarioRunResult | List[ScenarioRunResult]
        commentary     : dict
        risk_flags     : List[dict]
        human_approved : bool  (node only runs when True)

    Outputs written to state:
        report_paths : dict  {"json": str, "csv": str, "pdf": str}
    """
    if not state.get("human_approved"):
        return {"report_paths": {}}

    engine_result = state["engine_result"]
    commentary    = state.get("commentary") or {}
    risk_flags    = state.get("risk_flags") or []

    # Normalise to list for file_reporter functions (which accept List[ScenarioRunResult])
    results_list = (
        engine_result if isinstance(engine_result, list) else [engine_result]
    )

    from reporting.file_reporter import save_results_json, save_wave1_csv

    json_path = save_results_json(results_list)
    csv_path  = save_wave1_csv(results_list)
    pdf_path  = _generate_pdf(results_list, commentary, risk_flags)

    print(f"  [Report] PDF board summary → {pdf_path}")

    return {
        "report_paths": {
            "json": json_path,
            "csv":  csv_path,
            "pdf":  pdf_path,
        }
    }


# ── PDF helper ────────────────────────────────────────────────────────────────

def _generate_pdf(results: list, commentary: dict, risk_flags: list) -> str:
    """
    Build a 3-page PDF using reportlab.
      Page 1 — Executive Summary  (key metrics, top 3 flags, top 2 suggestions)
      Page 2 — Selected Portfolio Table
      Page 3 — Monte Carlo Analysis
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    # Use base case if multiple results
    base = next((r for r in results if r.scenario.id == 1), results[0])
    opt  = base.optimization
    mc   = base.monte_carlo
    mult = opt.total_npv / max(opt.total_capital_deployed, 1.0)

    os.makedirs("output", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join("output", f"board_summary_{ts}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles   = getSampleStyleSheet()
    navy     = colors.HexColor("#003366")
    silver   = colors.HexColor("#f0f4f8")
    midgrey  = colors.HexColor("#aaaaaa")

    title_sty = ParagraphStyle(
        "BoardTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=10,
        textColor=navy, alignment=TA_CENTER,
    )
    metric_sty = ParagraphStyle(
        "Metric", parent=styles["Normal"],
        fontSize=14, spaceAfter=5,
        textColor=navy, fontName="Helvetica-Bold",
    )
    h2_sty = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=12, spaceBefore=12, spaceAfter=5, textColor=navy,
    )
    body_sty = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, spaceAfter=3, leading=13,
    )
    flag_hi  = ParagraphStyle("FlagHi",  parent=body_sty,
                               textColor=colors.HexColor("#cc0000"))
    flag_med = ParagraphStyle("FlagMed", parent=body_sty,
                               textColor=colors.HexColor("#cc7700"))
    flag_lo  = ParagraphStyle("FlagLo",  parent=body_sty,
                               textColor=colors.HexColor("#007700"))

    def _flag_style(sev: str) -> ParagraphStyle:
        return flag_hi if sev == "HIGH" else (flag_med if sev == "MEDIUM" else flag_lo)

    tbl_header = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  navy),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [silver, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, midgrey),
        ("ALIGN",         (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

    story = []

    # ── PAGE 1: EXECUTIVE SUMMARY ─────────────────────────────────────────────
    story.append(Paragraph("Capital Portfolio Optimisation", title_sty))
    story.append(Paragraph("Board Executive Summary", styles["Heading2"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Strategic Value (NPV):   £{opt.total_npv/1e6:,.0f}M", metric_sty))
    story.append(Paragraph(f"Capital Deployed:        £{opt.total_capital_deployed/1e6:,.0f}M", metric_sty))
    story.append(Paragraph(f"Return Multiplier:       {mult:.2f}×", metric_sty))
    story.append(Paragraph(f"Deficit Probability:     {mc.deficit_probability*100:.4f}%", metric_sty))
    story.append(Spacer(1, 0.3*cm))

    if risk_flags:
        story.append(Paragraph("Risk Flags", h2_sty))
        for flag in risk_flags[:3]:
            sev = flag["severity"]
            story.append(Paragraph(f"[{sev}]  {flag['description']}", _flag_style(sev)))
            story.append(Paragraph(f"Metric: {flag['affected_metric']}", body_sty))
            story.append(Spacer(1, 0.15*cm))

    suggestions = commentary.get("suggestions", [])
    if suggestions:
        story.append(Paragraph("Strategic Recommendations", h2_sty))
        for sug in suggestions[:2]:
            story.append(Paragraph(
                f"#{sug.get('rank','?')} [{sug.get('urgency','—')}]  "
                f"{sug.get('action','')}",
                body_sty,
            ))
            story.append(Paragraph(sug.get("rationale", ""), body_sty))
            story.append(Spacer(1, 0.1*cm))

    story.append(PageBreak())

    # ── PAGE 2: SELECTED PORTFOLIO TABLE ─────────────────────────────────────
    story.append(Paragraph("Selected Portfolio", title_sty))
    story.append(Spacer(1, 0.3*cm))

    rows = [["Project", "Sector", "Capex (£M)", "NPV (£M)", "PI"]]
    for asset in opt.selected_assets:
        pi  = opt.profitability_indices.get(asset.id, 0.0)
        npv = opt.per_asset_npv.get(asset.id, 0.0)
        rows.append([
            asset.name, asset.sector,
            f"£{asset.capital_required/1e6:.2f}M",
            f"£{npv/1e6:.1f}M",
            f"{pi:.2f}×",
        ])
    rows.append([
        "TOTAL", "",
        f"£{opt.total_capital_deployed/1e6:.2f}M",
        f"£{opt.total_npv/1e6:.1f}M",
        f"{mult:.2f}×",
    ])

    tbl = Table(rows, colWidths=[5.5*cm, 3.2*cm, 2.6*cm, 2.6*cm, 2.1*cm])
    totals_style = TableStyle([
        ("BACKGROUND",  (0, -1), (-1, -1), colors.HexColor("#ddeeff")),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
    ])
    tbl.setStyle(tbl_header)
    tbl.setStyle(totals_style)
    story.append(tbl)
    story.append(PageBreak())

    # ── PAGE 3: MONTE CARLO SUMMARY ───────────────────────────────────────────
    story.append(Paragraph("Monte Carlo Risk Analysis", title_sty))
    story.append(Spacer(1, 0.3*cm))

    mc_rows = [
        ["Metric", "Value"],
        ["Iterations",              f"{mc.n_iterations:,}"],
        ["Mean Portfolio NPV",      f"£{mc.mean_total_npv/1e6:,.1f}M"],
        ["Std Deviation",           f"£{mc.std_total_npv/1e6:,.1f}M"],
        ["5th Percentile (P05)",    f"£{mc.p5_total_npv/1e6:,.1f}M"],
        ["95th Percentile (P95)",   f"£{mc.p95_total_npv/1e6:,.1f}M"],
        ["Deficit Simulations",     f"{mc.deficit_count:,} of {mc.n_iterations:,}"],
        ["Deficit Probability",     f"{mc.deficit_probability*100:.4f}%"],
    ]
    mc_tbl = Table(mc_rows, colWidths=[8*cm, 6*cm])
    mc_tbl.setStyle(tbl_header)
    story.append(mc_tbl)
    story.append(Spacer(1, 0.4*cm))

    interp = (
        commentary.get("chart_explanations", {}).get("monte_carlo")
        or (
            f"Across {mc.n_iterations:,} simulated economic futures the portfolio "
            f"produces a mean NPV of £{mc.mean_total_npv/1e6:.0f}M. In the worst "
            f"5% of outcomes (P05) NPV falls to £{mc.p5_total_npv/1e6:.0f}M; "
            f"in the best 5% (P95) it reaches £{mc.p95_total_npv/1e6:.0f}M. "
            f"Probability of a portfolio loss: {mc.deficit_probability*100:.4f}%."
        )
    )
    story.append(Paragraph("Interpretation", h2_sty))
    story.append(Paragraph(interp, body_sty))

    doc.build(story)
    return pdf_path
