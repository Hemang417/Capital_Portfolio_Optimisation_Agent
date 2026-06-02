"""
tests/test_workflow.py — Rigorous test suite for the LangGraph workflow.

All external calls (Groq API, Excel reading, LangGraph interrupt) are mocked.
No live API key or investment_pipeline.xlsx is needed to run this file.

Run:  python -m pytest tests/test_workflow.py -v
"""

import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.assets import Asset
from data.scenarios import Scenario, get_all_scenarios
from engine.monte_carlo import MonteCarloResult
from engine.optimizer import OptimizationResult
from models import ScenarioRunResult

import numpy as np


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_asset(
    aid: str = "T1",
    sector: str = "Technology",
    capex: float = 4_000_000,
    cfs=None,
    dr: float = 0.08,
    sigma: float = 0.12,
) -> Asset:
    if cfs is None:
        cfs = [1_500_000] * 8
    return Asset(
        id=aid, name=f"Asset {aid}", sector=sector,
        capital_required=capex, annual_cash_flows=cfs,
        discount_rate=dr, risk_sigma=sigma,
    )


def _make_scenario(sid: int = 1, sectors: list = None) -> Scenario:
    return Scenario(
        id=sid, name=f"Scenario {sid}", description="Test",
        cash_flow_modifier=1.0, discount_rate_delta=0.0,
        capex_modifier=1.0, risk_sigma_multiplier=1.0,
        eligible_sectors=sectors or [],
    )


def _make_opt_result(assets, npv: float = 80_000_000) -> OptimizationResult:
    capex = sum(a.capital_required for a in assets)
    pis   = {a.id: npv / capex for a in assets}
    return OptimizationResult(
        selected_assets=assets,
        total_capital_deployed=capex,
        total_npv=npv,
        remaining_budget=45_000_000 - capex,
        profitability_indices=pis,
        per_asset_npv={a.id: npv / len(assets) for a in assets},
    )


def _make_mc_result(n: int = 1_000) -> MonteCarloResult:
    samples = np.full(n, 80_000_000.0)
    return MonteCarloResult(
        n_iterations=n,
        deficit_count=0,
        deficit_probability=0.0,
        mean_total_npv=80_000_000.0,
        std_total_npv=500_000.0,
        p5_total_npv=79_000_000.0,
        p95_total_npv=81_000_000.0,
        all_npv_samples=samples,
    )


def _make_run_result(sid: int = 1, assets=None) -> ScenarioRunResult:
    if assets is None:
        assets = [_make_asset()]
    opt = _make_opt_result(assets)
    mc  = _make_mc_result()
    return ScenarioRunResult(
        scenario=_make_scenario(sid),
        optimization=opt,
        monte_carlo=mc,
    )


def _commentary_json() -> str:
    return json.dumps({
        "chart_explanations": {
            "npv_waterfall":      "Strong NPV build.",
            "monte_carlo":        "Low variance.",
            "sector_allocation":  "Well diversified.",
            "cashflow_projection": "Steady growth.",
        },
        "suggestions": [
            {"rank": 1, "urgency": "High",   "action": "Deploy capital.", "rationale": "20x return."},
            {"rank": 2, "urgency": "Medium",  "action": "Monitor risk.",   "rationale": "Std 0.6%."},
            {"rank": 3, "urgency": "Low",     "action": "Review Q4.",      "rationale": "Track CF."},
        ],
    })


# ── Test 1: Full graph executes without error on valid input ─────────────────

class TestFullGraphExecution:

    def test_full_graph_runs_on_valid_input(self):
        """
        Full graph executes without error when given valid assets.
        Uses run_all mode (no LLM needed for scenario retrieval).
        Commentary and risk_flag nodes use mocked ClaudeClient.
        interrupt() is mocked to immediately return 'yes'.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from workflow.graph import build_graph

        assets = [_make_asset("A1"), _make_asset("A2", "Renewables", 5_000_000)]

        with (
            patch("data.assets.get_asset_pool", return_value=assets),
            patch("llm.claude_client.ClaudeClient.__init__", return_value=None),
            patch("llm.claude_client.ClaudeClient._call",
                  return_value=_commentary_json()),
            patch("langgraph.types.interrupt", return_value="yes"),
        ):
            graph   = build_graph(MemorySaver())
            config  = {"configurable": {"thread_id": str(uuid.uuid4())}}
            initial = {
                "user_intent":   "run_all",
                "budget_gbp":    45_000_000.0,
                "mc_iterations": 100,
            }

            events = list(graph.stream(initial, config=config, stream_mode="updates"))

        completed_nodes = {k for ev in events for k in ev}
        assert "data_validation_node"    in completed_nodes
        assert "scenario_retrieval_node" in completed_nodes
        assert "financial_engine_node"   in completed_nodes
        assert "commentary_node"         in completed_nodes
        assert "risk_flag_node"          in completed_nodes


# ── Test 2: data_validation_node catches invalid sector ──────────────────────

class TestDataValidationNode:

    def test_invalid_sector_excluded_valid_assets_kept(self):
        """
        data_validation_node excludes the bad asset and continues
        with the remaining valid ones — does not abort.
        """
        from workflow.nodes import data_validation_node

        good  = _make_asset("G1", sector="Technology")
        bad   = _make_asset("B1", sector="INVALID_SECTOR_XYZ")

        with patch("data.assets.get_asset_pool", return_value=[good, bad]):
            result = data_validation_node({
                "user_intent":   "run_all",
                "budget_gbp":    45_000_000.0,
                "mc_iterations": 1_000,
                "validated_assets":  None,
                "validation_errors": None,
                "scenario":          None,
                "rag_confidence":    None,
                "engine_result":     None,
                "commentary":        None,
                "risk_flags":        None,
                "human_approved":    None,
                "report_paths":      None,
            })

        # Good asset retained
        assert len(result["validated_assets"]) == 1
        assert result["validated_assets"][0].id == "G1"
        # Error recorded for bad asset
        assert any("B1" in e for e in result["validation_errors"])

    def test_all_assets_valid_no_errors(self):
        """When all assets are valid, validation_errors is empty."""
        from workflow.nodes import data_validation_node

        assets = [_make_asset("V1"), _make_asset("V2", "Renewables")]

        with patch("data.assets.get_asset_pool", return_value=assets):
            result = data_validation_node({
                "user_intent": "run_all", "budget_gbp": 45e6,
                "mc_iterations": 100, "validated_assets": None,
                "validation_errors": None, "scenario": None,
                "rag_confidence": None, "engine_result": None,
                "commentary": None, "risk_flags": None,
                "human_approved": None, "report_paths": None,
            })

        assert result["validation_errors"] == []
        assert len(result["validated_assets"]) == 2

    def test_out_of_range_discount_rate_excluded(self):
        """Asset with discount_rate=0.95 (>0.30) must be excluded."""
        from workflow.nodes import data_validation_node

        bad = _make_asset("DR", dr=0.95)
        with patch("data.assets.get_asset_pool", return_value=[bad]):
            result = data_validation_node({
                "user_intent": "run_all", "budget_gbp": 45e6,
                "mc_iterations": 100, "validated_assets": None,
                "validation_errors": None, "scenario": None,
                "rag_confidence": None, "engine_result": None,
                "commentary": None, "risk_flags": None,
                "human_approved": None, "report_paths": None,
            })

        assert result["validated_assets"] == []
        assert any("DR" in e for e in result["validation_errors"])


# ── Test 3: human_approval_node with "no" ends before report generation ──────

class TestHumanApprovalNode:

    def test_rejection_sets_human_approved_false(self):
        """interrupt() returning 'no' → human_approved=False."""
        from workflow.nodes import human_approval_node

        run_result = _make_run_result()
        state = {
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": None, "engine_result": run_result,
            "commentary": {"chart_explanations": {}, "suggestions": []},
            "risk_flags": [], "human_approved": None, "report_paths": None,
        }

        with patch("langgraph.types.interrupt", return_value="no"):
            result = human_approval_node(state)

        assert result["human_approved"] is False

    def test_approval_sets_human_approved_true(self):
        """interrupt() returning 'yes' → human_approved=True."""
        from workflow.nodes import human_approval_node

        run_result = _make_run_result()
        state = {
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": None, "engine_result": run_result,
            "commentary": {"chart_explanations": {}, "suggestions": []},
            "risk_flags": [], "human_approved": None, "report_paths": None,
        }

        with patch("langgraph.types.interrupt", return_value="yes"):
            result = human_approval_node(state)

        assert result["human_approved"] is True

    def test_rejection_graph_does_not_reach_report_node(self):
        """
        When interrupt returns 'no', report_generation_node must NOT be called.
        Verify by checking completed nodes after streaming.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from workflow.graph import build_graph

        assets = [_make_asset()]

        with (
            patch("data.assets.get_asset_pool", return_value=assets),
            patch("llm.claude_client.ClaudeClient.__init__", return_value=None),
            patch("llm.claude_client.ClaudeClient._call",
                  return_value=_commentary_json()),
            patch("langgraph.types.interrupt", return_value="no"),
        ):
            graph   = build_graph(MemorySaver())
            config  = {"configurable": {"thread_id": str(uuid.uuid4())}}
            initial = {
                "user_intent":   "run_all",
                "budget_gbp":    45_000_000.0,
                "mc_iterations": 100,
            }
            events = list(graph.stream(initial, config=config, stream_mode="updates"))

        completed = {k for ev in events for k in ev}
        assert "report_generation_node" not in completed


# ── Test 4: Parallel nodes both write to state ────────────────────────────────

class TestParallelNodes:

    def _base_state(self, run_result):
        return {
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": None, "engine_result": run_result,
            "commentary": None, "risk_flags": None,
            "human_approved": None, "report_paths": None,
        }

    def test_commentary_node_writes_commentary_key(self):
        """commentary_node returns dict with 'commentary' key."""
        from workflow.nodes import commentary_node

        run_result = _make_run_result()

        with (
            patch("llm.claude_client.ClaudeClient.__init__", return_value=None),
            patch("llm.claude_client.ClaudeClient._call",
                  return_value=_commentary_json()),
        ):
            result = commentary_node(self._base_state(run_result))

        assert "commentary" in result
        assert "chart_explanations" in result["commentary"]
        assert "suggestions" in result["commentary"]
        assert len(result["commentary"]["suggestions"]) >= 1

    def test_risk_flag_node_writes_risk_flags_key(self):
        """risk_flag_node returns dict with 'risk_flags' key."""
        from workflow.nodes import risk_flag_node

        run_result = _make_run_result()
        result = risk_flag_node(self._base_state(run_result))

        assert "risk_flags" in result
        assert isinstance(result["risk_flags"], list)

    def test_both_nodes_in_graph_both_complete(self):
        """
        In the compiled graph, both commentary_node and risk_flag_node
        appear in the completed nodes set.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from workflow.graph import build_graph

        assets = [_make_asset()]

        with (
            patch("data.assets.get_asset_pool", return_value=assets),
            patch("llm.claude_client.ClaudeClient.__init__", return_value=None),
            patch("llm.claude_client.ClaudeClient._call",
                  return_value=_commentary_json()),
            patch("langgraph.types.interrupt", return_value="yes"),
        ):
            graph  = build_graph(MemorySaver())
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            events = list(graph.stream(
                {"user_intent": "run_all", "budget_gbp": 45e6, "mc_iterations": 100},
                config=config, stream_mode="updates",
            ))

        completed = {k for ev in events for k in ev}
        assert "commentary_node" in completed
        assert "risk_flag_node"  in completed


# ── Test 5: risk_flag_node HIGH flag on sector concentration > 60% ───────────

class TestRiskFlagNode:

    def _state(self, run_result, rag_conf=None):
        return {
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": rag_conf, "engine_result": run_result,
            "commentary": None, "risk_flags": None,
            "human_approved": None, "report_paths": None,
        }

    def test_high_sector_concentration_raises_high_flag(self):
        """
        When all capital is in a single sector (100% concentration > 60%),
        a HIGH severity flag must be raised.
        """
        from workflow.nodes import risk_flag_node

        # Both assets in Technology → 100% concentration
        assets = [
            _make_asset("T1", "Technology", 4_000_000),
            _make_asset("T2", "Technology", 6_000_000),
        ]
        opt    = _make_opt_result(assets, npv=80_000_000)
        mc     = _make_mc_result()
        result = ScenarioRunResult(
            scenario=_make_scenario(), optimization=opt, monte_carlo=mc
        )

        flags = risk_flag_node(self._state(result))["risk_flags"]

        high_flags = [f for f in flags if f["severity"] == "HIGH"]
        assert len(high_flags) >= 1
        assert any("Technology" in f["description"] for f in high_flags)

    def test_no_flag_when_sector_under_60_pct(self):
        """
        When no single sector exceeds 60%, no sector-concentration flag is raised.
        """
        from workflow.nodes import risk_flag_node

        assets = [
            _make_asset("A1", "Technology",     4_000_000),
            _make_asset("A2", "Renewables",      4_000_000),
            _make_asset("A3", "Infrastructure",  4_000_000),
        ]
        opt    = _make_opt_result(assets, npv=80_000_000)
        mc     = _make_mc_result()
        result = ScenarioRunResult(
            scenario=_make_scenario(), optimization=opt, monte_carlo=mc
        )

        flags = risk_flag_node(self._state(result))["risk_flags"]
        concentration_flags = [
            f for f in flags
            if "concentration" in f["description"].lower()
            or "concentration" in f["affected_metric"].lower()
        ]
        assert concentration_flags == []

    def test_low_rag_confidence_raises_medium_flag(self):
        """rag_confidence < 0.70 must produce a MEDIUM flag."""
        from workflow.nodes import risk_flag_node

        result = _make_run_result()
        flags  = risk_flag_node(self._state(result, rag_conf=0.55))["risk_flags"]

        medium_rag = [
            f for f in flags
            if f["severity"] == "MEDIUM" and "rag_confidence" in f["affected_metric"]
        ]
        assert len(medium_rag) == 1

    def test_high_std_dev_raises_medium_flag(self):
        """std_dev > 10% of mean_npv must produce a MEDIUM flag."""
        from workflow.nodes import risk_flag_node

        mc = MonteCarloResult(
            n_iterations=1_000, deficit_count=0, deficit_probability=0.0,
            mean_total_npv=80_000_000.0,
            std_total_npv=10_000_000.0,      # 12.5% of mean → triggers flag
            p5_total_npv=70_000_000.0, p95_total_npv=90_000_000.0,
            all_npv_samples=np.full(1_000, 80_000_000.0),
        )
        opt    = _make_opt_result([_make_asset()])
        result = ScenarioRunResult(
            scenario=_make_scenario(), optimization=opt, monte_carlo=mc
        )
        flags = risk_flag_node(self._state(result))["risk_flags"]
        medium = [f for f in flags if f["severity"] == "MEDIUM"
                  and "std_dev" in f["affected_metric"]]
        assert len(medium) == 1


# ── Test 6: report_generation_node produces all three output files ────────────

class TestReportGenerationNode:

    def test_generates_json_csv_pdf(self, tmp_path, monkeypatch):
        """
        report_generation_node must return paths for json, csv, and pdf,
        and all three files must exist on disk.
        """
        from workflow.nodes import report_generation_node

        # Redirect output to tmp_path
        monkeypatch.chdir(tmp_path)

        run_result = _make_run_result()
        state = {
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": None, "engine_result": run_result,
            "commentary": json.loads(_commentary_json()),
            "risk_flags": [
                {"severity": "LOW", "description": "Minor.",
                 "affected_metric": "test=0"}
            ],
            "human_approved": True,
            "report_paths": None,
        }

        result = report_generation_node(state)

        assert "report_paths" in result
        paths = result["report_paths"]
        assert set(paths.keys()) >= {"json", "csv", "pdf"}
        for fmt, path in paths.items():
            assert os.path.exists(path), f"{fmt} file not found: {path}"

    def test_skipped_when_not_approved(self):
        """report_generation_node returns empty dict when human_approved=False."""
        from workflow.nodes import report_generation_node

        result = report_generation_node({
            "user_intent": "run_all", "budget_gbp": 45e6,
            "mc_iterations": 100, "validated_assets": None,
            "validation_errors": None, "scenario": None,
            "rag_confidence": None, "engine_result": _make_run_result(),
            "commentary": {}, "risk_flags": [],
            "human_approved": False, "report_paths": None,
        })

        assert result == {"report_paths": {}}
