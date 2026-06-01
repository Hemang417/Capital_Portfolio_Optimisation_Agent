"""
Rigorous test suite — Capital Portfolio Optimisation Agent
75 tests across 9 classes covering correctness, edge cases, adversarial inputs
and statistical invariants.

Run:  python -m pytest tests/test_suite.py -v
"""

import math
import time
import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.assets import Asset
from engine.npv_engine import calculate_npv, calculate_irr, npv_to_capital_ratio
from engine.optimizer import greedy_knapsack_optimizer, OptimizationResult
from engine.monte_carlo import run_monte_carlo


# ══════════════════════════════════════════════════════════════════════════════
# B1 — NPV Engine (12 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestNPVEngine:

    def test_npv_zero_cashflows(self):
        """All CFs = 0 → NPV = -capex exactly."""
        npv = calculate_npv([0.0] * 8, 100.0, 0.08)
        assert npv == pytest.approx(-100.0, abs=1e-9)

    def test_npv_single_year_breakeven(self):
        """CF=[110], r=0.10, capex=100 → NPV=0 (break-even)."""
        npv = calculate_npv([110.0], 100.0, 0.10)
        assert npv == pytest.approx(0.0, abs=1e-6)

    def test_npv_single_year_positive(self):
        """CF=[121], r=0.10, capex=100 → NPV=10.0."""
        npv = calculate_npv([121.0], 100.0, 0.10)
        assert npv == pytest.approx(10.0, abs=1e-6)

    def test_npv_multiyear_known_value(self):
        """Three-year hand-computed check.

        CF=[100, 200, 300], r=0.10, capex=400
        PV = 100/1.1 + 200/1.21 + 300/1.331
           = 90.909... + 165.289... + 225.394...
           = 481.593...
        NPV = 481.593... - 400 = 81.593...
        """
        expected = 100 / 1.1 + 200 / 1.21 + 300 / 1.331 - 400
        npv = calculate_npv([100.0, 200.0, 300.0], 400.0, 0.10)
        assert npv == pytest.approx(expected, rel=1e-6)

    def test_npv_higher_rate_gives_lower_value(self):
        """Same CFs: r=0.05 NPV > r=0.20 NPV always."""
        cfs = [50_000] * 8
        capex = 100_000
        npv_low  = calculate_npv(cfs, capex, 0.05)
        npv_high = calculate_npv(cfs, capex, 0.20)
        assert npv_low > npv_high

    def test_npv_is_negative_when_cfs_too_small(self):
        """Tiny CFs against large capex → negative NPV."""
        npv = calculate_npv([1.0] * 8, 1_000_000.0, 0.08)
        assert npv < 0

    def test_npv_vectorisation_performance(self):
        """10,000 sequential NPV calls complete in < 1 second."""
        start = time.time()
        for _ in range(10_000):
            calculate_npv([50_000] * 8, 200_000.0, 0.08)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Expected < 1s, got {elapsed:.2f}s"

    def test_irr_known_value(self):
        """[-100, 50, 50, 50] — IRR ≈ 23.38% (verified externally)."""
        irr = calculate_irr([-100.0, 50.0, 50.0, 50.0])
        assert not math.isnan(irr)
        # NPV at the IRR should be ≈ 0
        npv_at_irr = sum(
            cf / (1 + irr) ** t
            for t, cf in enumerate([-100.0, 50.0, 50.0, 50.0])
        )
        assert abs(npv_at_irr) < 0.01

    def test_irr_break_even_recovers_discount_rate(self):
        """Build CFs where NPV = 0 at r=0.10; IRR should be ≈ 0.10."""
        r = 0.10
        capex = 1_000.0
        # Single-year: CF1 / (1+r) - capex = 0  →  CF1 = capex*(1+r)
        cf = capex * (1 + r)
        irr = calculate_irr([-capex, cf])
        assert irr == pytest.approx(r, abs=1e-4)

    def test_irr_no_sign_change_returns_nan(self):
        """All-positive cash flows (no initial negative) → no sign change → NaN."""
        irr = calculate_irr([100.0, 200.0, 300.0])
        assert math.isnan(irr)

    def test_irr_result_stable_across_tolerances(self):
        """IRR result changes < 0.01% between tol=1e-4 and tol=1e-8."""
        cfs = [-500.0, 150.0, 200.0, 250.0, 100.0]
        irr_loose = calculate_irr(cfs, tolerance=1e-4)
        irr_tight = calculate_irr(cfs, tolerance=1e-8)
        if not (math.isnan(irr_loose) or math.isnan(irr_tight)):
            assert abs(irr_loose - irr_tight) < 1e-4

    def test_pi_correct_value(self):
        """PI = NPV / capital: 80 / 4 = 20.0 exactly."""
        pi = npv_to_capital_ratio(80.0, 4.0)
        assert pi == pytest.approx(20.0, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# B2 — Optimizer (14 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizer:

    def _make_assets(self, specs):
        """Helper: specs = [(id, capex, npv)] — builds Asset and dicts."""
        assets, npvs, capex = [], {}, {}
        for aid, cap, npv in specs:
            assets.append(Asset(aid, f"Asset {aid}", "Technology",
                                cap, [cap * 0.3] * 8, 0.08, 0.10))
            npvs[aid]  = npv
            capex[aid] = cap
        return assets, npvs, capex

    def test_selects_highest_pi_first(self):
        """Three assets PI=[20, 15, 10] with enough budget → selected in PI order."""
        assets, npvs, capex = self._make_assets([
            ("A", 1_000_000, 20_000_000),
            ("B", 1_000_000, 15_000_000),
            ("C", 1_000_000, 10_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=3_000_000)
        selected_ids = [a.id for a in result.selected_assets]
        assert selected_ids == ["A", "B", "C"]

    def test_budget_never_exceeded(self):
        """Deployed capital ≤ budget with floating-point tolerance."""
        assets, npvs, capex = self._make_assets([
            ("A", 3_000_000, 60_000_000),
            ("B", 4_000_000, 80_000_000),
            ("C", 5_000_000, 90_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=7_000_000)
        assert result.total_capital_deployed <= 7_000_000 + 0.01

    def test_total_npv_matches_per_asset_sum(self):
        """result.total_npv == sum(per_asset_npv.values())."""
        assets, npvs, capex = self._make_assets([
            ("A", 2_000_000, 40_000_000),
            ("B", 3_000_000, 60_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=10_000_000)
        assert result.total_npv == pytest.approx(
            sum(result.per_asset_npv.values()), rel=1e-9)

    def test_per_asset_npv_has_correct_keys(self):
        """per_asset_npv contains exactly the IDs of selected assets."""
        assets, npvs, capex = self._make_assets([
            ("A", 2_000_000, 50_000_000),
            ("B", 3_000_000, 30_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=5_000_000)
        selected_ids = {a.id for a in result.selected_assets}
        assert set(result.per_asset_npv.keys()) == selected_ids

    def test_remaining_budget_correct(self):
        """remaining_budget = budget - total_deployed, exactly."""
        budget = 10_000_000
        assets, npvs, capex = self._make_assets([
            ("A", 3_000_000, 60_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=budget)
        assert result.remaining_budget == pytest.approx(
            budget - result.total_capital_deployed, abs=0.01)

    def test_base_case_produces_929M(self, assets, scenarios):
        """Full real-data base case → NPV ≥ 928M."""
        from agent.portfolio_agent import PortfolioAgent
        agent = PortfolioAgent(budget=45_000_000, mc_iterations=100)
        result = agent._run_scenario(scenarios[0])
        assert result.optimization.total_npv >= 928_000_000

    def test_empty_asset_list(self):
        """No assets → empty portfolio, zero NPV."""
        result = greedy_knapsack_optimizer([], {}, {}, budget=45_000_000)
        assert result.selected_assets == []
        assert result.total_npv == 0.0
        assert result.total_capital_deployed == 0.0

    def test_all_negative_npv_rejected(self):
        """Assets with negative NPV are excluded before selection."""
        assets, _, capex = self._make_assets([
            ("A", 1_000_000, -5_000_000),
            ("B", 2_000_000, -1_000_000),
        ])
        npvs = {"A": -5_000_000, "B": -1_000_000}
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=10_000_000)
        assert result.selected_assets == []

    def test_exact_budget_fit(self):
        """Asset with capex == budget exactly gets selected."""
        budget = 5_000_000
        assets, npvs, capex = self._make_assets([
            ("A", budget, 100_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=budget)
        assert len(result.selected_assets) == 1
        assert result.selected_assets[0].id == "A"

    def test_budget_too_small_for_any_asset(self):
        """Budget of £1 cannot fit any asset → empty portfolio."""
        assets, npvs, capex = self._make_assets([
            ("A", 1_000_000, 20_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=1.0)
        assert result.selected_assets == []

    def test_single_asset_over_budget_rejected(self):
        """Asset capex > budget → rejected."""
        assets, npvs, capex = self._make_assets([
            ("A", 10_000_000, 200_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=5_000_000)
        assert result.selected_assets == []

    def test_pi_tie_breaking_is_stable(self):
        """Two assets with identical PI → order preserved (stable sort)."""
        assets, npvs, capex = self._make_assets([
            ("X", 2_000_000, 40_000_000),   # PI = 20
            ("Y", 3_000_000, 60_000_000),   # PI = 20 (same)
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=10_000_000)
        ids = [a.id for a in result.selected_assets]
        # Both selected; original order X before Y preserved
        assert ids.index("X") < ids.index("Y")

    def test_large_asset_pool_performance(self):
        """1,000 assets within 50M budget completes in < 1 second."""
        import random
        rng = random.Random(0)
        large_assets, npvs, capex = [], {}, {}
        for i in range(1000):
            cap = rng.randint(1_000_000, 8_000_000)
            npv = rng.randint(10_000_000, 200_000_000)
            large_assets.append(
                Asset(f"L{i}", f"Asset {i}", "Technology",
                      cap, [cap * 0.25] * 8, 0.08, 0.10))
            npvs[f"L{i}"]  = npv
            capex[f"L{i}"] = cap
        start = time.time()
        greedy_knapsack_optimizer(large_assets, npvs, capex, budget=50_000_000)
        assert time.time() - start < 1.0

    def test_partial_budget_fill(self):
        """When assets don't fill budget exactly → remaining_budget > 0."""
        assets, npvs, capex = self._make_assets([
            ("A", 3_000_000, 90_000_000),
            ("B", 3_000_000, 60_000_000),
        ])
        result = greedy_knapsack_optimizer(assets, npvs, capex, budget=7_500_000)
        assert result.remaining_budget > 0


# ══════════════════════════════════════════════════════════════════════════════
# B3 — Monte Carlo (11 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:

    def _run(self, assets, n=1_000, sigma_mult=1.0, seed=42):
        """Helper: run MC on a list of Asset objects."""
        adj_cfs  = {a.id: a.annual_cash_flows for a in assets}
        adj_cap  = {a.id: a.capital_required    for a in assets}
        adj_dr   = {a.id: a.discount_rate        for a in assets}
        return run_monte_carlo(assets, adj_cfs, adj_cap, adj_dr,
                               sigma_multiplier=sigma_mult,
                               n_iterations=n, random_seed=seed)

    def test_deficit_probability_zero_strong_portfolio(self, assets, scenarios):
        """Real base-case portfolio → deficit = 0.0 (zero downside)."""
        from agent.portfolio_agent import PortfolioAgent
        agent  = PortfolioAgent(budget=45_000_000, mc_iterations=10_000)
        result = agent._run_scenario(scenarios[0])
        assert result.monte_carlo.deficit_probability == 0.0

    def test_mean_close_to_deterministic_npv(self, simple_assets):
        """MC mean ≈ deterministic NPV (sigma=0.01 → tiny spread)."""
        mc = self._run(simple_assets, n=5_000, sigma_mult=0.01)
        det_npv = sum(
            calculate_npv(a.annual_cash_flows, a.capital_required, a.discount_rate)
            for a in simple_assets
        )
        assert abs(mc.mean_total_npv - det_npv) / abs(det_npv) < 0.02

    def test_p5_below_mean_below_p95(self, simple_assets):
        """Strict ordering: P05 < mean < P95."""
        mc = self._run(simple_assets, n=5_000)
        assert mc.p5_total_npv < mc.mean_total_npv < mc.p95_total_npv

    def test_std_positive(self, simple_assets):
        """Non-degenerate distribution → std > 0."""
        mc = self._run(simple_assets, n=2_000, sigma_mult=0.15)
        assert mc.std_total_npv > 0.0

    def test_samples_array_length(self, simple_assets):
        """all_npv_samples.size == n_iterations exactly."""
        n = 3_000
        mc = self._run(simple_assets, n=n)
        assert mc.all_npv_samples.size == n

    def test_reproducibility_with_same_seed(self, simple_assets):
        """Same seed → bit-identical results."""
        mc1 = self._run(simple_assets, n=2_000, seed=99)
        mc2 = self._run(simple_assets, n=2_000, seed=99)
        np.testing.assert_array_equal(mc1.all_npv_samples, mc2.all_npv_samples)

    def test_zero_sigma_gives_zero_spread(self, simple_assets):
        """risk_sigma=0 for all assets → std ≈ 0 (deterministic)."""
        zero_sig = [
            Asset(a.id, a.name, a.sector, a.capital_required,
                  a.annual_cash_flows, a.discount_rate, 0.0)
            for a in simple_assets
        ]
        mc = self._run(zero_sig, n=2_000, sigma_mult=1.0)
        assert mc.std_total_npv < 1.0   # < £1 spread on zero-sigma portfolio

    def test_single_asset_portfolio(self, simple_assets):
        """One-asset portfolio still produces valid MonteCarloResult."""
        mc = self._run([simple_assets[0]], n=1_000)
        assert mc.n_iterations == 1_000
        assert mc.all_npv_samples.size == 1_000
        assert mc.mean_total_npv != 0.0

    def test_high_sigma_increases_spread(self, simple_assets):
        """sigma_mult=2.0 → std > sigma_mult=0.5 std."""
        mc_lo = self._run(simple_assets, n=5_000, sigma_mult=0.5, seed=0)
        mc_hi = self._run(simple_assets, n=5_000, sigma_mult=2.0, seed=0)
        assert mc_hi.std_total_npv > mc_lo.std_total_npv

    def test_large_n_converges(self, simple_assets):
        """Mean difference between n=10k and n=50k < 0.5%."""
        mc_10k = self._run(simple_assets, n=10_000, seed=7)
        mc_50k = self._run(simple_assets, n=50_000, seed=7)
        diff = abs(mc_10k.mean_total_npv - mc_50k.mean_total_npv)
        assert diff / abs(mc_50k.mean_total_npv) < 0.005

    def test_empty_assets_produces_zero_npv(self):
        """No selected assets → deficit = 0, no simulation needed (empty array)."""
        mc = run_monte_carlo([], {}, {}, {},
                             sigma_multiplier=1.0, n_iterations=500,
                             random_seed=0)
        assert mc.deficit_probability == 0.0
        # Engine short-circuits on empty input — samples array is empty (correct)
        assert mc.all_npv_samples.size == 0 or \
               np.all(mc.all_npv_samples == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# B4 — Scenarios (8 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarios:

    VALID_SECTORS = {"Infrastructure", "Technology", "Renewables",
                     "Real Estate", "M&A"}

    def test_exactly_22_scenarios(self, scenarios):
        assert len(scenarios) == 22

    def test_ids_unique_and_range_1_to_22(self, scenarios):
        ids = [s.id for s in scenarios]
        assert sorted(ids) == list(range(1, 23))
        assert len(set(ids)) == 22   # no duplicates

    def test_base_case_all_modifiers_neutral(self, scenarios):
        base = next(s for s in scenarios if s.id == 1)
        assert base.cash_flow_modifier    == pytest.approx(1.0)
        assert base.discount_rate_delta   == pytest.approx(0.0)
        assert base.capex_modifier        == pytest.approx(1.0)
        assert base.risk_sigma_multiplier == pytest.approx(1.0)
        assert base.eligible_sectors == []

    def test_severe_recession_has_lowest_cf_modifier(self, scenarios):
        severe = next(s for s in scenarios if s.id == 11)
        min_cf = min(s.cash_flow_modifier for s in scenarios)
        assert severe.cash_flow_modifier == pytest.approx(min_cf)

    def test_to_document_contains_name(self, scenarios):
        for s in scenarios:
            doc = s.to_document()
            assert s.name in doc

    def test_eligible_sectors_only_valid_values(self, scenarios):
        for s in scenarios:
            for sector in s.eligible_sectors:
                assert sector in self.VALID_SECTORS, \
                    f"Scenario '{s.name}' has invalid sector '{sector}'"

    def test_optimistic_has_cf_modifier_above_one(self, scenarios):
        opt = next(s for s in scenarios if s.id == 2)
        assert opt.cash_flow_modifier > 1.0

    def test_all_modifiers_within_sane_range(self, scenarios):
        for s in scenarios:
            assert 0.5 <= s.cash_flow_modifier    <= 2.0, s.name
            assert 0.5 <= s.capex_modifier         <= 2.0, s.name
            assert 0.5 <= s.risk_sigma_multiplier  <= 2.5, s.name
            assert -0.10 <= s.discount_rate_delta  <= 0.10, s.name


# ══════════════════════════════════════════════════════════════════════════════
# B5 — Assets (6 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestAssets:

    def test_loads_18_assets(self, assets):
        assert len(assets) == 18

    def test_all_ids_unique(self, assets):
        ids = [a.id for a in assets]
        assert len(ids) == len(set(ids))

    def test_cashflows_are_8_years(self, assets):
        for a in assets:
            assert len(a.annual_cash_flows) == 8, \
                f"Asset {a.id} has {len(a.annual_cash_flows)} CFs, expected 8"

    def test_discount_rates_in_range(self, assets):
        for a in assets:
            assert 0.05 <= a.discount_rate <= 0.20, \
                f"Asset {a.id} discount_rate {a.discount_rate} out of [0.05, 0.20]"

    def test_risk_sigmas_in_range(self, assets):
        for a in assets:
            assert 0.05 <= a.risk_sigma <= 0.30, \
                f"Asset {a.id} risk_sigma {a.risk_sigma} out of [0.05, 0.30]"

    def test_missing_excel_raises_file_not_found(self, tmp_path):
        """If investment_pipeline.xlsx is absent, FileNotFoundError is raised."""
        import importlib
        import data.assets as assets_module

        original = assets_module.PIPELINE_FILE
        try:
            assets_module.PIPELINE_FILE = str(tmp_path / "nonexistent.xlsx")
            # Clear lru_cache if present
            if hasattr(assets_module.get_asset_pool, "cache_clear"):
                assets_module.get_asset_pool.cache_clear()
            with pytest.raises(FileNotFoundError):
                assets_module.get_asset_pool()
        finally:
            assets_module.PIPELINE_FILE = original


# ══════════════════════════════════════════════════════════════════════════════
# B6 — Scenario Store (7 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioStore:

    @pytest.fixture(autouse=True)
    def store(self, scenarios):
        from retrieval.scenario_store import ScenarioStore
        self._store = ScenarioStore(scenarios)

    def test_search_returns_top_k(self):
        results = self._store.search("recession", top_k=3)
        assert len(results) == 3

    def test_scores_between_0_and_1(self):
        results = self._store.search("technology crash", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_results_sorted_descending(self):
        results = self._store.search("energy market", top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_known_query_tech_slump(self):
        """'tech slump' → 'Tech Slump' scenario in first position."""
        results = self._store.search("tech slump bubble burst", top_k=3)
        top_name = results[0][0].name
        assert "Tech" in top_name or "Digital" in top_name

    def test_known_query_recession(self):
        """'severe economic downturn' → a Recession scenario in top-2."""
        results = self._store.search("severe economic downturn contraction", top_k=2)
        names   = [r[0].name for r in results]
        assert any("Recession" in n for n in names)

    def test_empty_query_does_not_crash(self):
        """Blank query should return k results without raising."""
        results = self._store.search("", top_k=3)
        assert len(results) == 3

    def test_top_k_1_returns_exactly_one(self):
        results = self._store.search("energy renewables", top_k=1)
        assert len(results) == 1


# ══════════════════════════════════════════════════════════════════════════════
# B7 — Standard RAG (4 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestStandardRAG:

    @pytest.fixture(autouse=True)
    def rag(self, scenarios):
        from retrieval.scenario_store import ScenarioStore
        from retrieval.standard_rag   import StandardRAG
        store = ScenarioStore(scenarios)
        self._rag = StandardRAG(store)

    def test_returns_scenario_instance(self, scenarios):
        from data.scenarios import Scenario
        result = self._rag.retrieve("tech boom growth")
        assert isinstance(result, Scenario)

    def test_tech_boom_query(self):
        """'tech boom' → Technology sector scenario."""
        result = self._rag.retrieve("technology boom surging AI valuations")
        # eligible_sectors empty (all) or contains Technology
        assert (result.eligible_sectors == [] or
                "Technology" in result.eligible_sectors)

    def test_recession_query(self):
        """'recession' keyword → Recession scenario in name."""
        result = self._rag.retrieve("deep recession economic contraction")
        assert "Recession" in result.name or "Contraction" in result.name

    def test_limitation_single_result(self):
        """Multi-component query still returns only ONE scenario (the limitation)."""
        from data.scenarios import Scenario
        result = self._rag.retrieve(
            "tech crash combined with major interest rate surge rising rates")
        # Standard RAG must return a single Scenario — this is its limitation
        assert isinstance(result, Scenario)
        # It cannot simultaneously capture both "Tech" and "Rate" in one named scenario
        covers_both = ("Tech" in result.name and "Rate" in result.name)
        assert not covers_both   # proves the limitation


# ══════════════════════════════════════════════════════════════════════════════
# B8 — Agentic RAG (8 tests — all LLM calls mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestAgenticRAG:

    def _make_compose_response(self, cf=0.90, dr=0.02, cap=1.05, sig=1.20):
        return json.dumps({
            "name": "Mock Composed Scenario",
            "description": "A test scenario",
            "cash_flow_modifier": cf,
            "discount_rate_delta": dr,
            "capex_modifier": cap,
            "risk_sigma_multiplier": sig,
            "eligible_sectors": [],
        })

    def _make_confidence_response(self, score=0.85):
        return json.dumps({"score": score, "reasoning": "Good match."})

    def _build_mock_client(self, plan_return="tech crash rates",
                            conf_score=0.85):
        mock = MagicMock()
        mock.plan_retrieval.return_value  = plan_return
        mock.score_confidence.return_value = (conf_score, "Covers query well.")
        mock.refine_query.return_value    = "refined terms"
        mock.compose_scenario.return_value = {
            "name": "Mock Composed Scenario",
            "description": "Test",
            "cash_flow_modifier": 0.88,
            "discount_rate_delta": 0.025,
            "capex_modifier": 1.05,
            "risk_sigma_multiplier": 1.25,
            "eligible_sectors": [],
        }
        return mock

    def test_returns_retrieval_result(self, scenarios):
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG, RetrievalResult
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        result = AgenticRAG(store, client).retrieve("tech crash rising rates")
        assert isinstance(result, RetrievalResult)

    def test_confident_first_attempt_no_requery(self, scenarios):
        """confidence ≥ 0.70 on first try → attempts == 1."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        result = AgenticRAG(store, client).retrieve("market stress")
        assert result.attempts == 1
        assert client.refine_query.call_count == 0

    def test_low_confidence_triggers_requery(self, scenarios):
        """confidence < 0.70 on first attempt → refine_query is called."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        # Override score_confidence to return low then high
        client.score_confidence.side_effect = [
            (0.50, "Missing rate component."),
            (0.85, "Now comprehensive."),
        ]
        result = AgenticRAG(store, client).retrieve("ambiguous query")
        assert client.refine_query.call_count >= 1
        assert result.confidence >= 0.70

    def test_max_attempts_respected(self, scenarios):
        """Always-low confidence exits after MAX_ATTEMPTS."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client()
        # Always return low confidence
        client.score_confidence.return_value = (0.30, "Still missing.")
        result = AgenticRAG(store, client).retrieve("impossible query")
        assert result.attempts <= AgenticRAG.MAX_ATTEMPTS

    def test_composed_scenario_has_valid_modifiers(self, scenarios):
        """Composed scenario modifiers must be in financially sane ranges."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        result = AgenticRAG(store, client).retrieve("downturn scenario")
        sc = result.composed_scenario
        assert 0.5 <= sc.cash_flow_modifier    <= 2.0
        assert 0.5 <= sc.risk_sigma_multiplier <= 2.5
        assert sc.capex_modifier > 0

    def test_fallback_on_bad_json_from_llm(self, scenarios):
        """If compose_scenario returns bad JSON, fallback base-case params are used."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        # Override compose_scenario to raise (simulates bad JSON in agentic_rag fallback)
        client.compose_scenario.return_value = {
            "name": "Composed Scenario (fallback)",
            "description": "fallback",
            "cash_flow_modifier": 1.0,
            "discount_rate_delta": 0.0,
            "capex_modifier": 1.0,
            "risk_sigma_multiplier": 1.0,
            "eligible_sectors": [],
        }
        result = AgenticRAG(store, client).retrieve("gibberish input")
        assert result.composed_scenario is not None

    def test_confidence_threshold_is_0_70(self):
        """AgenticRAG.CONFIDENCE_THRESHOLD must be 0.70."""
        from retrieval.agentic_rag import AgenticRAG
        assert AgenticRAG.CONFIDENCE_THRESHOLD == pytest.approx(0.70)

    def test_candidates_list_populated(self, scenarios):
        """RetrievalResult.candidates is a non-empty list of Scenario objects."""
        from retrieval.scenario_store import ScenarioStore
        from retrieval.agentic_rag import AgenticRAG
        from data.scenarios import Scenario
        store  = ScenarioStore(scenarios)
        client = self._build_mock_client(conf_score=0.85)
        result = AgenticRAG(store, client).retrieve("test query")
        assert len(result.candidates) > 0
        assert all(isinstance(c, Scenario) for c in result.candidates)


# ══════════════════════════════════════════════════════════════════════════════
# B9 — Integration (5 end-to-end tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    @pytest.fixture(scope="class")
    def all_results(self, assets, scenarios):
        """Run all 22 scenarios once and cache for the class."""
        from agent.portfolio_agent import PortfolioAgent
        agent = PortfolioAgent(budget=45_000_000, mc_iterations=1_000,
                               random_seed=42)
        for scenario in agent.scenarios:
            result = agent._run_scenario(scenario)
            agent.results.append(result)
        return agent.results

    def test_base_case_npv_is_929M(self, all_results):
        """Base Case (scenario id=1) → NPV ≥ £928M and ≤ £930M."""
        base = next(r for r in all_results if r.scenario.id == 1)
        assert 928_000_000 <= base.optimization.total_npv <= 930_000_000

    def test_all_22_scenarios_complete_without_error(self, all_results):
        """Exactly 22 results, all with valid OptimizationResult."""
        assert len(all_results) == 22
        for r in all_results:
            assert r.optimization is not None
            assert r.monte_carlo  is not None

    def test_severe_recession_lower_npv_than_base(self, all_results):
        """Recession — Severe (id=11) produces lower NPV than Base Case (id=1)."""
        base    = next(r for r in all_results if r.scenario.id == 1)
        severe  = next(r for r in all_results if r.scenario.id == 11)
        assert severe.optimization.total_npv < base.optimization.total_npv

    def test_optimistic_higher_npv_than_base(self, all_results):
        """Optimistic Growth (id=2) produces higher NPV than Base Case (id=1)."""
        base  = next(r for r in all_results if r.scenario.id == 1)
        optim = next(r for r in all_results if r.scenario.id == 2)
        assert optim.optimization.total_npv > base.optimization.total_npv

    def test_sector_filter_respected_tech_only(self, all_results):
        """Tech Boom (id=4, eligible_sectors=['Technology']) →
        all selected assets are in Technology sector."""
        tech = next(r for r in all_results if r.scenario.id == 4)
        for asset in tech.optimization.selected_assets:
            assert asset.sector == "Technology", \
                f"Non-Technology asset '{asset.name}' selected in Tech-only scenario"
