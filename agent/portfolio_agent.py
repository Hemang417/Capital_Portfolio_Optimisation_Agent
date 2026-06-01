from typing import Dict, List, Tuple

from data.assets import Asset, get_asset_pool
from data.scenarios import Scenario, get_all_scenarios
from engine.monte_carlo import MonteCarloResult, run_monte_carlo
from engine.npv_engine import calculate_npv
from engine.optimizer import BUDGET_GBP, OptimizationResult, greedy_knapsack_optimizer
from models import ScenarioRunResult
from reporting.console_reporter import print_scenario_report, print_wave1_summary
from reporting.file_reporter import save_results_json, save_wave1_csv


class PortfolioAgent:
    """
    Modular financial agent that separates calculation logic from raw data.

    Processes all 22 operational scenarios autonomously — no manual
    reconfiguration between runs.  For each scenario the agent:
      1. Applies scenario modifiers to the asset pool (CF, capex, discount rate)
      2. Filters by sector mandate (if applicable)
      3. Runs the greedy NPV-ranked capital allocator within the 45M GBP budget
      4. Stress-tests the selected portfolio with a 10,000-iteration Monte Carlo
      5. Reports results to console and writes JSON / CSV output files
    """

    def __init__(
        self,
        budget: float = BUDGET_GBP,
        mc_iterations: int = 10_000,
        random_seed: int = 42,
    ) -> None:
        self.budget = budget
        self.mc_iterations = mc_iterations
        self.random_seed = random_seed
        self.assets: List[Asset] = get_asset_pool()
        self.scenarios: List[Scenario] = get_all_scenarios()
        self.results: List[ScenarioRunResult] = []

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop — iterates all 22 scenarios without manual reconfiguration."""
        n = len(self.scenarios)
        print(f"\n{'=' * 72}")
        print(f"  Capital Portfolio Optimisation Agent")
        print(f"  Budget: £{self.budget:,.0f}  |  Scenarios: {n}  |  MC iterations: {self.mc_iterations:,}")
        print(f"{'=' * 72}\n")

        for scenario in self.scenarios:
            result = self._run_scenario(scenario)
            self.results.append(result)
            print_scenario_report(result)

        self._generate_wave1_summary()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _run_scenario(self, scenario: Scenario) -> ScenarioRunResult:
        # Filter asset pool by sector mandate
        candidates = (
            self.assets
            if not scenario.eligible_sectors
            else [a for a in self.assets if a.sector in scenario.eligible_sectors]
        )

        # Apply scenario modifiers and compute adjusted NPV for each candidate
        adj_npvs, adj_capex, adj_cfs, adj_dr = self._apply_scenario(candidates, scenario)

        # Run greedy budget-constrained optimiser
        optimization = greedy_knapsack_optimizer(
            candidates, adj_npvs, adj_capex, self.budget
        )

        # Run Monte Carlo on the selected portfolio only
        selected = optimization.selected_assets
        mc_result = run_monte_carlo(
            selected_assets=selected,
            adjusted_cfs={a.id: adj_cfs[a.id] for a in selected},
            adjusted_capex={a.id: adj_capex[a.id] for a in selected},
            adjusted_discount_rates={a.id: adj_dr[a.id] for a in selected},
            sigma_multiplier=scenario.risk_sigma_multiplier,
            n_iterations=self.mc_iterations,
            random_seed=self.random_seed,
        )

        return ScenarioRunResult(scenario, optimization, mc_result)

    def _apply_scenario(
        self,
        assets: List[Asset],
        scenario: Scenario,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]], Dict[str, float]]:
        """Return (adj_npvs, adj_capex, adj_cfs, adj_dr) for each asset."""
        adj_npvs: Dict[str, float] = {}
        adj_capex: Dict[str, float] = {}
        adj_cfs: Dict[str, List[float]] = {}
        adj_dr: Dict[str, float] = {}

        for asset in assets:
            cfs = [cf * scenario.cash_flow_modifier for cf in asset.annual_cash_flows]
            capex = asset.capital_required * scenario.capex_modifier
            dr = asset.discount_rate + scenario.discount_rate_delta

            adj_cfs[asset.id] = cfs
            adj_capex[asset.id] = capex
            adj_dr[asset.id] = dr
            adj_npvs[asset.id] = calculate_npv(cfs, capex, dr)

        return adj_npvs, adj_capex, adj_cfs, adj_dr

    def run_query(self, query: str) -> None:
        """
        Natural language query mode — agentic RAG resolves the query into a
        Scenario object, then the existing financial pipeline runs unchanged.

        Shows the Standard RAG result first to illustrate the failure mode
        (single vector match, no reasoning), then the Agentic RAG result
        (planned retrieval → confidence scoring → re-query if needed →
        composed scenario).  Only the Agentic RAG scenario is passed to the
        financial engine.
        """
        from llm.claude_client import ClaudeClient
        from retrieval.agentic_rag import AgenticRAG
        from retrieval.scenario_store import ScenarioStore
        from retrieval.standard_rag import StandardRAG

        print(f"\n{'=' * 72}")
        print(f"  Agentic RAG — Natural Language Query Mode")
        print(f"  Budget: £{self.budget:,.0f}  |  MC iterations: {self.mc_iterations:,}")
        print(f"{'=' * 72}")
        print(f"\n  Query: \"{query}\"\n")

        store  = ScenarioStore(self.scenarios)
        client = ClaudeClient()

        # Standard RAG — direct vector match, no reasoning (demonstrates limitation)
        std_scenario = StandardRAG(store).retrieve(query)
        print(f"  [Standard RAG]  Mapped to : '{std_scenario.name}'")
        print(f"                  (direct TF-IDF match — single scenario, no reasoning)\n")

        # Agentic RAG — plan → retrieve → score → re-query → compose
        print("  [Agentic  RAG]  Planning retrieval ...", flush=True)
        rag_result = AgenticRAG(store, client).retrieve(query)

        print(f"  [Agentic  RAG]  Composed  : '{rag_result.composed_scenario.name}'")
        print(f"                  Confidence: {rag_result.confidence:.2f}"
              f"  |  Retrieval attempts: {rag_result.attempts}")
        print(f"                  Reasoning : {rag_result.reasoning}")
        print(f"                  Candidates: "
              f"{', '.join(s.name for s in rag_result.candidates)}\n")

        # Run the full financial pipeline with the composed scenario
        run_result = self._run_scenario(rag_result.composed_scenario)
        self.results.append(run_result)
        print_scenario_report(run_result)

    def _generate_wave1_summary(self) -> None:
        base_case = next(r for r in self.results if r.scenario.id == 1)
        print_wave1_summary(self.results, base_case)
        save_results_json(self.results)
        save_wave1_csv(self.results)
