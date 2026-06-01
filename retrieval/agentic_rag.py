"""
Agentic RAG — the rebuilt retrieval architecture.

Replaces direct vector lookup with a reasoning-driven pipeline:

  Step 1  plan_retrieval    LLM analyses the query and produces targeted
                            search terms rather than using the raw query.

  Step 2  vector search     ScenarioStore retrieves top-3 candidates using
                            the planned search terms.

  Step 3  score_confidence  LLM evaluates how well the candidates cover the
                            query's economic conditions. Returns 0.0–1.0.

  Step 4  re-query loop     If confidence < CONFIDENCE_THRESHOLD, LLM refines
                            the search terms targeting the identified gaps and
                            the cycle repeats. Max MAX_ATTEMPTS iterations.

  Step 5  compose_scenario  LLM blends parameters from retrieved candidates
                            into a single custom Scenario object that the
                            financial engine can run directly.

Only when confidence is sufficient (or MAX_ATTEMPTS is exhausted) does the
agent proceed to the financial calculation layer. This prevents corrupted
context from reaching the NPV engine.
"""

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from retrieval.scenario_store import ScenarioStore

if TYPE_CHECKING:
    from data.scenarios import Scenario
    from llm.claude_client import ClaudeClient


@dataclass
class RetrievalResult:
    composed_scenario: "Scenario"
    candidates: "List[Scenario]"
    confidence: float
    attempts: int
    reasoning: str


class AgenticRAG:
    CONFIDENCE_THRESHOLD: float = 0.70
    MAX_ATTEMPTS: int = 3

    def __init__(self, store: ScenarioStore, client: "ClaudeClient") -> None:
        self.store = store
        self.client = client

    def retrieve(self, query: str) -> RetrievalResult:
        from data.scenarios import get_all_scenarios
        scenarios = get_all_scenarios()

        # Step 1 — LLM plans what to search for
        search_terms = self.client.plan_retrieval(query, scenarios)

        # Step 2 — Vector search on planned terms
        candidates = self.store.search(search_terms, top_k=3)

        # Step 3 — LLM scores confidence on retrieved candidates
        confidence, reasoning = self.client.score_confidence(query, candidates, scenarios)

        attempts = 1

        # Step 4 — Re-query loop when confidence is below threshold
        for _ in range(self.MAX_ATTEMPTS - 1):
            if confidence >= self.CONFIDENCE_THRESHOLD:
                break
            refined = self.client.refine_query(
                query, candidates, confidence, reasoning, scenarios
            )
            candidates = self.store.search(refined, top_k=3)
            confidence, reasoning = self.client.score_confidence(
                query, candidates, scenarios
            )
            attempts += 1

        # Step 5 — Compose a custom Scenario from retrieved context
        scenario = self._build_scenario(
            self.client.compose_scenario(query, candidates, confidence, scenarios)
        )

        return RetrievalResult(
            composed_scenario=scenario,
            candidates=[s for s, _ in candidates],
            confidence=confidence,
            attempts=attempts,
            reasoning=reasoning,
        )

    @staticmethod
    def _build_scenario(params: dict) -> "Scenario":
        from data.scenarios import Scenario
        return Scenario(
            id=0,
            name=params.get("name", "Composed Scenario"),
            description=params.get("description", "LLM-composed scenario"),
            cash_flow_modifier=float(params.get("cash_flow_modifier", 1.0)),
            discount_rate_delta=float(params.get("discount_rate_delta", 0.0)),
            capex_modifier=float(params.get("capex_modifier", 1.0)),
            risk_sigma_multiplier=float(params.get("risk_sigma_multiplier", 1.0)),
            eligible_sectors=list(params.get("eligible_sectors", [])),
        )
