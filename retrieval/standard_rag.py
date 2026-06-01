"""
Standard RAG — the first retrieval architecture (kept for contrast).

Performs a direct TF-IDF vector lookup: embed the query, find the closest
scenario by cosine similarity, return it immediately with no reasoning step.

Limitation demonstrated: a query like "moderate tech slowdown with rising rates"
maps to exactly ONE scenario (whichever is closest in vector space), silently
dropping all other stressors mentioned. When the query is ambiguous or
multi-faceted, the retrieved context is incomplete and corrupts NPV calculations
downstream — the failure mode that motivated rebuilding the layer as AgenticRAG.
"""

from typing import TYPE_CHECKING

from retrieval.scenario_store import ScenarioStore

if TYPE_CHECKING:
    from data.scenarios import Scenario


class StandardRAG:
    def __init__(self, store: ScenarioStore) -> None:
        self.store = store

    def retrieve(self, query: str) -> "Scenario":
        """
        Direct vector lookup — returns the single closest scenario, no reasoning.
        Fast but unreliable for ambiguous or multi-component queries.
        """
        results = self.store.search(query, top_k=1)
        return results[0][0]
