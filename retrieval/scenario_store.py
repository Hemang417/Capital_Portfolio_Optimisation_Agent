"""
TF-IDF vector index over the 22 scenario documents.
Used by both StandardRAG (direct lookup) and AgenticRAG (LLM-planned lookup).
"""

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from data.scenarios import Scenario


class ScenarioStore:
    def __init__(self, scenarios: "List[Scenario]") -> None:
        self.scenarios = scenarios
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        docs = [s.to_document() for s in scenarios]
        self._matrix = self._vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3) -> "List[Tuple[Scenario, float]]":
        """
        Embed the query and return the top-k most similar scenarios
        as [(Scenario, cosine_similarity_score), ...] sorted descending.
        """
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.scenarios[i], float(sims[i])) for i in top_indices]
