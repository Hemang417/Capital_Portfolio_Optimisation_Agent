"""
Anthropic SDK wrapper for the agentic RAG retrieval layer.

Uses prompt caching on the 22 scenario documents — they are stable within a
session, so every call after the first hits the 5-minute ephemeral cache and
does not re-tokenise the corpus.

Models:
  PLANNING_MODEL  — claude-haiku-4-5-20251001  (fast, used for query planning
                    and refinement where speed matters more than precision)
  REASONING_MODEL — claude-sonnet-4-6          (accurate, used for confidence
                    scoring and scenario composition)
"""

import json
import os
from typing import TYPE_CHECKING, List, Tuple

import anthropic

if TYPE_CHECKING:
    from data.scenarios import Scenario

PLANNING_MODEL  = "claude-haiku-4-5-20251001"
REASONING_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a financial scenario analyst for a capital portfolio optimisation system.
You have access to a database of 22 economic scenarios, each defined by scalar modifiers applied to
a portfolio of strategic investment projects: cash_flow_modifier, discount_rate_delta,
capex_modifier, risk_sigma_multiplier, and optional sector filters.

Your role is to interpret natural language queries about economic conditions and map them accurately
to scenario parameters, so the downstream NPV optimiser and Monte Carlo engine can run correctly.
Precision matters: incorrect parameters corrupt NPV calculations and produce unreliable output."""


class ClaudeClient:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Set it with:  set ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def _corpus_block(self, scenarios: "List[Scenario]") -> dict:
        """Cached content block containing all scenario documents."""
        docs = "\n\n".join(
            f"[{i+1}] {s.to_document()}" for i, s in enumerate(scenarios)
        )
        return {
            "type": "text",
            "text": f"SCENARIO DATABASE:\n\n{docs}",
            "cache_control": {"type": "ephemeral"},
        }

    def _call(self, model: str, system: str, user_text: str,
              cached_blocks: list | None = None) -> str:
        content: list = []
        if cached_blocks:
            content.extend(cached_blocks)
        content.append({"type": "text", "text": user_text})

        response = self.client.messages.create(
            model=model,
            max_tokens=512,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def plan_retrieval(self, query: str, scenarios: "List[Scenario]") -> str:
        """
        Step 1 — Query planning.
        Returns a refined search string that captures the economic conditions
        in the query as terms likely to match scenario documents.
        """
        prompt = (
            f"User query: \"{query}\"\n\n"
            "Analyse what economic conditions, stressors, or market dynamics this query describes. "
            "Then produce a concise search string (under 30 words) using terms that would match "
            "relevant scenarios in the database above. Return ONLY the search string, no explanation."
        )
        return self._call(PLANNING_MODEL, _SYSTEM_PROMPT, prompt,
                          cached_blocks=[self._corpus_block(scenarios)])

    def score_confidence(
        self,
        query: str,
        candidates: "List[Tuple[Scenario, float]]",
        scenarios: "List[Scenario]",
    ) -> Tuple[float, str]:
        """
        Step 3 — Confidence scoring.
        Returns (score 0.0–1.0, reasoning string).
        Score < 0.70 triggers a re-query cycle.
        """
        cand_text = "\n".join(
            f"  - [{s.name}] sim={sim:.3f}: {s.description}"
            for s, sim in candidates
        )
        prompt = (
            f"User query: \"{query}\"\n\n"
            f"Retrieved candidates:\n{cand_text}\n\n"
            "Score how well these candidates collectively address the query. "
            "Consider: do they cover all the economic stressors mentioned? "
            "Are there missing components (e.g. the query mentions rate hikes but no rate scenario was retrieved)?\n\n"
            "Return valid JSON only — no markdown, no explanation outside JSON:\n"
            '{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
        )
        raw = self._call(REASONING_MODEL, _SYSTEM_PROMPT, prompt,
                         cached_blocks=[self._corpus_block(scenarios)])
        try:
            parsed = json.loads(raw)
            return float(parsed["score"]), str(parsed["reasoning"])
        except (json.JSONDecodeError, KeyError):
            return 0.5, "Could not parse confidence response — defaulting to 0.5"

    def refine_query(
        self,
        query: str,
        candidates: "List[Tuple[Scenario, float]]",
        score: float,
        reasoning: str,
        scenarios: "List[Scenario]",
    ) -> str:
        """
        Step 4 — Query refinement.
        Returns a new search string addressing the gaps identified in reasoning.
        """
        cand_names = ", ".join(s.name for s, _ in candidates)
        prompt = (
            f"User query: \"{query}\"\n"
            f"Previous retrieval returned: {cand_names}\n"
            f"Confidence score: {score:.2f}. Gap identified: {reasoning}\n\n"
            "Produce a new search string (under 30 words) targeting the missing components. "
            "Return ONLY the search string."
        )
        return self._call(PLANNING_MODEL, _SYSTEM_PROMPT, prompt,
                          cached_blocks=[self._corpus_block(scenarios)])

    def compose_scenario(
        self,
        query: str,
        candidates: "List[Tuple[Scenario, float]]",
        confidence: float,
        scenarios: "List[Scenario]",
    ) -> dict:
        """
        Step 5 — Scenario composition.
        Returns a dict of Scenario constructor kwargs derived from the query
        and the retrieved candidates.
        """
        cand_text = "\n".join(
            f"  - {s.name}: cf={s.cash_flow_modifier}, dr_delta={s.discount_rate_delta}, "
            f"capex={s.capex_modifier}, sigma={s.risk_sigma_multiplier}, "
            f"sectors={s.eligible_sectors or 'all'}"
            for s, _ in candidates
        )
        prompt = (
            f"User query: \"{query}\"\n\n"
            f"Best matching scenarios retrieved (confidence {confidence:.2f}):\n{cand_text}\n\n"
            "Compose a single custom scenario that accurately represents the economic conditions "
            "in the user's query. Blend the retrieved parameters proportionally where multiple "
            "stressors are present. Keep eligible_sectors as an empty list unless the query "
            "explicitly restricts to specific sectors.\n\n"
            "Return valid JSON only — no markdown:\n"
            '{"name": "<short name>", "description": "<one sentence>", '
            '"cash_flow_modifier": <float>, "discount_rate_delta": <float>, '
            '"capex_modifier": <float>, "risk_sigma_multiplier": <float>, '
            '"eligible_sectors": [<strings or empty>]}'
        )
        raw = self._call(REASONING_MODEL, _SYSTEM_PROMPT, prompt,
                         cached_blocks=[self._corpus_block(scenarios)])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: return base-case parameters
            return {
                "name": "Composed Scenario (fallback)",
                "description": query,
                "cash_flow_modifier": 1.0,
                "discount_rate_delta": 0.0,
                "capex_modifier": 1.0,
                "risk_sigma_multiplier": 1.0,
                "eligible_sectors": [],
            }
