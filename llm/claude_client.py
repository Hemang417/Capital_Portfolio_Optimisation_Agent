"""
LLM client for the agentic RAG retrieval layer.

Active implementation: Groq API (free tier)
  Models used:
    PLANNING_MODEL  — llama-3.1-8b-instant     (fast, query planning & refinement)
    REASONING_MODEL — llama-3.3-70b-versatile  (accurate, confidence scoring & composition)
  Get a free API key at: https://console.groq.com

Commented out: Anthropic SDK implementation (kept for reference)
  To switch back, uncomment the Anthropic block and comment the Groq block.
  Anthropic uses prompt caching (cache_control ephemeral) on the scenario corpus
  so re-query loop calls hit the 5-minute cache — see commented code below.
"""

import json
import os
from typing import TYPE_CHECKING, List, Tuple

# ── Active: Groq ──────────────────────────────────────────────────────────────
from groq import Groq

# ── Commented: Anthropic (kept for reference) ─────────────────────────────────
# import anthropic

if TYPE_CHECKING:
    from data.scenarios import Scenario

# ── Groq models ───────────────────────────────────────────────────────────────
PLANNING_MODEL  = "llama-3.1-8b-instant"       # fast, cheap — query planning & refinement
REASONING_MODEL = "llama-3.3-70b-versatile"    # accurate — confidence scoring & composition

# ── Anthropic models (commented) ──────────────────────────────────────────────
# PLANNING_MODEL  = "claude-haiku-4-5-20251001"
# REASONING_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a financial scenario analyst for a capital portfolio optimisation system.
You have access to a database of 22 economic scenarios, each defined by scalar modifiers applied to
a portfolio of strategic investment projects: cash_flow_modifier, discount_rate_delta,
capex_modifier, risk_sigma_multiplier, and optional sector filters.

Your role is to interpret natural language queries about economic conditions and map them accurately
to scenario parameters, so the downstream NPV optimiser and Monte Carlo engine can run correctly.
Precision matters: incorrect parameters corrupt NPV calculations and produce unreliable output."""


def _build_corpus(scenarios: "List[Scenario]") -> str:
    docs = "\n\n".join(
        f"[{i+1}] {s.to_document()}" for i, s in enumerate(scenarios)
    )
    return f"SCENARIO DATABASE:\n\n{docs}"


class ClaudeClient:
    """
    LLM client — currently backed by Groq.
    The class name is kept as ClaudeClient so no other files need changing.
    """

    def __init__(self) -> None:
        # ── Groq initialisation ───────────────────────────────────────────────
        api_key = (
            os.environ.get("GROQ_API_KEY")
            or _streamlit_secret("GROQ_API_KEY")
        )
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then set it with:  set GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key=api_key)

        # ── Anthropic initialisation (commented) ──────────────────────────────
        # api_key = os.environ.get("ANTHROPIC_API_KEY")
        # if not api_key:
        #     raise EnvironmentError("ANTHROPIC_API_KEY not set. set ANTHROPIC_API_KEY=sk-ant-...")
        # self.client = anthropic.Anthropic(api_key=api_key)

    def _call(self, model: str, system: str, user_text: str,
              corpus: str | None = None) -> str:
        # ── Groq call ─────────────────────────────────────────────────────────
        full_user = f"{corpus}\n\n{user_text}" if corpus else user_text
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": full_user},
            ],
        )
        return response.choices[0].message.content.strip()

        # ── Anthropic call (commented) ────────────────────────────────────────
        # content: list = []
        # if corpus:
        #     content.append({
        #         "type": "text",
        #         "text": corpus,
        #         "cache_control": {"type": "ephemeral"},   # 5-min cache on scenario corpus
        #     })
        # content.append({"type": "text", "text": user_text})
        # response = self.client.messages.create(
        #     model=model,
        #     max_tokens=512,
        #     system=[{"type": "text", "text": system,
        #              "cache_control": {"type": "ephemeral"}}],
        #     messages=[{"role": "user", "content": content}],
        # )
        # return response.content[0].text.strip()

    # ── Public API (identical interface for both backends) ────────────────────

    def plan_retrieval(self, query: str, scenarios: "List[Scenario]") -> str:
        """Step 1 — produce targeted search terms from the natural language query."""
        prompt = (
            f"User query: \"{query}\"\n\n"
            "Analyse what economic conditions, stressors, or market dynamics this query describes. "
            "Then produce a concise search string (under 30 words) using terms that would match "
            "relevant scenarios in the database above. Return ONLY the search string, no explanation."
        )
        return self._call(PLANNING_MODEL, _SYSTEM_PROMPT, prompt,
                          corpus=_build_corpus(scenarios))

    def score_confidence(
        self,
        query: str,
        candidates: "List[Tuple[Scenario, float]]",
        scenarios: "List[Scenario]",
    ) -> Tuple[float, str]:
        """Step 3 — score 0.0–1.0 how well candidates address the query."""
        cand_text = "\n".join(
            f"  - [{s.name}] sim={sim:.3f}: {s.description}"
            for s, sim in candidates
        )
        prompt = (
            f"User query: \"{query}\"\n\n"
            f"Retrieved candidates:\n{cand_text}\n\n"
            "Score how well these candidates collectively address the query. "
            "Consider: do they cover all economic stressors mentioned? "
            "Are there missing components?\n\n"
            "Return valid JSON only — no markdown, no text outside JSON:\n"
            '{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
        )
        raw = self._call(REASONING_MODEL, _SYSTEM_PROMPT, prompt,
                         corpus=_build_corpus(scenarios))
        try:
            # Strip markdown fences if model wraps in ```json ... ```
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(clean)
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
        """Step 4 — produce refined search terms targeting identified gaps."""
        cand_names = ", ".join(s.name for s, _ in candidates)
        prompt = (
            f"User query: \"{query}\"\n"
            f"Previous retrieval returned: {cand_names}\n"
            f"Confidence score: {score:.2f}. Gap identified: {reasoning}\n\n"
            "Produce a new search string (under 30 words) targeting the missing components. "
            "Return ONLY the search string."
        )
        return self._call(PLANNING_MODEL, _SYSTEM_PROMPT, prompt,
                          corpus=_build_corpus(scenarios))

    def compose_scenario(
        self,
        query: str,
        candidates: "List[Tuple[Scenario, float]]",
        confidence: float,
        scenarios: "List[Scenario]",
    ) -> dict:
        """Step 5 — blend retrieved candidates into a single custom Scenario dict."""
        cand_text = "\n".join(
            f"  - {s.name}: cf={s.cash_flow_modifier}, dr_delta={s.discount_rate_delta}, "
            f"capex={s.capex_modifier}, sigma={s.risk_sigma_multiplier}, "
            f"sectors={s.eligible_sectors or 'all'}"
            for s, _ in candidates
        )
        prompt = (
            f"User query: \"{query}\"\n\n"
            f"Best matching scenarios (confidence {confidence:.2f}):\n{cand_text}\n\n"
            "Compose a single custom scenario representing the economic conditions in the query. "
            "Blend parameters proportionally where multiple stressors are present. "
            "Keep eligible_sectors empty unless the query explicitly restricts to specific sectors.\n\n"
            "Return valid JSON only — no markdown:\n"
            '{"name": "<short name>", "description": "<one sentence>", '
            '"cash_flow_modifier": <float>, "discount_rate_delta": <float>, '
            '"capex_modifier": <float>, "risk_sigma_multiplier": <float>, '
            '"eligible_sectors": []}'
        )
        raw = self._call(REASONING_MODEL, _SYSTEM_PROMPT, prompt,
                         corpus=_build_corpus(scenarios))
        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {
                "name": "Composed Scenario (fallback)",
                "description": query,
                "cash_flow_modifier": 1.0,
                "discount_rate_delta": 0.0,
                "capex_modifier": 1.0,
                "risk_sigma_multiplier": 1.0,
                "eligible_sectors": [],
            }


def _streamlit_secret(key: str) -> str | None:
    """Read from Streamlit secrets if running inside a Streamlit app."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None
