"""
Streamlit web interface for the Capital Portfolio Optimisation Agent.
Deploys on Streamlit Community Cloud — set GROQ_API_KEY in app secrets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Capital Portfolio Optimisation Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_assets():
    from data.assets import get_asset_pool
    return get_asset_pool()

@st.cache_data(show_spinner=False)
def load_scenarios():
    from data.scenarios import get_all_scenarios
    return get_all_scenarios()

def run_all_scenarios(budget: float, mc_iters: int, seed: int):
    from agent.portfolio_agent import PortfolioAgent
    agent = PortfolioAgent(budget=budget, mc_iterations=mc_iters, random_seed=seed)
    for scenario in agent.scenarios:
        result = agent._run_scenario(scenario)
        agent.results.append(result)
    return agent.results

def results_to_df(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Scenario": r.scenario.name,
            "Capital Deployed (£M)": round(r.optimization.total_capital_deployed / 1e6, 2),
            "Budget Used (%)": round(r.optimization.total_capital_deployed / r.optimization.total_capital_deployed * 100, 1)
            if r.optimization.total_capital_deployed > 0 else 0,
            "Strategic Value (£M)": round(r.optimization.total_npv / 1e6, 1),
            "Return Multiplier": round(r.optimization.total_npv / max(r.optimization.total_capital_deployed, 1), 2),
            "MC Mean NPV (£M)": round(r.monte_carlo.mean_total_npv / 1e6, 1),
            "P05 NPV (£M)": round(r.monte_carlo.p5_total_npv / 1e6, 1),
            "Deficit Prob (%)": round(r.monte_carlo.deficit_probability * 100, 4),
            "Projects Selected": len(r.optimization.selected_assets),
        })
    return pd.DataFrame(rows)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")
    budget_m = st.slider("Capital Budget (£M)", min_value=10, max_value=200,
                         value=45, step=5)
    budget = budget_m * 1_000_000

    mc_iters = st.select_slider(
        "Monte Carlo Iterations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    seed = st.number_input("Random Seed", value=42, step=1)

    st.divider()
    st.caption("**Agentic RAG Mode** requires a Groq API key.")
    groq_key = st.text_input("GROQ_API_KEY (optional)", type="password",
                              help="Get a free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    st.caption(
        "MSc Financial Data Analytics · "
        "[GitHub](https://github.com/Hemang417/Capital_Portfolio_Optimisation_Agent)"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Capital Portfolio Optimisation Agent")
st.caption(
    "Autonomous capital allocation across 22 economic scenarios · "
    "Budget-constrained NPV optimiser · 10,000-iteration Monte Carlo stress test"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🏦 22-Scenario Analysis", "🤖 Natural Language Query", "📖 About"]
)

# ── Tab 1: 22-Scenario Analysis ───────────────────────────────────────────────
with tab1:
    st.subheader("Run all 22 economic scenarios")
    st.write(
        "The agent applies each scenario's modifiers (cash flow multiplier, "
        "discount rate delta, capex multiplier, volatility) to the full project "
        "pipeline, selects the optimal portfolio within your budget using a "
        "greedy NPV/capex ranker, then stress-tests with Monte Carlo."
    )

    if st.button("▶ Run Analysis", type="primary", key="run_all"):
        with st.spinner(f"Running 22 scenarios · {mc_iters:,} MC iterations each …"):
            try:
                results = run_all_scenarios(budget, mc_iters, int(seed))
                st.session_state["results"] = results
            except Exception as e:
                st.error(f"Error: {e}")

    if "results" in st.session_state:
        results = st.session_state["results"]
        df = results_to_df(results)
        base = df[df["Scenario"] == "Base Case"].iloc[0]

        # ── Headline metrics ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Wave-1 Roadmap — Base Case")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategic Value",
                  f"£{base['Strategic Value (£M)']:,.0f}M")
        c2.metric("Capital Deployed",
                  f"£{base['Capital Deployed (£M)']:,.0f}M")
        c3.metric("Return Multiplier",
                  f"{base['Return Multiplier']:.2f}×")
        c4.metric("Deficit Probability",
                  f"{base['Deficit Prob (%)']:.4f}%")

        # ── Base case portfolio ───────────────────────────────────────────────
        base_result = next(r for r in results if r.scenario.id == 1)
        st.divider()
        st.subheader("Selected Portfolio (Base Case)")
        portfolio_rows = []
        for asset in base_result.optimization.selected_assets:
            pi = base_result.optimization.profitability_indices.get(asset.id, 0)
            npv = base_result.optimization.total_npv / len(base_result.optimization.selected_assets)
            portfolio_rows.append({
                "Project": asset.name,
                "Sector": asset.sector,
                "Capex (£M)": round(asset.capital_required / 1e6, 2),
                "PI": round(pi, 2),
            })
        st.dataframe(pd.DataFrame(portfolio_rows), use_container_width=True, hide_index=True)

        # ── Strategic value chart ─────────────────────────────────────────────
        st.divider()
        st.subheader("Strategic Value Across All 22 Scenarios (£M)")
        chart_df = df[["Scenario", "Strategic Value (£M)"]].set_index("Scenario")
        st.bar_chart(chart_df, color="#2196F3")

        # ── Full results table ────────────────────────────────────────────────
        st.divider()
        st.subheader("Full 22-Scenario Results Table")
        st.dataframe(
            df.style.background_gradient(
                subset=["Strategic Value (£M)", "Return Multiplier"], cmap="Blues"
            ).format({
                "Capital Deployed (£M)": "{:.2f}",
                "Strategic Value (£M)": "{:.1f}",
                "Return Multiplier": "{:.2f}×",
                "Deficit Prob (%)": "{:.4f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Download ──────────────────────────────────────────────────────────
        csv = df.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv,
                           file_name="wave1_roadmap.csv", mime="text/csv")

# ── Tab 2: Natural Language Query ─────────────────────────────────────────────
with tab2:
    st.subheader("Agentic RAG — Natural Language Query Mode")
    st.write(
        "Type a description of an economic scenario in plain English. "
        "The agent plans retrieval, scores confidence, re-queries if needed, "
        "then composes custom scenario parameters and runs the full financial pipeline."
    )

    example_queries = [
        "What happens to our portfolio in a tech market crash with rising interest rates?",
        "Simulate a severe recession with supply chain disruptions",
        "How does the portfolio perform under an energy transition boom?",
        "Model geopolitical stress affecting infrastructure investments",
    ]
    selected_example = st.selectbox("Or pick an example query:", ["— type your own —"] + example_queries)

    query_input = st.text_area(
        "Your query",
        value="" if selected_example == "— type your own —" else selected_example,
        height=80,
        placeholder='e.g. "What if there is a moderate tech slowdown with rising rates?"',
    )

    if st.button("▶ Run Query", type="primary", key="run_query"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is required for this mode. Enter it in the sidebar.")
        else:
            from retrieval.scenario_store import ScenarioStore
            from retrieval.standard_rag import StandardRAG
            from retrieval.agentic_rag import AgenticRAG
            from llm.claude_client import ClaudeClient
            from agent.portfolio_agent import PortfolioAgent

            agent = PortfolioAgent(budget=budget, mc_iterations=mc_iters,
                                   random_seed=int(seed))
            store  = ScenarioStore(agent.scenarios)

            # Standard RAG — show limitation first
            std_scenario = StandardRAG(store).retrieve(query_input)
            st.info(
                f"**Standard RAG** mapped to: **'{std_scenario.name}'**  \n"
                f"*(direct TF-IDF match — single scenario, no reasoning)*"
            )

            # Agentic RAG
            with st.spinner("Agentic RAG: planning retrieval …"):
                try:
                    client = ClaudeClient()
                    rag_result = AgenticRAG(store, client).retrieve(query_input)
                except EnvironmentError as e:
                    st.error(str(e))
                    st.stop()

            st.success(
                f"**Agentic RAG** composed: **'{rag_result.composed_scenario.name}'**  \n"
                f"Confidence: **{rag_result.confidence:.2f}** · "
                f"Retrieval attempts: **{rag_result.attempts}**"
            )
            st.caption(f"Reasoning: {rag_result.reasoning}")
            st.caption(f"Candidates used: {', '.join(s.name for s in rag_result.candidates)}")

            # Composed scenario parameters
            with st.expander("Composed scenario parameters"):
                sc = rag_result.composed_scenario
                params_df = pd.DataFrame([{
                    "cash_flow_modifier":   sc.cash_flow_modifier,
                    "discount_rate_delta":  sc.discount_rate_delta,
                    "capex_modifier":       sc.capex_modifier,
                    "risk_sigma_multiplier":sc.risk_sigma_multiplier,
                    "eligible_sectors":     ", ".join(sc.eligible_sectors) or "All",
                }])
                st.dataframe(params_df, use_container_width=True, hide_index=True)

            # Run financial pipeline
            with st.spinner("Running NPV optimiser and Monte Carlo …"):
                run_result = agent._run_scenario(rag_result.composed_scenario)

            opt = run_result.optimization
            mc  = run_result.monte_carlo

            st.divider()
            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategic Value",   f"£{opt.total_npv/1e6:,.1f}M")
            c2.metric("Capital Deployed",  f"£{opt.total_capital_deployed/1e6:,.1f}M")
            c3.metric("Return Multiplier", f"{opt.total_npv/max(opt.total_capital_deployed,1):.2f}×")
            c4.metric("Deficit Prob",      f"{mc.deficit_probability*100:.4f}%")

            mc_cols = st.columns(3)
            mc_cols[0].metric("MC Mean NPV", f"£{mc.mean_total_npv/1e6:,.1f}M")
            mc_cols[1].metric("P05 (downside)", f"£{mc.p5_total_npv/1e6:,.1f}M")
            mc_cols[2].metric("P95 (upside)",   f"£{mc.p95_total_npv/1e6:,.1f}M")

            # Selected portfolio
            st.subheader("Selected Portfolio")
            p_rows = [{
                "Project": a.name,
                "Sector": a.sector,
                "Capex (£M)": round(a.capital_required / 1e6, 2),
                "PI": round(opt.profitability_indices.get(a.id, 0), 2),
            } for a in opt.selected_assets]
            if p_rows:
                st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No assets selected — all projects had negative NPV under this scenario.")

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
**Capital Portfolio Optimisation Agent** — MSc Financial Data Analytics

A modular financial agent that autonomously selects optimal asset allocations across
22 distinct economic scenarios using net present value ranking, a budget-constrained
greedy optimiser, and a 10,000-iteration Monte Carlo shock engine.

---

### Architecture

```
investment_pipeline.xlsx  (data source)
        ↓
data/assets.py            reads project pipeline
data/scenarios.py         22 scenario definitions
        ↓
engine/npv_engine.py      NPV / IRR calculations (NumPy)
engine/optimizer.py       greedy knapsack, sorted by PI = NPV / capex
engine/monte_carlo.py     vectorised 10,000-iteration Monte Carlo
        ↓
agent/portfolio_agent.py  orchestrates all 22 scenarios autonomously
        ↓
retrieval/                agentic RAG layer (natural language → Scenario)
  scenario_store.py       TF-IDF vector index
  standard_rag.py         direct vector lookup (first version, kept for contrast)
  agentic_rag.py          plan → retrieve → confidence score → re-query → compose
llm/claude_client.py      Groq API (free) · Anthropic code kept, commented
```

### Key Results (Base Case)
| Metric | Value |
|---|---|
| Capital budget | £45,000,000 |
| Strategic value (Wave-1 NPV) | £929,000,000 |
| Return multiplier | 20.64× |
| Deficit probability | 0.0000% across all 22 scenarios |

### Tech Stack
- **Python** · NumPy · pandas · scikit-learn
- **Groq API** (LLaMA 3.3 70B) for agentic RAG
- **Streamlit** for web interface

### Source Code
[github.com/Hemang417/Capital_Portfolio_Optimisation_Agent](https://github.com/Hemang417/Capital_Portfolio_Optimisation_Agent)
""")
