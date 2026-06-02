"""
Capital Portfolio Optimisation Agent — Streamlit Dashboard
Bloomberg-style dark financial interface.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Capital Portfolio Optimisation Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar hidden by default on mobile
)

# ── Design constants ──────────────────────────────────────────────────────────
CYAN   = "#00d4ff"
GREEN  = "#00ff88"
DANGER = "#ff4444"
AMBER  = "#ffaa00"

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c0d8f0", size=11),
    margin=dict(l=40, r=10, t=44, b=40),
    autosize=True,
)

SECTOR_COLORS = {
    "Technology":     CYAN,
    "Renewables":     GREEN,
    "Infrastructure": AMBER,
    "Real Estate":    "#aa88ff",
    "M&A":            "#ff6688",
}

AREA_PALETTE = [CYAN, GREEN, AMBER, "#aa88ff", "#ff6688",
                "#ff8844", "#88aaff", "#44ffcc", "#ffcc44", "#cc44ff"]

# ── CSS injection (mobile-first) ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
.stApp { background-color: #0a0e1a; color: #e0e6f0; }

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8899aa !important;
    font-size: 0.85rem;
}

/* Tighten padding on all screens; extra-tight on mobile */
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px;
}

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a0e1a 60%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.4rem 1.2rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 50px rgba(0, 212, 255, 0.07);
}
.hero-title {
    font-size: clamp(1.2rem, 5vw, 2rem);
    font-weight: 700;
    color: #00d4ff;
    margin: 0 0 0.3rem 0;
    letter-spacing: 0.01em;
    word-break: break-word;
}
.hero-subtitle {
    font-size: clamp(0.65rem, 2.5vw, 0.82rem);
    color: #5577aa;
    margin: 0 0 1rem 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1.6;
}

/* KPI grid: 2 columns on mobile, 4 on desktop */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem;
}
@media (min-width: 640px) {
    .kpi-grid { grid-template-columns: repeat(4, 1fr); gap: 1rem; }
    .hero-section { padding: 2rem 2.5rem 1.8rem; }
}

.kpi-card {
    background: linear-gradient(135deg, #0f1923 0%, #131f2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 7px;
    padding: 0.75rem 0.9rem;
    transition: box-shadow 0.2s, border-color 0.2s;
}
@media (min-width: 640px) {
    .kpi-card { padding: 1rem 1.2rem; }
}
.kpi-card:hover {
    box-shadow: 0 0 22px rgba(0, 212, 255, 0.18);
    border-color: #00d4ff;
}
.kpi-label {
    font-size: 0.63rem;
    color: #5577aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: clamp(1.1rem, 4vw, 1.65rem);
    font-weight: 700;
    color: #00d4ff;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}
.kpi-value.green  { color: #00ff88; }
.kpi-value.white  { color: #e0e6f0; }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background-color: transparent;
    border-bottom: 2px solid #1e2a3a;
    gap: 0;
    margin-bottom: 0.5rem;
    overflow-x: auto;           /* allow horizontal scroll on tiny screens */
    -webkit-overflow-scrolling: touch;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #5577aa;
    font-size: clamp(0.72rem, 2.5vw, 0.85rem);
    font-weight: 500;
    padding: 0.5rem 0.8rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
    white-space: nowrap;
}
@media (min-width: 640px) {
    [data-testid="stTabs"] button[role="tab"] { padding: 0.55rem 1.4rem; }
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #00d4ff !important;
    background: rgba(0,212,255,0.04) !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1923 0%, #131f2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 7px;
    padding: 0.6rem 0.75rem;
}
[data-testid="stMetricLabel"] {
    color: #5577aa !important;
    font-size: clamp(0.6rem, 2vw, 0.72rem) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: clamp(1rem, 3.5vw, 1.35rem) !important;
    font-weight: 700 !important;
}

/* ── Section headers ──────────────────────────────────────────────────────── */
h2, h3 {
    color: #c0d8f0 !important;
    font-weight: 600 !important;
    border-left: 3px solid #00d4ff;
    padding-left: 0.7rem;
    margin-top: 1.2rem !important;
    font-size: clamp(0.95rem, 3vw, 1.2rem) !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    overflow: hidden;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* ── RAG comparison cards ─────────────────────────────────────────────────── */
.rag-card {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 7px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.rag-card.agent   { border-color: #00ff88; box-shadow: 0 0 18px rgba(0,255,136,0.09); }
.rag-card.standard{ border-color: #ff4444; box-shadow: 0 0 18px rgba(255,68,68,0.09); }
.rag-tag { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; font-weight: 600; }
.rag-tag.agent    { color: #00ff88; }
.rag-tag.standard { color: #ff4444; }
.rag-name { font-size: clamp(0.9rem, 3vw, 1.05rem); font-weight: 600; color: #e0e6f0; }
.rag-meta { font-size: 0.75rem; color: #5577aa; margin-top: 0.3rem; line-height: 1.5; }

/* ── Params / arch cards ──────────────────────────────────────────────────── */
.params-card {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 7px;
    padding: 0.85rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: clamp(0.72rem, 2.5vw, 0.85rem);
    color: #8fc8e8;
    line-height: 1.9;
    margin-bottom: 1rem;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
.arch-card {
    background: #0f1923;
    border: 1px solid #1e3a5f;
    border-radius: 7px;
    padding: 1rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: clamp(0.65rem, 2vw, 0.8rem);
    color: #8fc8e8;
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* ── Badge chips ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    background: #131f2e;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    color: #00d4ff;
    margin: 0.2rem 0.12rem;
    font-weight: 500;
}

/* ── Misc ─────────────────────────────────────────────────────────────────── */
hr { border-color: #1e2a3a !important; margin: 1rem 0 !important; }
[data-testid="stAlert"] {
    background: #0f1923 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #8899aa !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0066aa, #0088cc) !important;
    border: 1px solid #00d4ff !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    width: 100% !important;     /* full-width tap target on mobile */
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0088cc, #00aadd) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
}

/* Expander mobile-friendly */
[data-testid="stExpander"] {
    border: 1px solid #1e2a3a !important;
    border-radius: 7px !important;
    background: #0f1923 !important;
}

/* Make selectboxes readable on mobile */
[data-testid="stSelectbox"] label { font-size: 0.78rem !important; color: #8899aa !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_assets():
    from data.assets import get_asset_pool
    return get_asset_pool()

@st.cache_data(show_spinner=False)
def _load_scenarios():
    from data.scenarios import get_all_scenarios
    return get_all_scenarios()

def run_all_scenarios(budget: float, mc_iters: int, seed: int,
                      progress_cb=None):
    from agent.portfolio_agent import PortfolioAgent
    agent = PortfolioAgent(budget=budget, mc_iterations=mc_iters, random_seed=seed)
    for i, scenario in enumerate(agent.scenarios):
        if progress_cb:
            progress_cb(i, scenario.name)
        result = agent._run_scenario(scenario)
        agent.results.append(result)
    return agent.results

def results_to_df(results, budget: float) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Scenario":               r.scenario.name,
            "Capital Deployed (£M)":  round(r.optimization.total_capital_deployed / 1e6, 2),
            "Budget Used (%)":        round(r.optimization.total_capital_deployed / budget * 100, 1),
            "Strategic Value (£M)":   round(r.optimization.total_npv / 1e6, 1),
            "Return Multiplier":      round(r.optimization.total_npv /
                                            max(r.optimization.total_capital_deployed, 1), 2),
            "MC Mean NPV (£M)":       round(r.monte_carlo.mean_total_npv / 1e6, 1),
            "P05 NPV (£M)":           round(r.monte_carlo.p5_total_npv / 1e6, 1),
            "Deficit Prob (%)":       round(r.monte_carlo.deficit_probability * 100, 4),
            "Projects Selected":      len(r.optimization.selected_assets),
        })
    return pd.DataFrame(rows)


# ── Chart functions ───────────────────────────────────────────────────────────

def _build_waterfall_chart(result) -> go.Figure:
    assets  = result.optimization.selected_assets
    npv_map = result.optimization.per_asset_npv
    names   = [a.name for a in assets]
    npvs_m  = [npv_map.get(a.id, 0.0) / 1e6 for a in assets]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(assets) + ["total"],
        x=names + ["TOTAL"],
        y=npvs_m + [sum(npvs_m)],
        texttemplate="£%{y:.0f}M",
        textposition="outside",
        textfont=dict(size=9, color="#c0d8f0"),
        connector=dict(line=dict(color="#1e3a5f", width=1, dash="dot")),
        increasing=dict(marker=dict(color=GREEN,  line=dict(color="#0a0e1a", width=0.5))),
        totals=dict(   marker=dict(color=CYAN,    line=dict(color="#0a0e1a", width=0.5))),
        decreasing=dict(marker=dict(color=DANGER, line=dict(color="#0a0e1a", width=0.5))),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="NPV Build-Up by Project — Wave-1 Portfolio",
                   font=dict(color=CYAN, size=13)),
        yaxis_title="NPV Contribution (£M)",
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        height=380,
    )
    return fig


def _build_sector_donut(result) -> go.Figure:
    sector_capex: dict = {}
    for a in result.optimization.selected_assets:
        sector_capex[a.sector] = sector_capex.get(a.sector, 0.0) + a.capital_required

    labels = list(sector_capex.keys())
    values = [sector_capex[l] / 1e6 for l in labels]
    colors = [SECTOR_COLORS.get(l, "#4488aa") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.56,
        marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e0e6f0"),
        hovertemplate="<b>%{label}</b><br>£%{value:.1f}M · %{percent}<extra></extra>",
        pull=[0.03] * len(labels),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Capital Allocation by Sector", font=dict(color=CYAN, size=13)),
        showlegend=True,
        legend=dict(font=dict(size=10, color="#8899aa"), orientation="v",
                    x=1.02, y=0.5),
        annotations=[dict(text="Sector<br>Weights", x=0.5, y=0.5,
                          font=dict(size=10, color="#5577aa"), showarrow=False)],
        height=380,
    )
    return fig


def _build_mc_histogram(mc) -> go.Figure:
    fig = go.Figure()

    if mc.all_npv_samples.size > 0:
        samples_m = mc.all_npv_samples / 1e6
        fig.add_trace(go.Histogram(
            x=samples_m, nbinsx=80, name="Simulated NPV",
            marker=dict(color=CYAN, opacity=0.65,
                        line=dict(color="#0a0e1a", width=0.3)),
            hovertemplate="NPV: £%{x:.0f}M  Count: %{y}<extra></extra>",
        ))
        for val, color, dash, label in [
            (mc.p5_total_npv / 1e6,   DANGER, "dash",  f"P05  £{mc.p5_total_npv/1e6:.0f}M"),
            (mc.mean_total_npv / 1e6, CYAN,   "solid", f"Mean £{mc.mean_total_npv/1e6:.0f}M"),
            (mc.p95_total_npv / 1e6,  GREEN,  "dash",  f"P95  £{mc.p95_total_npv/1e6:.0f}M"),
        ]:
            fig.add_vline(x=val, line_dash=dash, line_color=color, line_width=1.8,
                          annotation_text=label,
                          annotation_position="top",
                          annotation_font=dict(color=color, size=9))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Monte Carlo Distribution — 10,000 Simulations",
                   font=dict(color=CYAN, size=13)),
        xaxis_title="Total Portfolio NPV (£M)",
        yaxis_title="Frequency",
        showlegend=False,
        height=380,
    )
    return fig


def _build_cashflow_area_chart(result, scenario) -> go.Figure:
    years  = list(range(1, 9))
    assets = result.optimization.selected_assets
    fig    = go.Figure()

    for i, asset in enumerate(assets):
        color     = AREA_PALETTE[i % len(AREA_PALETTE)]
        r, g, b   = hex_to_rgb(color)
        fill_rgba = f"rgba({r},{g},{b},0.22)"
        adj_cfs   = [cf * scenario.cash_flow_modifier / 1e6
                     for cf in asset.annual_cash_flows]
        fig.add_trace(go.Scatter(
            x=years, y=adj_cfs,
            name=asset.name,
            mode="lines",
            fill="tozeroy" if i == 0 else "tonexty",
            stackgroup="one",
            line=dict(width=1.2, color=color),
            fillcolor=fill_rgba,
            hovertemplate=f"<b>{asset.name}</b><br>Year %{{x}}: £%{{y:.1f}}M<extra></extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Projected Portfolio Cash Flows — 8 Year Horizon",
                   font=dict(color=CYAN, size=13)),
        xaxis=dict(title="Year", tickvals=years,
                   ticktext=[f"Y{y}" for y in years]),
        yaxis_title="Annual Cash Flow (£M)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.42,
                    font=dict(size=9, color="#6688aa")),
        height=380,
    )
    return fig


def _build_scenario_bar(df: pd.DataFrame) -> go.Figure:
    df_s  = df.sort_values("Strategic Value (£M)", ascending=True).copy()
    mn    = df_s["Strategic Value (£M)"].min()
    mx    = df_s["Strategic Value (£M)"].max()
    rng   = mx - mn or 1.0

    colors = []
    for v in df_s["Strategic Value (£M)"]:
        t = (v - mn) / rng
        colors.append(f"rgb({int(220*(1-t))},{int(180*t+55)},60)")

    fig = go.Figure(go.Bar(
        x=df_s["Strategic Value (£M)"],
        y=df_s["Scenario"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#0a0e1a", width=0.4)),
        text=[f"£{v:,.0f}M" for v in df_s["Strategic Value (£M)"]],
        textposition="outside",
        textfont=dict(size=9, color="#6688aa"),
        hovertemplate="<b>%{y}</b><br>£%{x:,.0f}M<extra></extra>",
    ))

    base_rows = df_s[df_s["Scenario"] == "Base Case"]
    if not base_rows.empty:
        bv = base_rows.iloc[0]["Strategic Value (£M)"]
        fig.add_annotation(
            x=bv, y="Base Case",
            text=" ◀ BASE CASE",
            showarrow=False,
            font=dict(color=CYAN, size=10),
            xanchor="left",
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Strategic Value Across 22 Economic Scenarios",
                   font=dict(color=CYAN, size=13)),
        xaxis_title="Strategic Value (£M)",
        yaxis=dict(automargin=True, tickfont=dict(size=8.5)),
        height=500,
    )
    return fig


def _build_risk_return_scatter(df: pd.DataFrame) -> go.Figure:
    is_base = df["Scenario"] == "Base Case"
    other   = df[~is_base]
    base    = df[is_base]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=other["Deficit Prob (%)"],
        y=other["Return Multiplier"],
        mode="markers",
        text=other["Scenario"],
        marker=dict(
            size=(other["Capital Deployed (£M)"].clip(lower=1) * 0.35).clip(upper=18),
            color=CYAN, opacity=0.65,
            line=dict(color="#0a0e1a", width=1),
        ),
        hovertemplate="<b>%{text}</b><br>Deficit: %{x:.4f}%<br>Return: %{y:.2f}×<extra></extra>",
        name="Scenarios",
    ))
    if not base.empty:
        fig.add_trace(go.Scatter(
            x=base["Deficit Prob (%)"],
            y=base["Return Multiplier"],
            mode="markers+text",
            text=["Base Case"],
            textposition="top right",
            textfont=dict(size=10, color=CYAN),
            marker=dict(size=14, color=CYAN, symbol="star",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>Base Case</b><br>Deficit: %{x:.4f}%<br>Return: %{y:.2f}×<extra></extra>",
            name="Base Case",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Risk vs Return — 22 Scenarios",
                   font=dict(color=CYAN, size=13)),
        xaxis_title="Deficit Probability (%)",
        yaxis_title="Return Multiplier (×)",
        showlegend=False,
        height=420,
    )
    return fig


def _build_scenario_heatmap(df: pd.DataFrame) -> go.Figure:
    cols   = ["Return Multiplier", "Capital Deployed (£M)", "MC Mean NPV (£M)", "Deficit Prob (%)"]
    df_h   = df.set_index("Scenario")[cols].copy()
    df_raw = df_h.copy()

    # Normalise 0–1 for colour; invert Deficit Prob (lower = better → higher colour)
    df_n = (df_h - df_h.min()) / (df_h.max() - df_h.min() + 1e-9)
    df_n["Deficit Prob (%)"] = 1 - df_n["Deficit Prob (%)"]

    text_vals = [[f"{df_raw.iloc[r, c]:.1f}" for c in range(len(cols))]
                 for r in range(len(df_h))]

    fig = go.Figure(go.Heatmap(
        z=df_n.values,
        x=cols,
        y=df_h.index.tolist(),
        colorscale="RdYlGn",
        showscale=True,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=8.5),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#8899aa", size=9)),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Scenario Metrics Heatmap",
                   font=dict(color=CYAN, size=13)),
        xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
        yaxis=dict(automargin=True, tickfont=dict(size=8)),
        height=500,
    )
    return fig


def _build_delta_bar(query_result, base_result) -> go.Figure:
    labels  = ["Strategic Value (£M)", "MC Mean NPV (£M)", "P05 NPV (£M)", "Return Multiplier"]
    q_vals  = [
        query_result.optimization.total_npv / 1e6,
        query_result.monte_carlo.mean_total_npv / 1e6,
        query_result.monte_carlo.p5_total_npv / 1e6,
        query_result.optimization.total_npv /
            max(query_result.optimization.total_capital_deployed, 1),
    ]
    b_vals  = [
        base_result.optimization.total_npv / 1e6,
        base_result.monte_carlo.mean_total_npv / 1e6,
        base_result.monte_carlo.p5_total_npv / 1e6,
        base_result.optimization.total_npv /
            max(base_result.optimization.total_capital_deployed, 1),
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base Case", x=labels, y=b_vals,
                         marker_color=CYAN, opacity=0.45))
    fig.add_trace(go.Bar(name="Query Result", x=labels, y=q_vals,
                         marker_color=GREEN, opacity=0.85))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Query Result vs Base Case",
                   font=dict(color=CYAN, size=13)),
        barmode="group",
        legend=dict(font=dict(color="#8899aa", size=11)),
        height=350,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#00d4ff;font-weight:700;font-size:1rem;'
                'margin-bottom:1rem;letter-spacing:0.05em">⚙ PARAMETERS</div>',
                unsafe_allow_html=True)

    budget_m = st.slider("Capital Budget (£M)", 10, 200, 45, 5)
    budget   = budget_m * 1_000_000

    mc_iters = st.select_slider(
        "Monte Carlo Iterations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    seed = st.number_input("Random Seed", value=42, step=1,
                           help="Fix for reproducible Monte Carlo results")

    st.markdown("---")
    st.markdown('<div style="color:#5577aa;font-size:0.78rem;'
                'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">'
                'Agentic RAG (optional)</div>', unsafe_allow_html=True)
    groq_key = st.text_input("GROQ_API_KEY", type="password",
                              help="Free key at console.groq.com — required for NL Query tab")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if "last_run" in st.session_state:
        st.success(f"✓ Last run: {st.session_state['last_run']}")

    st.markdown("---")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
  <div class="hero-title">Capital Portfolio Optimisation Agent</div>
  <div class="hero-subtitle">
    Autonomous NPV Optimiser &nbsp;·&nbsp; Monte Carlo Risk Engine &nbsp;·&nbsp;
    Agentic RAG &nbsp;·&nbsp; 22 Scenarios
  </div>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Wave-1 Strategic Value</div>
      <div class="kpi-value">£929M</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Capital Deployed</div>
      <div class="kpi-value white">£45M</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Return Multiplier</div>
      <div class="kpi-value green">20.64×</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Downside Deficit Probability</div>
      <div class="kpi-value green">0.0000%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Portfolio Dashboard",
    "📈  Scenario Analysis",
    "🤖  Natural Language Query",
    "📖  About",
    "⚙️  Workflow Run",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Wave-1 Roadmap")
    st.caption("Run all 22 scenarios and explore the optimal portfolio in the base case.")

    if st.button("▶  Run Full Analysis", type="primary", key="run_all"):
        prog      = st.progress(0.0, text="Initialising…")
        prog_text = st.empty()
        try:
            def _cb(i, name):
                pct = (i + 1) / 22
                prog.progress(pct,
                    text=f"Scenario {i+1}/22 — {name}")
            results = run_all_scenarios(budget, mc_iters, int(seed),
                                        progress_cb=_cb)
            st.session_state["results"]  = results
            st.session_state["budget"]   = budget
            st.session_state["last_run"] = f"£{budget_m}M · {mc_iters:,} iters"
            prog.empty()
            prog_text.empty()
        except Exception as e:
            prog.empty()
            st.error(f"Analysis error: {e}")

    if "results" not in st.session_state:
        st.markdown("""
<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:7px;
padding:2.5rem;text-align:center;margin-top:1rem">
  <div style="font-size:2.5rem;margin-bottom:0.6rem">📊</div>
  <div style="font-size:1rem;color:#8899aa">Configure parameters in the sidebar,
  then click <b style="color:#00d4ff">▶ Run Full Analysis</b> to load results.</div>
</div>""", unsafe_allow_html=True)
    else:
        results     = st.session_state["results"]
        run_budget  = st.session_state.get("budget", budget)
        df          = results_to_df(results, run_budget)
        base_result = next(r for r in results if r.scenario.id == 1)
        base_row    = df[df["Scenario"] == "Base Case"].iloc[0]

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategic Value",
                  f"£{base_row['Strategic Value (£M)']:,.0f}M")
        c2.metric("Capital Deployed",
                  f"£{base_row['Capital Deployed (£M)']:,.0f}M")
        c3.metric("Return Multiplier",
                  f"{base_row['Return Multiplier']:.2f}×")
        c4.metric("Deficit Probability",
                  f"{base_row['Deficit Prob (%)']:.4f}%")

        st.markdown("---")

        # Row 2: Waterfall + Donut
        st.markdown("### Portfolio Composition")
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(_build_waterfall_chart(base_result),
                            use_container_width=True,
                            key="base_waterfall",
                            config={"displayModeBar": False})
        with col_r:
            st.plotly_chart(_build_sector_donut(base_result),
                            use_container_width=True,
                            key="base_donut",
                            config={"displayModeBar": False})

        # Row 3: MC Histogram + Cash Flow
        st.markdown("### Risk Analysis & Cash Flow Projections")
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.plotly_chart(_build_mc_histogram(base_result.monte_carlo),
                            use_container_width=True,
                            key="base_mc_hist",
                            config={"displayModeBar": True,
                                    "modeBarButtonsToRemove": ["toImage"]})
            # MC stats summary row
            mc = base_result.monte_carlo
            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("Std Dev",      f"£{mc.std_total_npv/1e6:.1f}M")
            ms2.metric("VaR (95%)",    f"£{(mc.mean_total_npv - mc.p5_total_npv)/1e6:.1f}M below mean")
            ms3.metric("P95 Upside",   f"£{mc.p95_total_npv/1e6:,.0f}M")
        with col_r2:
            st.plotly_chart(
                _build_cashflow_area_chart(base_result, base_result.scenario),
                use_container_width=True,
                key="base_cashflow_area",
                config={"displayModeBar": True,
                        "modeBarButtonsToRemove": ["toImage"]})

        # Row 4: Portfolio table with IRR column
        st.markdown("### Selected Portfolio — Base Case")
        from engine.npv_engine import calculate_irr
        port_rows = []
        for asset in base_result.optimization.selected_assets:
            pi  = base_result.optimization.profitability_indices.get(asset.id, 0)
            npv = base_result.optimization.per_asset_npv.get(asset.id, 0)
            raw_irr = calculate_irr([-asset.capital_required] + asset.annual_cash_flows)
            irr_str = f"{raw_irr*100:.1f}%" if not np.isnan(raw_irr) else "N/A"
            port_rows.append({
                "Project":       asset.name,
                "Sector":        asset.sector,
                "Capex (£M)":    round(asset.capital_required / 1e6, 2),
                "NPV (£M)":      round(npv / 1e6, 1),
                "PI":            round(pi, 3),
                "IRR":           irr_str,
            })
        st.dataframe(
            pd.DataFrame(port_rows).style.format({
                "Capex (£M)": "{:.2f}",
                "NPV (£M)":   "{:.1f}",
                "PI":         "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Scenario drill-down expander
        with st.expander("🔍 Drill into another scenario"):
            all_names   = [r.scenario.name for r in results]
            chosen_name = st.selectbox("Select scenario:", all_names,
                                       index=0, key="drilldown_sel")
            chosen_res  = next(r for r in results
                               if r.scenario.name == chosen_name)
            chosen_row  = df[df["Scenario"] == chosen_name].iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Strategic Value",
                      f"£{chosen_row['Strategic Value (£M)']:,.0f}M")
            d2.metric("Capital Deployed",
                      f"£{chosen_row['Capital Deployed (£M)']:,.0f}M")
            d3.metric("Return Multiplier",
                      f"{chosen_row['Return Multiplier']:.2f}×")
            d4.metric("Deficit Probability",
                      f"{chosen_row['Deficit Prob (%)']:.4f}%")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.plotly_chart(_build_waterfall_chart(chosen_res),
                                use_container_width=True,
                                key=f"drill_waterfall_{chosen_name}",
                                config={"displayModeBar": False})
            with dc2:
                st.plotly_chart(_build_sector_donut(chosen_res),
                                use_container_width=True,
                                key=f"drill_donut_{chosen_name}",
                                config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Scenario Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Cross-Scenario Performance")
    st.caption("Run the Portfolio Dashboard first to load scenario results.")

    if "results" not in st.session_state:
        st.markdown("""
<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:7px;
padding:2.5rem;text-align:center;margin-top:1rem">
  <div style="font-size:2.5rem;margin-bottom:0.6rem">📈</div>
  <div style="font-size:1rem;color:#8899aa">No results yet — run the analysis in
  <b style="color:#00d4ff">📊 Portfolio Dashboard</b> first.</div>
</div>""", unsafe_allow_html=True)
    else:
        results    = st.session_state["results"]
        run_budget = st.session_state.get("budget", budget)
        df         = results_to_df(results, run_budget)

        # Full-width scenario bar
        st.plotly_chart(_build_scenario_bar(df), use_container_width=True,
                        key="tab2_scenario_bar",
                        config={"displayModeBar": False})

        st.markdown("---")

        # Risk-Return + Heatmap
        st.markdown("### Risk-Return Profile & Metrics Heatmap")
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.plotly_chart(_build_risk_return_scatter(df),
                            use_container_width=True,
                            key="tab2_risk_return",
                            config={"displayModeBar": False})
        with col_r:
            st.plotly_chart(_build_scenario_heatmap(df),
                            use_container_width=True,
                            key="tab2_heatmap",
                            config={"displayModeBar": False})

        st.markdown("---")

        # Scenario comparison
        st.markdown("### Side-by-Side Scenario Comparison")
        all_names = df["Scenario"].tolist()
        cmp_c1, cmp_c2 = st.columns(2)
        scen_a = cmp_c1.selectbox("Scenario A:", all_names,
                                   index=0, key="cmp_a")
        scen_b = cmp_c2.selectbox("Scenario B:", all_names,
                                   index=2, key="cmp_b")
        res_a = next(r for r in results if r.scenario.name == scen_a)
        res_b = next(r for r in results if r.scenario.name == scen_b)
        st.plotly_chart(_build_delta_bar(res_a, res_b),
                        use_container_width=True,
                        key=f"tab2_compare_{scen_a}_{scen_b}",
                        config={"displayModeBar": False})

        st.markdown("---")

        # Full results table
        st.markdown("### Full Results Table")
        st.dataframe(
            df.style.format({
                "Capital Deployed (£M)": "{:.2f}",
                "Budget Used (%)":       "{:.1f}%",
                "Strategic Value (£M)":  "{:.1f}",
                "Return Multiplier":     "{:.2f}×",
                "MC Mean NPV (£M)":      "{:.1f}",
                "P05 NPV (£M)":          "{:.1f}",
                "Deficit Prob (%)":      "{:.4f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇  Download CSV",
            df.to_csv(index=False),
            file_name="wave1_scenario_analysis.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Natural Language Query
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Agentic RAG — Natural Language Query")
    st.caption(
        "Describe an economic scenario in plain English. The agent plans retrieval, "
        "scores confidence, re-queries if needed, composes custom scenario parameters, "
        "then runs the full financial pipeline."
    )

    examples = [
        "— type your own —",
        "Tech market crash with rising interest rates",
        "Severe recession with supply chain disruptions",
        "Energy transition acceleration — green mandate boom",
        "Geopolitical stress affecting infrastructure investments",
        "Post-pandemic recovery with low rates and tech boom",
    ]
    sel = st.selectbox("Example queries:", examples)
    query = st.text_area(
        "Describe your scenario:",
        value="" if sel == "— type your own —" else sel,
        height=80,
        placeholder='e.g. "Moderate tech slowdown with rising interest rates"',
    )

    run_query = st.button("▶  Analyse Scenario", type="primary", key="run_query")

    if run_query:
        if not query.strip():
            st.warning("Please enter a scenario description.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is required for this mode. Enter it in the sidebar.")
        else:
            from retrieval.scenario_store import ScenarioStore
            from retrieval.standard_rag import StandardRAG
            from retrieval.agentic_rag import AgenticRAG
            from llm.claude_client import ClaudeClient
            from agent.portfolio_agent import PortfolioAgent

            agent  = PortfolioAgent(budget=budget, mc_iterations=mc_iters,
                                    random_seed=int(seed))
            store  = ScenarioStore(agent.scenarios)

            # Standard RAG (shown for contrast)
            std_scenario = StandardRAG(store).retrieve(query)

            # Agentic RAG
            with st.spinner("Agentic RAG: planning retrieval and scoring confidence…"):
                try:
                    client     = ClaudeClient()
                    rag_result = AgenticRAG(store, client).retrieve(query)
                except EnvironmentError as e:
                    st.error(str(e))
                    st.stop()

            # RAG comparison cards
            col_std, col_agt = st.columns(2)
            with col_std:
                st.markdown(f"""
                <div class="rag-card standard">
                  <div class="rag-tag standard">⚡ Standard RAG — Direct TF-IDF Match</div>
                  <div class="rag-name">{std_scenario.name}</div>
                  <div class="rag-meta">Single nearest-vector lookup · no reasoning step<br>
                  Limitation: drops multi-component stressors silently</div>
                </div>""", unsafe_allow_html=True)

            with col_agt:
                st.markdown(f"""
                <div class="rag-card agent">
                  <div class="rag-tag agent">🧠 Agentic RAG — LLM Composed</div>
                  <div class="rag-name">{rag_result.composed_scenario.name}</div>
                  <div class="rag-meta">
                    Confidence: <b style="color:#00ff88">{rag_result.confidence:.2f}</b>
                    &nbsp;·&nbsp; Retrieval attempts: <b>{rag_result.attempts}</b><br>
                    {rag_result.reasoning[:140]}{'…' if len(rag_result.reasoning) > 140 else ''}
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Composed scenario parameters
            sc = rag_result.composed_scenario
            st.markdown(f"""
            <div class="params-card">
<span style="color:#00d4ff;font-weight:700">COMPOSED SCENARIO PARAMETERS</span>

  Name                  :  {sc.name}
  cash_flow_modifier    :  {sc.cash_flow_modifier:.3f}
  discount_rate_delta   : {sc.discount_rate_delta:+.3f}
  capex_modifier        :  {sc.capex_modifier:.3f}
  risk_sigma_multiplier :  {sc.risk_sigma_multiplier:.2f}
  eligible_sectors      :  {', '.join(sc.eligible_sectors) if sc.eligible_sectors else 'ALL'}

  Candidates used       :  {', '.join(s.name for s in rag_result.candidates)}
            </div>""", unsafe_allow_html=True)

            # Run financial pipeline
            with st.spinner("Running NPV optimiser and Monte Carlo…"):
                run_result = agent._run_scenario(rag_result.composed_scenario)

            opt = run_result.optimization
            mc  = run_result.monte_carlo

            # KPI metrics
            st.markdown("### Pipeline Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategic Value",
                      f"£{opt.total_npv/1e6:,.1f}M")
            c2.metric("Capital Deployed",
                      f"£{opt.total_capital_deployed/1e6:,.1f}M")
            c3.metric("Return Multiplier",
                      f"{opt.total_npv/max(opt.total_capital_deployed,1):.2f}×")
            c4.metric("Deficit Probability",
                      f"{mc.deficit_probability*100:.4f}%")

            c5, c6, c7 = st.columns(3)
            c5.metric("MC Mean NPV",   f"£{mc.mean_total_npv/1e6:,.1f}M")
            c6.metric("P05 Downside",  f"£{mc.p5_total_npv/1e6:,.1f}M")
            c7.metric("P95 Upside",    f"£{mc.p95_total_npv/1e6:,.1f}M")

            st.markdown("---")

            # Delta chart vs base case
            if "results" in st.session_state:
                base_r = next(r for r in st.session_state["results"]
                              if r.scenario.id == 1)
                st.markdown("### Query Result vs Base Case")
                st.plotly_chart(_build_delta_bar(run_result, base_r),
                                use_container_width=True,
                                key="tab3_delta_bar",
                                config={"displayModeBar": False})
            else:
                st.markdown("""
<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:6px;
padding:1rem;text-align:center;color:#5577aa;font-size:0.88rem">
  Run <b style="color:#00d4ff">📊 Portfolio Dashboard</b> first to enable Base Case comparison.
</div>""", unsafe_allow_html=True)

            # Selected portfolio
            st.markdown("### Selected Portfolio")
            p_rows = [{
                "Project":    a.name,
                "Sector":     a.sector,
                "Capex (£M)": round(a.capital_required / 1e6, 2),
                "NPV (£M)":   round(opt.per_asset_npv.get(a.id, 0) / 1e6, 1),
                "PI":         round(opt.profitability_indices.get(a.id, 0), 3),
            } for a in opt.selected_assets]
            if p_rows:
                st.dataframe(pd.DataFrame(p_rows), use_container_width=True,
                             hide_index=True)
            else:
                st.warning("No assets selected — all projects had negative NPV under this scenario.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### About This Project")

    st.markdown("""
<div class="arch-card"><span style="color:#00d4ff;font-size:0.88rem;font-weight:700">SYSTEM ARCHITECTURE</span>

  investment_pipeline.xlsx   ──  Project data source (18 assets · 8-year cash flows)
          │
  data/assets.py             ──  Asset dataclass loader (Excel → typed objects)
  data/scenarios.py          ──  22 Scenario definitions (modifiers + sector mandates)
          │
  engine/npv_engine.py       ──  Discounted cash flow + IRR  (NumPy vectorised)
  engine/optimizer.py        ──  Greedy knapsack  ·  ranked by PI = NPV / capex
  engine/monte_carlo.py      ──  10,000-iteration shock engine  (N × A × T tensor)
          │
  agent/portfolio_agent.py   ──  Orchestrator  ·  runs all 22 scenarios autonomously
          │
  retrieval/scenario_store   ──  TF-IDF vector index over scenario documents
  retrieval/standard_rag     ──  Direct nearest-vector lookup  (baseline · limitation shown)
  retrieval/agentic_rag      ──  Plan → retrieve → score → re-query → compose
  llm/claude_client.py       ──  Groq API  (LLaMA 3.3 70B)  ·  Anthropic code commented
          │
  app.py                     ──  This Streamlit dashboard</div>
""", unsafe_allow_html=True)

    st.markdown("### Base Case Results")
    st.markdown("""
| Metric | Value |
|---|---|
| Capital budget | £45,000,000 |
| Wave-1 strategic value (NPV) | £929,000,000 |
| Return multiplier | 20.64× |
| Deficit probability | 0.0000% across all 22 scenarios |
| Monte Carlo iterations | 10,000 per scenario |
| Scenarios processed | 22 |
| Investment horizon | 8 years (Y1 – Y8) |
| Asset universe | 18 strategic projects |
""")

    st.markdown("### Tech Stack")
    st.markdown("""
<div style="margin-top:0.5rem;margin-bottom:1.5rem">
  <span class="badge">Python 3.11+</span>
  <span class="badge">NumPy</span>
  <span class="badge">pandas</span>
  <span class="badge">scikit-learn</span>
  <span class="badge">Streamlit</span>
  <span class="badge">Plotly</span>
  <span class="badge">Groq API</span>
  <span class="badge">LLaMA 3.3 70B</span>
  <span class="badge">TF-IDF RAG</span>
  <span class="badge">Monte Carlo</span>
  <span class="badge">DCF / NPV</span>
  <span class="badge">Greedy Knapsack</span>
  <span class="badge">LangGraph</span>
  <span class="badge">ReportLab</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Workflow Run  (LangGraph multi-agent pipeline)
# ════════════════════════════════════════════════════════════════════════════
_WF_NODES = [
    "data_validation_node",
    "scenario_retrieval_node",
    "financial_engine_node",
    "commentary_node",
    "risk_flag_node",
    "human_approval_node",
    "report_generation_node",
]
_WF_LABELS = {
    "data_validation_node":    "1  Data Validation",
    "scenario_retrieval_node": "2  Scenario Retrieval",
    "financial_engine_node":   "3  Financial Engine",
    "commentary_node":         "4a Commentary  (parallel)",
    "risk_flag_node":          "4b Risk Flags  (parallel)",
    "human_approval_node":     "5  Human Approval",
    "report_generation_node":  "6  Report Generation",
}
_STATUS_ICON = {
    "pending":  "⬜",
    "running":  "🔄",
    "complete": "✅",
    "failed":   "❌",
    "skipped":  "⏭️",
}


def _wf_status_html(node_status: dict) -> str:
    rows = ""
    for node in _WF_NODES:
        st_val = node_status.get(node, "pending")
        icon   = _STATUS_ICON.get(st_val, "⬜")
        colour = (
            "#00ff88" if st_val == "complete"
            else "#ff4444" if st_val == "failed"
            else "#00d4ff" if st_val == "running"
            else "#5577aa"
        )
        rows += (
            f'<div style="display:flex;align-items:center;gap:0.6rem;'
            f'padding:0.35rem 0;border-bottom:1px solid #1e2a3a">'
            f'<span style="font-size:1.1rem">{icon}</span>'
            f'<span style="color:{colour};font-size:0.85rem;font-family:monospace">'
            f'{_WF_LABELS.get(node, node)}</span></div>'
        )
    return f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:7px;padding:0.8rem 1rem">{rows}</div>'


with tab5:
    st.markdown("### LangGraph Multi-Agent Workflow")
    st.caption(
        "Runs the full 6-node pipeline: data validation → scenario retrieval → "
        "financial engine → commentary + risk flags (parallel) → human approval → "
        "report generation (JSON + CSV + PDF)."
    )

    # ── Initialise session state ──────────────────────────────────────────────
    if "wf_graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        from workflow.graph import build_graph
        st.session_state.wf_graph        = build_graph(MemorySaver())
        st.session_state.wf_thread_id    = None
        st.session_state.wf_node_status  = {n: "pending" for n in _WF_NODES}
        st.session_state.wf_interrupted  = False
        st.session_state.wf_interrupt_pl = {}
        st.session_state.wf_commentary   = {}
        st.session_state.wf_risk_flags   = []
        st.session_state.wf_report_paths = {}

    # ── Sidebar-style controls ────────────────────────────────────────────────
    wf_col1, wf_col2 = st.columns([2, 2])
    with wf_col1:
        wf_mode = st.selectbox(
            "Mode",
            ["Run All 22 Scenarios", "Natural Language Query"],
            key="wf_mode",
        )
    with wf_col2:
        wf_query = ""
        if wf_mode == "Natural Language Query":
            wf_query = st.text_input(
                "Query",
                placeholder='e.g. "tech crash with rising rates"',
                key="wf_query_input",
            )

    wf_budget   = st.session_state.get("budget",  45_000_000)
    wf_mc_iters = st.session_state.get("mc_iters", 10_000)

    # ── Status panel ─────────────────────────────────────────────────────────
    status_placeholder = st.empty()
    status_placeholder.markdown(
        _wf_status_html(st.session_state.wf_node_status),
        unsafe_allow_html=True,
    )

    # ── Start button ──────────────────────────────────────────────────────────
    if st.button("▶  Start Workflow", type="primary", key="wf_start"):
        import uuid as _uuid
        from langgraph.types import Command

        thread_id = str(_uuid.uuid4())
        st.session_state.wf_thread_id    = thread_id
        st.session_state.wf_node_status  = {n: "pending" for n in _WF_NODES}
        st.session_state.wf_interrupted  = False
        st.session_state.wf_interrupt_pl = {}
        st.session_state.wf_commentary   = {}
        st.session_state.wf_risk_flags   = []
        st.session_state.wf_report_paths = {}

        intent = "run_all" if wf_mode == "Run All 22 Scenarios" else (wf_query or "run_all")
        config = {"configurable": {"thread_id": thread_id}}
        graph  = st.session_state.wf_graph

        initial = {
            "user_intent":   intent,
            "budget_gbp":    wf_budget,
            "mc_iterations": wf_mc_iters,
        }

        # Stream until graph pauses or completes
        try:
            for event in graph.stream(initial, config=config, stream_mode="updates"):
                for node_name in event:
                    st.session_state.wf_node_status[node_name] = "complete"
                    # Capture commentary and risk flags as they arrive
                    node_out = event[node_name]
                    if "commentary"  in node_out:
                        st.session_state.wf_commentary  = node_out["commentary"]
                    if "risk_flags"  in node_out:
                        st.session_state.wf_risk_flags  = node_out["risk_flags"]
                    if "report_paths" in node_out:
                        st.session_state.wf_report_paths = node_out["report_paths"]
                status_placeholder.markdown(
                    _wf_status_html(st.session_state.wf_node_status),
                    unsafe_allow_html=True,
                )
        except Exception as _e:
            st.error(f"Workflow error: {_e}")

        # Check for interrupt
        snapshot = graph.get_state(config)
        is_interrupted = bool(snapshot.next)
        if is_interrupted:
            try:
                for task in (snapshot.tasks or []):
                    if getattr(task, "interrupts", None):
                        st.session_state.wf_interrupt_pl = task.interrupts[0].value
                        break
            except Exception:
                pass
            st.session_state.wf_interrupted = True
        st.rerun()

    # ── Commentary stream panel ────────────────────────────────────────────────
    if st.session_state.wf_commentary:
        st.markdown("---")
        st.markdown("### Commentary")
        expls = st.session_state.wf_commentary.get("chart_explanations", {})
        if expls:
            for chart, text in expls.items():
                st.markdown(
                    f'<div class="params-card"><b style="color:#00d4ff">'
                    f'{chart.replace("_"," ").title()}</b><br>{text}</div>',
                    unsafe_allow_html=True,
                )
        sugs = st.session_state.wf_commentary.get("suggestions", [])
        if sugs:
            st.markdown("**Strategic Suggestions**")
            for s in sugs:
                urg = s.get("urgency", "—")
                col = "#ff4444" if urg == "High" else "#ffaa00" if urg == "Medium" else "#00ff88"
                st.markdown(
                    f'<div style="background:#0f1923;border-left:3px solid {col};'
                    f'padding:0.6rem 1rem;border-radius:0 5px 5px 0;margin:0.3rem 0">'
                    f'<span style="color:{col};font-size:0.7rem;text-transform:uppercase">'
                    f'{urg}</span>  {s.get("action","")}<br>'
                    f'<span style="color:#5577aa;font-size:0.8rem">{s.get("rationale","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Risk flags stream panel ───────────────────────────────────────────────
    if st.session_state.wf_risk_flags:
        st.markdown("---")
        st.markdown("### Risk Flags")
        for flag in st.session_state.wf_risk_flags:
            sev   = flag["severity"]
            col   = "#ff4444" if sev == "HIGH" else "#ffaa00" if sev == "MEDIUM" else "#00ff88"
            icon  = "🔴" if sev == "HIGH" else "🟡" if sev == "MEDIUM" else "🟢"
            st.markdown(
                f'<div class="rag-card" style="border-color:{col}">'
                f'{icon} <b style="color:{col}">[{sev}]</b>  {flag["description"]}<br>'
                f'<span style="color:#5577aa;font-size:0.78rem">'
                f'Metric: {flag["affected_metric"]}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Human approval checkpoint ─────────────────────────────────────────────
    if st.session_state.get("wf_interrupted"):
        pl = st.session_state.wf_interrupt_pl
        res = pl.get("results", {})
        st.markdown("---")
        st.markdown(
            '<div style="background:#0f1923;border:2px solid #ffaa00;border-radius:8px;'
            'padding:1.2rem 1.5rem;margin:0.5rem 0">'
            '<span style="color:#ffaa00;font-size:0.75rem;text-transform:uppercase;'
            'letter-spacing:0.1em">⏸ Human Approval Required</span>'
            f'<div style="margin-top:0.6rem;color:#e0e6f0;font-size:0.95rem">'
            f'Strategic Value: <b style="color:#00d4ff">£{res.get("total_npv_m",0):,.1f}M</b>  |  '
            f'Return: <b style="color:#00ff88">{res.get("return_multiplier",0):.2f}×</b>  |  '
            f'Deficit: <b>{res.get("deficit_probability_pct",0):.4f}%</b></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        ap_c1, ap_c2 = st.columns(2)
        if ap_c1.button("✅  Approve — Generate Reports", type="primary", key="wf_approve"):
            from langgraph.types import Command
            config = {"configurable": {"thread_id": st.session_state.wf_thread_id}}
            graph  = st.session_state.wf_graph
            st.session_state.wf_node_status["human_approval_node"] = "complete"
            for event in graph.stream(
                Command(resume="yes"), config=config, stream_mode="updates"
            ):
                for node_name in event:
                    st.session_state.wf_node_status[node_name] = "complete"
                    node_out = event[node_name]
                    if "report_paths" in node_out:
                        st.session_state.wf_report_paths = node_out["report_paths"]
            st.session_state.wf_interrupted = False
            st.rerun()

        if ap_c2.button("❌  Reject — End Workflow", key="wf_reject"):
            from langgraph.types import Command
            config = {"configurable": {"thread_id": st.session_state.wf_thread_id}}
            graph  = st.session_state.wf_graph
            st.session_state.wf_node_status["human_approval_node"] = "complete"
            for event in graph.stream(
                Command(resume="no"), config=config, stream_mode="updates"
            ):
                pass
            st.session_state.wf_interrupted = False
            st.info("Workflow terminated — no reports generated.")
            st.rerun()

    # ── Report paths ──────────────────────────────────────────────────────────
    if st.session_state.wf_report_paths:
        st.markdown("---")
        st.markdown("### Reports Generated")
        for fmt, path in st.session_state.wf_report_paths.items():
            st.markdown(
                f'<div class="params-card">'
                f'<b style="color:#00d4ff">{fmt.upper()}</b>  {path}</div>',
                unsafe_allow_html=True,
            )
        # PDF download button
        pdf_path = st.session_state.wf_report_paths.get("pdf")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as _f:
                st.download_button(
                    "⬇  Download PDF Board Summary",
                    data=_f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key="wf_pdf_download",
                )

