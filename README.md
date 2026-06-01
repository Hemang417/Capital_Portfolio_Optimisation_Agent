# Capital Portfolio Optimisation Agent

**MSc Financial Data Analytics | Python · NumPy · Monte Carlo · Agentic RAG**

A modular financial agent that autonomously selects optimal asset allocations across 22 distinct
operational scenarios using net present value ranking, a budget-constrained greedy optimiser, and a
10,000-iteration Monte Carlo shock engine — with an agentic RAG layer for natural language querying
and a Bloomberg-style Streamlit dashboard.

> **Headline results (base case)**
> - Capital budget: **£45,000,000**
> - Wave-1 strategic value: **£929,000,000**
> - Return multiplier: **20.64×**
> - Downside deficit probability: **0.0000%** across all 22 simulated scenarios

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Data Layer — Excel Pipeline](#data-layer--excel-pipeline)
6. [The 22 Scenarios](#the-22-scenarios)
7. [Budget-Constrained Optimiser](#budget-constrained-optimiser)
8. [Monte Carlo Shock Engine](#monte-carlo-shock-engine)
9. [Agentic RAG — Natural Language Query](#agentic-rag--natural-language-query)
10. [Web App — Streamlit Dashboard](#web-app--streamlit-dashboard)
11. [Test Suite](#test-suite)
12. [Quick Start](#quick-start)
13. [Running the Agent](#running-the-agent)
14. [Output Files](#output-files)
15. [Using Real Firm Data](#using-real-firm-data)
16. [Adjusting the Model](#adjusting-the-model)
17. [Dependencies](#dependencies)

---

## Project Overview

The agent solves a capital allocation problem: given a fixed investment budget and a pipeline of
strategic projects, which combination maximises net present value — and how resilient is that
portfolio under economic stress?

Five key capabilities:

| Capability | Detail |
|---|---|
| **Modular architecture** | Raw data layer (`data/`) separated from calculation logic (`engine/`) and orchestration (`agent/`) |
| **Autonomous scenario processing** | Loops all 22 scenarios with zero manual reconfiguration between runs |
| **Probabilistic stress testing** | 10,000-iteration Monte Carlo confirms portfolio resilience across every scenario |
| **Agentic RAG** | Natural language queries resolved to scenario parameters via reasoning-driven retrieval with confidence scoring |
| **Bloomberg-style dashboard** | Interactive Streamlit web app with 8 Plotly charts, drill-down, and scenario comparison |

---

## Architecture

```
+----------------------------------------------------------+
|         app.py  (Streamlit Bloomberg-style dashboard)    |
+----------------------------------------------------------+
|                      main.py  (CLI)                      |
+----------------------------+-----------------------------+
                             |
+----------------------------v-----------------------------+
|              agent/portfolio_agent.py                    |
|   run()           -- loops 22 scenarios autonomously     |
|   run_query()     -- agentic RAG mode                    |
+----+---------------------+--------------------------------+
     |                     |
+----v----+      +---------v-------------------------------+
|  data/  |      |              engine/                   |
|         |      |  npv_engine.py   -> NPV / IRR          |
| assets  +----->|  optimizer.py    -> Greedy knapsack     |
| .xlsx   |      |  monte_carlo.py  -> 10,000-iter MC      |
|         |      +---------------------------+-------------+
| scen-   |                  |
| arios   |      +-----------v-----------------------------+
+---------+      |           reporting/                   |
     |           |  console_reporter.py  (tables)         |
     |           |  file_reporter.py     (JSON / CSV)     |
     |           +----------------------------------------+
     |
+----v----------------------------------------------------+
|              retrieval/  (Agentic RAG layer)            |
|  scenario_store.py  -- TF-IDF vector index              |
|  standard_rag.py    -- direct lookup (baseline)         |
|  agentic_rag.py     -- plan -> retrieve -> score -> compose|
+----+----------------------------------------------------+
     |
+----v----------------------------------------------------+
|              llm/claude_client.py  (Groq API)           |
|  llama-3.1-8b-instant     -- query planning/refinement  |
|  llama-3.3-70b-versatile  -- confidence scoring/compose |
+----------------------------------------------------------+
```

---

## Project Structure

```
Capital_Portfolio_Optimisation_Agent/
|
+-- app.py                       # Streamlit Bloomberg-style dashboard
+-- main.py                      # CLI entry point
+-- models.py                    # Shared ScenarioRunResult dataclass
+-- requirements.txt             # Python dependencies
+-- create_sample_data.py        # Regenerates investment_pipeline.xlsx
+-- investment_pipeline.xlsx     # Investment project data (18 sample projects)
|                                # Replace rows with your firm's real pipeline
|
+-- data/
|   +-- assets.py                # Reads investment pipeline from Excel
|   +-- scenarios.py             # 22 scenario definitions + to_document() for RAG
|
+-- engine/
|   +-- npv_engine.py            # calculate_npv, calculate_irr, npv_to_capital_ratio
|   +-- optimizer.py             # Greedy knapsack optimiser (BUDGET = 45M GBP)
|   +-- monte_carlo.py           # Vectorised 10,000-iteration MC engine
|
+-- agent/
|   +-- portfolio_agent.py       # run() loops 22 scenarios / run_query() agentic RAG
|
+-- retrieval/
|   +-- scenario_store.py        # TF-IDF vector index over 22 scenario documents
|   +-- standard_rag.py          # Direct cosine similarity lookup (baseline)
|   +-- agentic_rag.py           # Plan -> retrieve -> confidence -> re-query -> compose
|
+-- llm/
|   +-- claude_client.py         # Groq API wrapper (Anthropic SDK kept, commented)
|
+-- reporting/
|   +-- console_reporter.py      # Formatted console output per scenario
|   +-- file_reporter.py         # JSON + CSV file writer
|
+-- tests/
|   +-- conftest.py              # Shared pytest fixtures
|   +-- test_suite.py            # 75 tests across 9 classes
|
+-- .streamlit/
|   +-- config.toml              # Dark theme for Streamlit Cloud
|
+-- output/                      # Created at runtime
    +-- results_<timestamp>.json
    +-- wave1_roadmap_<timestamp>.csv
```

---

## How It Works

### Standard mode — `python main.py`

For each of the 22 scenarios:

```
1. Load 18 investment projects from investment_pipeline.xlsx

2. Apply scenario modifiers:
      adjusted_CF    = base_CF    x cash_flow_modifier
      adjusted_capex = base_capex x capex_modifier
      adjusted_rate  = base_rate  + discount_rate_delta

3. Calculate scenario-adjusted NPV for every project:
      NPV = sum( CF_t / (1+r)^t )  -  capex

4. Rank projects by Profitability Index:
      PI = NPV / capex

5. Greedy budget allocation:
      Select highest-PI projects until budget is exhausted

6. Monte Carlo stress test (10,000 iterations):
      Shock each cash flow: CF_shocked = CF x (1 + sigma x epsilon),  epsilon ~ N(0,1)
      Report: mean, std dev, P5, P95, deficit probability

7. Print scenario report, save JSON + CSV
```

### Agentic RAG mode — `python main.py --query "..."`

```
1. User submits natural language query

2. Standard RAG (shown for contrast):
      TF-IDF embed query -> cosine similarity -> top-1 scenario
      Limitation: drops multi-component stressors silently

3. Agentic RAG:
      Step 1  LLM plans search terms from the query
      Step 2  TF-IDF vector search on planned terms -> top-3 candidates
      Step 3  LLM scores confidence 0.0-1.0 on retrieved candidates
      Step 4  If confidence < 0.70: LLM refines terms, repeat (max 3 attempts)
      Step 5  LLM composes custom Scenario from retrieved candidates

4. Custom Scenario passed to financial engine -> NPV -> MC -> results
```

---

## Data Layer — Excel Pipeline

**`investment_pipeline.xlsx`** is the project data source. Each row is one investment project.
The file ships with 18 sample UK-flavoured projects. Replace them with your firm's real pipeline —
no code changes required.

### What the Excel file is for

Provides the **investment projects** (the *what* to invest in):
- Project names, sectors, upfront costs
- 8 years of projected net cash flows
- Risk-adjusted discount rates
- Volatility estimates for Monte Carlo

### What the Excel file is NOT for

The **22 scenarios** (the *conditions* to test under) are defined separately in `data/scenarios.py`
and are what the agentic RAG searches through. The Excel file has no role in RAG retrieval.

### Column Reference

| Column | Type | Example | Description |
|---|---|---|---|
| `id` | Text | `P01` | Unique project identifier |
| `name` | Text | `National Grid Digitalisation` | Project name shown in all reports |
| `sector` | Text | `Infrastructure` | Must match scenario sector filters |
| `capex` | Number | `5000000` | Total upfront capital in GBP (no commas) |
| `Y1` | Number | `12295000` | Net cash inflow Year 1 (revenue minus operating costs) |
| `Y2` to `Y8` | Number | `15369000` | Net cash inflows Years 2-8 (use 0 if project ends earlier) |
| `discount_rate` | Decimal | `0.075` | Risk-adjusted required return (8% = 0.08) |
| `risk_sigma` | Decimal | `0.11` | Cash flow volatility for Monte Carlo shocks |

### Valid Sectors

```
Infrastructure    Renewables    Technology    Real Estate    M&A
```

### Risk Sigma Guide

| Value | Risk Profile | Typical use |
|---|---|---|
| 0.09 - 0.11 | Very stable | Regulated infrastructure, REITs |
| 0.12 - 0.14 | Moderate | Energy, logistics, cloud platforms |
| 0.15 - 0.17 | Higher risk | Technology, SaaS, healthcare |
| 0.18 - 0.20 | High risk | M&A, early-stage, emerging markets |

---

## The 22 Scenarios

Each scenario applies four scalar modifiers to every project before the optimiser runs.
No code changes are needed between scenarios — the agent reads all 22 from `data/scenarios.py`.

| # | Scenario | CF x | DR + | Capex x | Sigma x | Sector Filter |
|---|---|---|---|---|---|---|
| 1 | Base Case | 1.00 | 0.000 | 1.00 | 1.00 | All |
| 2 | Optimistic Growth | 1.15 | -0.010 | 1.00 | 0.80 | All |
| 3 | Pessimistic Contraction | 0.85 | +0.020 | 1.05 | 1.30 | All |
| 4 | Tech Boom | 1.20 | -0.010 | 0.95 | 0.90 | Technology |
| 5 | Tech Slump | 0.80 | +0.030 | 1.10 | 1.40 | Technology |
| 6 | Energy Transition Acceleration | 1.18 | -0.010 | 0.95 | 0.85 | Renewables |
| 7 | Energy Price Shock | 0.88 | +0.020 | 1.08 | 1.25 | Renewables, Infrastructure |
| 8 | M&A Wave | 1.12 | 0.000 | 1.05 | 1.10 | M&A, Technology |
| 9 | M&A Freeze | 0.90 | +0.010 | 1.15 | 1.20 | M&A |
| 10 | Recession - Mild | 0.90 | +0.015 | 1.03 | 1.15 | All |
| 11 | Recession - Severe | 0.75 | +0.030 | 1.10 | 1.50 | All |
| 12 | Interest Rate Surge (+200bps) | 1.00 | +0.020 | 1.00 | 1.10 | All |
| 13 | Interest Rate Cut (-100bps) | 1.00 | -0.010 | 1.00 | 0.90 | All |
| 14 | Infrastructure Supercycle | 1.15 | 0.000 | 0.95 | 0.90 | Infrastructure |
| 15 | Real Estate Correction | 0.82 | +0.020 | 1.10 | 1.30 | Real Estate |
| 16 | ESG Premium | 1.10 | -0.010 | 1.00 | 0.80 | Renewables |
| 17 | Regulatory Headwind | 0.92 | +0.010 | 1.08 | 1.15 | All |
| 18 | Supply Chain Disruption | 0.88 | +0.010 | 1.12 | 1.25 | All |
| 19 | Digital Transformation Wave | 1.14 | 0.000 | 0.95 | 0.85 | Technology |
| 20 | Geopolitical Stress | 0.87 | +0.020 | 1.10 | 1.35 | All |
| 21 | Currency Depreciation | 0.93 | +0.010 | 1.00 | 1.20 | All |
| 22 | Green Bonds Stimulus | 1.12 | -0.015 | 0.97 | 0.85 | Renewables, Infrastructure |

---

## Budget-Constrained Optimiser

**File:** `engine/optimizer.py`

Implements a greedy knapsack algorithm ranked by **Profitability Index (PI = NPV / capex)**,
measuring strategic value delivered per pound of capital invested.

**Base-case selection trace:**

| Rank | Project | PI | Capex | Running Total |
|---|---|---|---|---|
| 1 | Tier 3 Data Centre Campus | 23.75x | 4.0M | 4.0M |
| 2 | South West Solar Farm Portfolio | 22.00x | 4.0M | 8.0M |
| 3 | AI Asset Monitoring Platform | 21.25x | 4.0M | 12.0M |
| 4 | Azure Cloud ERP Migration | 20.89x | 4.5M | 16.5M |
| 5 | Urban 5G Fibre Network Rollout | 20.60x | 5.0M | 21.5M |
| 6 | Grid-Scale Battery Storage | 20.50x | 4.0M | 25.5M |
| 7 | National Grid Digitalisation | 20.00x | 5.0M | 30.5M |
| 8 | Midlands Distribution Hub | 20.00x | 4.5M | 35.0M |
| 9 | North Sea Offshore Wind Phase-II | 19.40x | 5.0M | 40.0M |
| 10 | NHS SaaS Efficiency Platform | 19.00x | 5.0M | 45.0M |

Total: £45,000,000 capital deployed, £929,000,000 NPV.

---

## Monte Carlo Shock Engine

**File:** `engine/monte_carlo.py`

Fully vectorised NumPy implementation across a `(10,000 x n_assets x 8 years)` tensor.
No Python loops over iterations — the entire simulation runs as a single NumPy tensor operation.

**Shock formula:**
```
CF_shocked[n, a, t] = CF[a, t] x (1 + sigma_eff[a] x epsilon[n, a, t])

where:
  n              = iteration index (0 to 9,999)
  a              = asset index
  t              = year index (0 to 7)
  epsilon        ~ N(0,1)  independent standard normal shock per CF
  sigma_eff[a]   = asset.risk_sigma x scenario.risk_sigma_multiplier
```

**Base-case output:**
```
Mean portfolio NPV    :   928,943,777 GBP
Std dev               :    14,013,510 GBP
5th percentile  (P05) :   906,358,375 GBP  <- worst 5% of outcomes
95th percentile (P95) :   952,071,382 GBP
Deficit probability   :        0.0000%
```

---

## Agentic RAG — Natural Language Query

**Files:** `retrieval/`, `llm/claude_client.py`

Allows querying the agent in plain English. The retrieval layer resolves the query into a custom
`Scenario` object, which the financial engine runs identically to any of the 22 hardcoded ones.

### The failure mode that motivated the rebuild (Standard RAG)

An early implementation used direct TF-IDF vector lookup — mapping a query to the single closest
scenario with no reasoning step:

```
Query:  "moderate tech slowdown with rising interest rates"
Standard RAG -> "Recession - Mild"   (drops both the tech and rate components entirely)
```

This caused corrupted NPV calculations downstream when queries described multi-component economic
conditions. The standard RAG was kept in `retrieval/standard_rag.py` as a baseline for comparison.

### How the agentic layer resolves it

```
Step 1  plan_retrieval     LLM analyses query, produces targeted search terms
Step 2  vector search      TF-IDF retrieves top-3 scenario candidates
Step 3  score_confidence   LLM scores 0.0-1.0 -- does this cover all stressors?
Step 4  re-query loop      If score < 0.70: LLM refines terms, repeat (max 3 attempts)
Step 5  compose_scenario   LLM blends candidate parameters into a single Scenario
```

Agentic result for the same query:
```
Composed: "Tech Slowdown + Rate Pressure"
  cash_flow_modifier    = 0.90    (blends Tech Slump 0.80 + Rate Surge 1.00)
  discount_rate_delta   = +0.025  (blends +0.03 and +0.02)
  capex_modifier        = 1.05
  risk_sigma_multiplier = 1.25
  confidence            = 0.84,  attempts = 1
```

The agent only proceeds to the financial engine once confidence is sufficient, preventing
unreliable context from reaching the NPV calculations.

### LLM backend

| Model | Role |
|---|---|
| `llama-3.1-8b-instant` (Groq) | Fast — query planning and refinement |
| `llama-3.3-70b-versatile` (Groq) | Accurate — confidence scoring and scenario composition |

Groq provides free API access at [console.groq.com](https://console.groq.com). The Anthropic SDK
implementation with prompt caching is kept in `llm/claude_client.py` (commented) for reference.

### Usage

```bash
set GROQ_API_KEY=gsk_...
python main.py --query "What happens to our portfolio in a tech crash with rising rates?"
```

---

## Web App — Streamlit Dashboard

**File:** `app.py`

A Bloomberg-style dark financial dashboard built with Streamlit and Plotly.

### Four tabs

| Tab | Contents |
|---|---|
| **Portfolio Dashboard** | Per-scenario progress bar, Wave-1 KPI cards, NPV waterfall chart, sector allocation donut, Monte Carlo distribution histogram (P05/Mean/P95), 8-year stacked cashflow area chart, portfolio table with IRR column, scenario drill-down expander |
| **Scenario Analysis** | 22-scenario horizontal bar (red-to-green gradient), risk-return scatter plot (base case starred), scenario metrics heatmap, side-by-side scenario A vs B comparison |
| **Natural Language Query** | Standard RAG card (red) vs Agentic RAG card (green), composed scenario parameters, full NPV/MC results, query vs base-case delta chart |
| **About** | Architecture diagram, key metrics table, tech stack |

### Run locally

```bash
streamlit run app.py
```

### Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Set repository: `Hemang417/Capital_Portfolio_Optimisation_Agent`, branch: `main`, file: `app.py`
4. Click **Advanced settings** → **Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** — live URL in ~2 minutes

---

## Test Suite

**Files:** `tests/conftest.py`, `tests/test_suite.py`

A rigorous pytest suite covering correctness, edge cases, adversarial inputs, and statistical
invariants across every module. **75 tests across 9 classes — all passing in under 5 seconds.**

```bash
python -m pytest tests/test_suite.py -v
```

### Coverage by class

| Class | Tests | What is verified |
|---|---|---|
| `TestNPVEngine` | 12 | Known-value NPV assertions (hand-computed), IRR binary search convergence, NaN on no sign-change, profitability index formula, 10,000-call vectorisation performance |
| `TestOptimizer` | 14 | PI ordering, budget never exceeded, `per_asset_npv` integrity, real 929M base-case assertion, empty portfolio, all-negative NPV, exact budget fit, over-budget rejection, stable sort on PI ties, 1,000-asset performance |
| `TestMonteCarlo` | 11 | Zero deficit on real base-case portfolio, MC mean within 2% of deterministic NPV, P05 < mean < P95 ordering, reproducibility with fixed seed, zero-sigma degeneracy, n=10k vs n=50k convergence, empty-asset short-circuit |
| `TestScenarios` | 8 | Exactly 22 scenarios, unique IDs 1–22, base-case neutral modifiers, severe recession has lowest CF modifier, `to_document()` contents, valid sector names, modifier sanity ranges |
| `TestAssets` | 6 | 18 assets loaded, unique IDs, 8-year cash flows, discount rate and sigma in valid ranges, `FileNotFoundError` on missing Excel |
| `TestScenarioStore` | 7 | `top_k` count, cosine scores ∈ [0,1], sorted descending, known-query semantic routing, empty query no crash |
| `TestStandardRAG` | 4 | Return type, semantic routing, limitation proof (single result for multi-component query) |
| `TestAgenticRAG` | 8 | All LLM calls mocked — confident first attempt skips re-query, low confidence triggers refinement loop, `MAX_ATTEMPTS` respected, bad-JSON fallback, `CONFIDENCE_THRESHOLD = 0.70`, candidates list populated |
| `TestIntegration` | 5 | Full real-data pipeline: base case 929M, all 22 scenarios complete without error, recession < base < optimistic NPV ordering, sector filter enforced in tech-only scenario |

### Sample output

```
tests/test_suite.py::TestNPVEngine::test_npv_zero_cashflows PASSED
tests/test_suite.py::TestNPVEngine::test_npv_single_year_breakeven PASSED
tests/test_suite.py::TestNPVEngine::test_irr_known_value PASSED
...
tests/test_suite.py::TestIntegration::test_base_case_npv_is_929M PASSED
tests/test_suite.py::TestIntegration::test_sector_filter_respected_tech_only PASSED

75 passed in 4.52s
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Hemang417/Capital_Portfolio_Optimisation_Agent.git
cd Capital_Portfolio_Optimisation_Agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the agent (investment_pipeline.xlsx included)
python main.py

# 4. Or launch the web dashboard
streamlit run app.py

# 5. Run the test suite
python -m pytest tests/test_suite.py -v
```

---

## Running the Agent

```bash
# Default run -- all 22 scenarios, £45M budget, 10,000 MC iterations
python main.py

# Custom budget
python main.py --budget 60000000

# Custom Monte Carlo iterations
python main.py --mc-iterations 50000

# Natural language query mode (requires GROQ_API_KEY)
python main.py --query "What if there is a tech crash with rising interest rates?"

# Combine flags
python main.py --budget 75000000 --mc-iterations 20000 --seed 7
```

---

## Output Files

All output is saved to the `output/` directory, created automatically on first run.

### `results_<timestamp>.json`
```json
{
  "scenario_id": 1,
  "scenario_name": "Base Case",
  "optimization": {
    "selected_asset_ids": ["P10", "P06", "P05", "..."],
    "total_capital_deployed_gbp": 45000000,
    "total_npv_gbp": 929000000,
    "return_multiplier": 20.64
  },
  "monte_carlo": {
    "deficit_probability": 0.0,
    "mean_total_npv_gbp": 928943777,
    "p5_total_npv_gbp": 906358375,
    "p95_total_npv_gbp": 952071382
  }
}
```

### `wave1_roadmap_<timestamp>.csv`

| Column | Description |
|---|---|
| `scenario_id` / `scenario_name` | Scenario identifier |
| `capital_deployed_gbp` | Total capital allocated |
| `total_npv_gbp` | Total portfolio NPV |
| `return_multiplier` | NPV divided by capital |
| `mc_mean_npv_gbp` | Monte Carlo mean NPV |
| `deficit_probability` | Fraction of MC runs with negative total NPV |
| `selected_asset_ids` | Pipe-separated list of selected project IDs |

---

## Using Real Firm Data

1. Open `investment_pipeline.xlsx`
2. Replace the sample rows (P01–P18) with your firm's actual investment projects
3. Run `python main.py` — no code changes required

To regenerate the sample template:
```bash
python create_sample_data.py
```

---

## Adjusting the Model

| What to change | How |
|---|---|
| Capital budget (£45M) | `python main.py --budget 60000000` or edit `BUDGET_GBP` in `engine/optimizer.py` |
| Portfolio NPV (£929M) | Increase cash flows Y1–Y8 in `investment_pipeline.xlsx` |
| Return multiplier (20.64×) | Derived automatically as Total NPV / Budget |
| Deficit probability | Lower `risk_sigma` in Excel, or increase cash flows relative to capex |
| Scenario definitions | Edit `data/scenarios.py` |
| Sector filters per scenario | Edit `eligible_sectors` in `data/scenarios.py` |
| MC iterations | `python main.py --mc-iterations 50000` |
| LLM backend | Switch Groq / Anthropic in `llm/claude_client.py` (both implementations present) |

---

## Dependencies

```
numpy>=1.26.0       vectorised NPV and Monte Carlo calculations
pandas>=2.0.0       Excel pipeline reading
openpyxl>=3.1.0     Excel file support (.xlsx)
groq>=0.9.0         Groq API for agentic RAG (free tier)
scikit-learn>=1.3.0 TF-IDF vector index for scenario retrieval
streamlit>=1.35.0   Web dashboard
plotly>=5.18.0      Interactive financial charts
pytest>=7.4.0       Test suite
```

```bash
pip install -r requirements.txt
```

---

*MSc Financial Data Analytics*
