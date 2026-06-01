"""
Generates investment_pipeline.xlsx — the data file the agent reads at runtime.

To use your firm's real data:
  1. Open investment_pipeline.xlsx
  2. Replace rows P01–P18 with your actual investment projects
  3. Run: python main.py

Column guide:
  id            — unique project code (any string, e.g. "PRJ-001")
  name          — project name
  sector        — one of: Infrastructure, Technology, Renewables,
                  Real Estate, M&A  (must match scenario sector filters)
  capex         — total upfront capital required in GBP (no £ sign, no commas)
  Y1 … Y8      — net annual cash flows in GBP for years 1–8
                  (revenue minus operating costs, not including the capex)
                  use 0 for years beyond the project's life
  discount_rate — risk-adjusted required return as a decimal (e.g. 0.08 = 8%)
  risk_sigma    — cash flow volatility for Monte Carlo shocks
                  0.10 = stable infrastructure
                  0.15 = moderate (technology / energy)
                  0.20 = high risk (M&A / early-stage)
"""

import pandas as pd


def _cfs(capex: float, target_npv: float, r: float):
    """Compute 8-year cash flows that yield exactly target_npv at rate r."""
    pattern = [0.8, 1.0, 1.2, 1.3, 1.4, 1.4, 1.3, 1.1]
    dfs = [(1.0 + r) ** -(t + 1) for t in range(8)]
    ws = sum(p * d for p, d in zip(pattern, dfs))
    scale = (target_npv + capex) / ws
    return [round(p * scale) for p in pattern]


# ── Project definitions ───────────────────────────────────────────────────────
# Format: (id, name, sector, capex, target_npv, discount_rate, risk_sigma)
#
# The 10 high-PI projects (P01–P10) are selected by the optimiser in the base
# case, deploying exactly £45,000,000 and delivering £929,000,000 NPV.
# The 8 low-PI projects (P11–P18) are rejected — they have large capex and
# lower returns per pound invested.
#
# Replace these rows with your firm's real pipeline data.
# ─────────────────────────────────────────────────────────────────────────────
PROJECTS = [
    # ── Selected portfolio (high PI) ─────────────────────────────────────────
    ("P01", "National Grid Digitalisation",     "Infrastructure",  5_000_000, 100_000_000, 0.075, 0.11),
    ("P02", "North Sea Offshore Wind Phase-II", "Renewables",      5_000_000,  97_000_000, 0.080, 0.13),
    ("P03", "Azure Cloud ERP Migration",        "Technology",      4_500_000,  94_000_000, 0.090, 0.14),
    ("P04", "Midlands Distribution Hub",        "Infrastructure",  4_500_000,  90_000_000, 0.075, 0.10),
    ("P05", "AI Asset Monitoring Platform",     "Technology",      4_000_000,  85_000_000, 0.090, 0.15),
    ("P06", "South West Solar Farm Portfolio",  "Renewables",      4_000_000,  88_000_000, 0.075, 0.12),
    ("P07", "Urban 5G Fibre Network Rollout",   "Infrastructure",  5_000_000, 103_000_000, 0.075, 0.11),
    ("P08", "NHS SaaS Efficiency Platform",     "Technology",      5_000_000,  95_000_000, 0.095, 0.16),
    ("P09", "Grid-Scale Battery Storage",       "Renewables",      4_000_000,  82_000_000, 0.080, 0.13),
    ("P10", "Tier 3 Data Centre Campus",        "Technology",      4_000_000,  95_000_000, 0.080, 0.12),
    # ── Rejected portfolio (low PI — large capex, lower return ratio) ────────
    ("P11", "Biotech Acquisition Programme",    "M&A",            10_000_000,  85_000_000, 0.100, 0.20),
    ("P12", "City Office Portfolio",            "Real Estate",    12_000_000, 100_000_000, 0.070, 0.09),
    ("P13", "Factory Automation & Robotics",    "Infrastructure",  8_000_000,  64_000_000, 0.080, 0.11),
    ("P14", "High Street Retail Fund",          "Real Estate",     9_000_000,  80_000_000, 0.075, 0.10),
    ("P15", "Chemicals Plant Refurbishment",    "Infrastructure", 15_000_000,  75_000_000, 0.090, 0.14),
    ("P16", "Legacy Systems Modernisation",     "Technology",      7_000_000,  42_000_000, 0.100, 0.18),
    ("P17", "Gas Peaker Plant Investment",      "Infrastructure", 20_000_000, 120_000_000, 0.085, 0.12),
    ("P18", "Business Park REIT Acquisition",  "Real Estate",    11_000_000, 100_000_000, 0.070, 0.09),
]


def build_dataframe() -> pd.DataFrame:
    rows = []
    for pid, name, sector, capex, npv, r, sigma in PROJECTS:
        cfs = _cfs(capex, npv, r)
        rows.append({
            "id":            pid,
            "name":          name,
            "sector":        sector,
            "capex":         capex,
            "Y1":            cfs[0],
            "Y2":            cfs[1],
            "Y3":            cfs[2],
            "Y4":            cfs[3],
            "Y5":            cfs[4],
            "Y6":            cfs[5],
            "Y7":            cfs[6],
            "Y8":            cfs[7],
            "discount_rate": r,
            "risk_sigma":    sigma,
        })
    return pd.DataFrame(rows)


def main():
    df = build_dataframe()
    output_path = "investment_pipeline.xlsx"
    df.to_excel(output_path, index=False)

    print(f"Created {output_path}  ({len(df)} projects)")
    print()
    print(f"  {'ID':<4}  {'Name':<38}  {'Sector':<16}  {'Capex (£M)':>10}  {'NPV* (£M)':>10}  {'PI':>6}")
    print("  " + "-" * 90)
    for _, row in df.iterrows():
        cfs = [row[f"Y{t}"] for t in range(1, 9)]
        r = row["discount_rate"]
        pv = sum(cf / (1 + r) ** (t + 1) for t, cf in enumerate(cfs))
        npv = pv - row["capex"]
        pi = npv / row["capex"]
        print(f"  {row['id']:<4}  {row['name']:<38}  {row['sector']:<16}"
              f"  {row['capex']/1e6:>10.2f}  {npv/1e6:>10.1f}  {pi:>5.2f}x")

    selected = df[df["id"].isin([f"P{i:02d}" for i in range(1, 11)])]
    total_capex = selected["capex"].sum()
    print()
    print(f"  Base-case portfolio (P01–P10):")
    print(f"    Total capex : £{total_capex:>14,.0f}")
    print(f"  * NPV figures are indicative base-case values at stated discount rates.")
    print()
    print("To run the agent:")
    print("  python main.py")
    print()
    print("To use your firm's real data:")
    print("  1. Open investment_pipeline.xlsx")
    print("  2. Replace rows with your actual project pipeline")
    print("  3. Run: python main.py  (no code changes needed)")


if __name__ == "__main__":
    main()
