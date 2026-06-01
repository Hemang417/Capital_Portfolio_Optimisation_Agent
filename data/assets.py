from dataclasses import dataclass
from typing import List


@dataclass
class Asset:
    id: str
    name: str
    sector: str
    capital_required: float
    annual_cash_flows: List[float]
    discount_rate: float
    risk_sigma: float


def _cfs(capex: float, npv: float, r: float) -> List[float]:
    """Generate 8-year cash flows whose NPV at rate r equals the target."""
    pattern = [0.8, 1.0, 1.2, 1.3, 1.4, 1.4, 1.3, 1.1]
    dfs = [(1.0 + r) ** -(t + 1) for t in range(8)]
    weighted_sum = sum(p * d for p, d in zip(pattern, dfs))
    scale = (npv + capex) / weighted_sum
    return [p * scale for p in pattern]


def get_asset_pool() -> List[Asset]:
    """
    18 strategic investment opportunities.

    Assets A01-A10 form the optimal base-case portfolio:
      total capex = 45,000,000 GBP
      total NPV   = 929,000,000 GBP  (20.64x return)

    Assets A11-A18 have lower profitability indices and are rejected by the
    greedy optimiser in unconstrained (all-sector) scenarios.
    """
    return [
        # ── High-PI assets (selected in base case) ──────────────────────────
        Asset(
            "A01", "National Grid Digitalisation", "Infrastructure",
            capital_required=5_000_000,
            annual_cash_flows=_cfs(5_000_000, 100_000_000, 0.075),
            discount_rate=0.075,
            risk_sigma=0.11,
        ),
        Asset(
            "A02", "Offshore Wind Phase-II", "Renewables",
            capital_required=5_000_000,
            annual_cash_flows=_cfs(5_000_000, 97_000_000, 0.080),
            discount_rate=0.080,
            risk_sigma=0.13,
        ),
        Asset(
            "A03", "Cloud ERP Rollout", "Technology",
            capital_required=4_500_000,
            annual_cash_flows=_cfs(4_500_000, 94_000_000, 0.090),
            discount_rate=0.090,
            risk_sigma=0.14,
        ),
        Asset(
            "A04", "Logistics Hub Expansion", "Infrastructure",
            capital_required=4_500_000,
            annual_cash_flows=_cfs(4_500_000, 90_000_000, 0.075),
            discount_rate=0.075,
            risk_sigma=0.10,
        ),
        Asset(
            "A05", "AI Predictive Maintenance", "Technology",
            capital_required=4_000_000,
            annual_cash_flows=_cfs(4_000_000, 85_000_000, 0.090),
            discount_rate=0.090,
            risk_sigma=0.15,
        ),
        Asset(
            "A06", "Solar Farm Portfolio", "Renewables",
            capital_required=4_000_000,
            annual_cash_flows=_cfs(4_000_000, 88_000_000, 0.075),
            discount_rate=0.075,
            risk_sigma=0.12,
        ),
        Asset(
            "A07", "Smart City Fibre Rollout", "Infrastructure",
            capital_required=5_000_000,
            annual_cash_flows=_cfs(5_000_000, 103_000_000, 0.075),
            discount_rate=0.075,
            risk_sigma=0.11,
        ),
        Asset(
            "A08", "Healthcare SaaS Platform", "Technology",
            capital_required=5_000_000,
            annual_cash_flows=_cfs(5_000_000, 95_000_000, 0.095),
            discount_rate=0.095,
            risk_sigma=0.16,
        ),
        Asset(
            "A09", "Battery Storage Network", "Renewables",
            capital_required=4_000_000,
            annual_cash_flows=_cfs(4_000_000, 82_000_000, 0.080),
            discount_rate=0.080,
            risk_sigma=0.13,
        ),
        Asset(
            "A10", "Data Centre Co-location", "Technology",
            capital_required=4_000_000,
            annual_cash_flows=_cfs(4_000_000, 95_000_000, 0.080),
            discount_rate=0.080,
            risk_sigma=0.12,
        ),
        # ── Low-PI assets (rejected by greedy optimiser in base case) ───────
        Asset(
            "A11", "M&A Biotech Roll-up", "M&A",
            capital_required=10_000_000,
            annual_cash_flows=_cfs(10_000_000, 85_000_000, 0.100),
            discount_rate=0.100,
            risk_sigma=0.20,
        ),
        Asset(
            "A12", "Real Estate Portfolio", "Real Estate",
            capital_required=12_000_000,
            annual_cash_flows=_cfs(12_000_000, 100_000_000, 0.070),
            discount_rate=0.070,
            risk_sigma=0.09,
        ),
        Asset(
            "A13", "Manufacturing Automation", "Infrastructure",
            capital_required=8_000_000,
            annual_cash_flows=_cfs(8_000_000, 64_000_000, 0.080),
            discount_rate=0.080,
            risk_sigma=0.11,
        ),
        Asset(
            "A14", "Retail Property Fund", "Real Estate",
            capital_required=9_000_000,
            annual_cash_flows=_cfs(9_000_000, 80_000_000, 0.075),
            discount_rate=0.075,
            risk_sigma=0.10,
        ),
        Asset(
            "A15", "Chemical Plant Upgrade", "Infrastructure",
            capital_required=15_000_000,
            annual_cash_flows=_cfs(15_000_000, 75_000_000, 0.090),
            discount_rate=0.090,
            risk_sigma=0.14,
        ),
        Asset(
            "A16", "Legacy IT Refresh", "Technology",
            capital_required=7_000_000,
            annual_cash_flows=_cfs(7_000_000, 42_000_000, 0.100),
            discount_rate=0.100,
            risk_sigma=0.18,
        ),
        Asset(
            "A17", "Conventional Power Plant", "Infrastructure",
            capital_required=20_000_000,
            annual_cash_flows=_cfs(20_000_000, 120_000_000, 0.085),
            discount_rate=0.085,
            risk_sigma=0.12,
        ),
        Asset(
            "A18", "Office REIT Acquisition", "Real Estate",
            capital_required=11_000_000,
            annual_cash_flows=_cfs(11_000_000, 100_000_000, 0.070),
            discount_rate=0.070,
            risk_sigma=0.09,
        ),
    ]
