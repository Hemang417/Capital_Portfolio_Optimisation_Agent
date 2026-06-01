from dataclasses import dataclass, field
from typing import List


@dataclass
class Scenario:
    id: int
    name: str
    description: str
    cash_flow_modifier: float
    discount_rate_delta: float
    capex_modifier: float
    risk_sigma_multiplier: float
    eligible_sectors: List[str] = field(default_factory=list)


def get_all_scenarios() -> List[Scenario]:
    """
    22 operational scenarios processed autonomously without manual reconfiguration.

    Each scenario applies four scalar modifiers to every asset before the
    optimiser runs:
      cash_flow_modifier    — multiplied against each annual cash flow
      discount_rate_delta   — added to each asset's base discount rate
      capex_modifier        — multiplied against each asset's capital_required
      risk_sigma_multiplier — scales each asset's risk_sigma for Monte Carlo

    eligible_sectors: if non-empty, restricts the optimiser to assets whose
    sector is in this list (simulates a sector-specific investment mandate).
    An empty list means all sectors are eligible.
    """
    return [
        Scenario(
            id=1, name="Base Case",
            description="Reference run — all modifiers at unity, all sectors eligible",
            cash_flow_modifier=1.00, discount_rate_delta=0.000,
            capex_modifier=1.00, risk_sigma_multiplier=1.00,
        ),
        Scenario(
            id=2, name="Optimistic Growth",
            description="Bull market: enhanced cash flows, lower risk, compressed spreads",
            cash_flow_modifier=1.15, discount_rate_delta=-0.010,
            capex_modifier=1.00, risk_sigma_multiplier=0.80,
        ),
        Scenario(
            id=3, name="Pessimistic Contraction",
            description="Macro downturn: reduced revenues, higher discount rates, cost inflation",
            cash_flow_modifier=0.85, discount_rate_delta=+0.020,
            capex_modifier=1.05, risk_sigma_multiplier=1.30,
        ),
        Scenario(
            id=4, name="Tech Boom",
            description="AI/SaaS valuations surge — technology-only mandate, enhanced CFs",
            cash_flow_modifier=1.20, discount_rate_delta=-0.010,
            capex_modifier=0.95, risk_sigma_multiplier=0.90,
            eligible_sectors=["Technology"],
        ),
        Scenario(
            id=5, name="Tech Slump",
            description="Post-bubble correction — technology-only mandate, degraded CFs",
            cash_flow_modifier=0.80, discount_rate_delta=+0.030,
            capex_modifier=1.10, risk_sigma_multiplier=1.40,
            eligible_sectors=["Technology"],
        ),
        Scenario(
            id=6, name="Energy Transition Acceleration",
            description="Policy tailwind drives renewables — green-energy mandate, cost reduction",
            cash_flow_modifier=1.18, discount_rate_delta=-0.010,
            capex_modifier=0.95, risk_sigma_multiplier=0.85,
            eligible_sectors=["Renewables"],
        ),
        Scenario(
            id=7, name="Energy Price Shock",
            description="Supply disruption hits energy & infrastructure cash flows",
            cash_flow_modifier=0.88, discount_rate_delta=+0.020,
            capex_modifier=1.08, risk_sigma_multiplier=1.25,
            eligible_sectors=["Renewables", "Infrastructure"],
        ),
        Scenario(
            id=8, name="M&A Wave",
            description="Consolidation premium lifts M&A and technology deal flow",
            cash_flow_modifier=1.12, discount_rate_delta=0.000,
            capex_modifier=1.05, risk_sigma_multiplier=1.10,
            eligible_sectors=["M&A", "Technology"],
        ),
        Scenario(
            id=9, name="M&A Freeze",
            description="Deal activity collapses — only M&A pipeline assets considered",
            cash_flow_modifier=0.90, discount_rate_delta=+0.010,
            capex_modifier=1.15, risk_sigma_multiplier=1.20,
            eligible_sectors=["M&A"],
        ),
        Scenario(
            id=10, name="Recession — Mild",
            description="GDP contraction -1.5%: moderate cash flow erosion across all sectors",
            cash_flow_modifier=0.90, discount_rate_delta=+0.015,
            capex_modifier=1.03, risk_sigma_multiplier=1.15,
        ),
        Scenario(
            id=11, name="Recession — Severe",
            description="GDP contraction -4%, credit crunch: significant stress across all sectors",
            cash_flow_modifier=0.75, discount_rate_delta=+0.030,
            capex_modifier=1.10, risk_sigma_multiplier=1.50,
        ),
        Scenario(
            id=12, name="Interest Rate Surge (+200bps)",
            description="Central bank tightening: discount rates elevated, CFs unchanged",
            cash_flow_modifier=1.00, discount_rate_delta=+0.020,
            capex_modifier=1.00, risk_sigma_multiplier=1.10,
        ),
        Scenario(
            id=13, name="Interest Rate Cut (-100bps)",
            description="Easing cycle: lower discount rates boost NPVs across all assets",
            cash_flow_modifier=1.00, discount_rate_delta=-0.010,
            capex_modifier=1.00, risk_sigma_multiplier=0.90,
        ),
        Scenario(
            id=14, name="Infrastructure Supercycle",
            description="Government capex plans drive infrastructure demand — sector mandate",
            cash_flow_modifier=1.15, discount_rate_delta=0.000,
            capex_modifier=0.95, risk_sigma_multiplier=0.90,
            eligible_sectors=["Infrastructure"],
        ),
        Scenario(
            id=15, name="Real Estate Correction",
            description="Property price decline hits real estate cash flows — sector mandate",
            cash_flow_modifier=0.82, discount_rate_delta=+0.020,
            capex_modifier=1.10, risk_sigma_multiplier=1.30,
            eligible_sectors=["Real Estate"],
        ),
        Scenario(
            id=16, name="ESG Premium",
            description="ESG capital inflows compress renewable energy discount rates",
            cash_flow_modifier=1.10, discount_rate_delta=-0.010,
            capex_modifier=1.00, risk_sigma_multiplier=0.80,
            eligible_sectors=["Renewables"],
        ),
        Scenario(
            id=17, name="Regulatory Headwind",
            description="New compliance costs reduce cash flows and raise capex across all sectors",
            cash_flow_modifier=0.92, discount_rate_delta=+0.010,
            capex_modifier=1.08, risk_sigma_multiplier=1.15,
        ),
        Scenario(
            id=18, name="Supply Chain Disruption",
            description="Logistics cost surge inflates capex and reduces operational CFs",
            cash_flow_modifier=0.88, discount_rate_delta=+0.010,
            capex_modifier=1.12, risk_sigma_multiplier=1.25,
        ),
        Scenario(
            id=19, name="Digital Transformation Wave",
            description="Enterprise digitisation surge — technology-only mandate, strong CFs",
            cash_flow_modifier=1.14, discount_rate_delta=0.000,
            capex_modifier=0.95, risk_sigma_multiplier=0.85,
            eligible_sectors=["Technology"],
        ),
        Scenario(
            id=20, name="Geopolitical Stress",
            description="Sanctions and trade war reduce cross-border cash flows, raise risk",
            cash_flow_modifier=0.87, discount_rate_delta=+0.020,
            capex_modifier=1.10, risk_sigma_multiplier=1.35,
        ),
        Scenario(
            id=21, name="Currency Depreciation",
            description="GBP weakness vs USD/EUR erodes imported-input cash flows",
            cash_flow_modifier=0.93, discount_rate_delta=+0.010,
            capex_modifier=1.00, risk_sigma_multiplier=1.20,
        ),
        Scenario(
            id=22, name="Green Bonds Stimulus",
            description="Subsidised green finance boosts renewables and infrastructure CFs",
            cash_flow_modifier=1.12, discount_rate_delta=-0.015,
            capex_modifier=0.97, risk_sigma_multiplier=0.85,
            eligible_sectors=["Renewables", "Infrastructure"],
        ),
    ]
