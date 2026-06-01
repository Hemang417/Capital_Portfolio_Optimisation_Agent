from dataclasses import dataclass, field
from typing import Dict, List

from data.assets import Asset

BUDGET_GBP: float = 45_000_000.0


@dataclass
class OptimizationResult:
    selected_assets: List[Asset]
    total_capital_deployed: float
    total_npv: float
    remaining_budget: float
    profitability_indices: Dict[str, float]


def greedy_knapsack_optimizer(
    assets: List[Asset],
    scenario_npvs: Dict[str, float],
    scenario_capex: Dict[str, float],
    budget: float = BUDGET_GBP,
) -> OptimizationResult:
    """
    Greedy capital allocation sorted by profitability index (NPV / capex).

    Assets with non-positive scenario-adjusted NPV are excluded before sorting.
    Python's sort is stable, so ties in PI preserve the original asset-pool order.
    The loop tries every remaining candidate at each step, so smaller assets are
    considered after larger ones fail to fit — naturally filling budget gaps.
    """
    viable = [a for a in assets if scenario_npvs.get(a.id, 0.0) > 0.0]

    pis: Dict[str, float] = {
        a.id: scenario_npvs[a.id] / scenario_capex[a.id]
        for a in viable
    }

    sorted_assets = sorted(viable, key=lambda a: pis[a.id], reverse=True)

    selected: List[Asset] = []
    total_capex = 0.0
    total_npv = 0.0

    for asset in sorted_assets:
        capex = scenario_capex[asset.id]
        if total_capex + capex <= budget + 1e-2:
            selected.append(asset)
            total_capex += capex
            total_npv += scenario_npvs[asset.id]

    return OptimizationResult(
        selected_assets=selected,
        total_capital_deployed=total_capex,
        total_npv=total_npv,
        remaining_budget=budget - total_capex,
        profitability_indices=pis,
    )
