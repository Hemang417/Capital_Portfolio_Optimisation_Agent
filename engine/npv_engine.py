from typing import List

import numpy as np


def calculate_npv(
    cash_flows: List[float],
    initial_investment: float,
    discount_rate: float,
) -> float:
    """
    NPV = Σ CF_t / (1+r)^t  −  initial_investment
    cash_flows[0] is Year-1 cash flow (t=1).
    """
    if not cash_flows:
        return -initial_investment
    t = np.arange(1, len(cash_flows) + 1)
    pv = np.sum(np.array(cash_flows) / (1.0 + discount_rate) ** t)
    return float(pv - initial_investment)


def calculate_irr(
    cash_flows_with_capex: List[float],
    tolerance: float = 1e-6,
    max_iter: int = 1_000,
) -> float:
    """
    Internal rate of return via binary search.
    cash_flows_with_capex[0] is the initial outflow (negative capex).
    Returns float('nan') if no sign change is found in [-50%, 1000%].
    """
    def _npv_at(rate: float) -> float:
        t = np.arange(len(cash_flows_with_capex))
        return float(np.sum(np.array(cash_flows_with_capex) / (1.0 + rate) ** t))

    lo, hi = -0.5, 10.0
    if _npv_at(lo) * _npv_at(hi) > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        npv_mid = _npv_at(mid)
        if abs(npv_mid) < tolerance or (hi - lo) / 2.0 < tolerance:
            return mid
        if _npv_at(lo) * npv_mid < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def npv_to_capital_ratio(npv: float, capital: float) -> float:
    """Profitability index = NPV / capital_required."""
    if capital == 0:
        return float("inf")
    return npv / capital
