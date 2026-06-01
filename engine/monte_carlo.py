from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data.assets import Asset


@dataclass
class MonteCarloResult:
    n_iterations: int
    deficit_count: int
    deficit_probability: float
    mean_total_npv: float
    std_total_npv: float
    p5_total_npv: float
    p95_total_npv: float
    all_npv_samples: np.ndarray


def run_monte_carlo(
    selected_assets: List[Asset],
    adjusted_cfs: Dict[str, List[float]],
    adjusted_capex: Dict[str, float],
    adjusted_discount_rates: Dict[str, float],
    sigma_multiplier: float,
    n_iterations: int = 10_000,
    random_seed: int = 42,
) -> MonteCarloResult:
    """
    Vectorised 10,000-iteration Monte Carlo shock engine.

    For each iteration a multiplicative shock is applied independently to
    every cash flow of every selected asset:

        CF_shocked[n, a, t] = CF[a, t] * (1 + σ_eff[a] * ε[n, a, t])

    where ε ~ N(0,1) and σ_eff[a] = asset.risk_sigma * sigma_multiplier.

    The total portfolio NPV is then summed across assets and iterations.
    deficit_probability = P(total NPV < 0).
    """
    if not selected_assets:
        empty = np.array([])
        return MonteCarloResult(n_iterations, 0, 0.0, 0.0, 0.0, 0.0, 0.0, empty)

    n_assets = len(selected_assets)
    n_years = len(adjusted_cfs[selected_assets[0].id])

    # ── Build static matrices ────────────────────────────────────────────────

    # CF matrix: (A, T)
    cf_matrix = np.array(
        [adjusted_cfs[a.id] for a in selected_assets], dtype=np.float64
    )

    # Capex vector: (A,)
    capex_vec = np.array(
        [adjusted_capex[a.id] for a in selected_assets], dtype=np.float64
    )

    # Discount matrix: (A, T)  —  denominator (1+r)^t for t = 1..T
    discount_matrix = np.array(
        [
            [(1.0 + adjusted_discount_rates[a.id]) ** (t + 1) for t in range(n_years)]
            for a in selected_assets
        ],
        dtype=np.float64,
    )

    # Effective sigma vector: (A,)
    sigma_vec = np.array(
        [a.risk_sigma * sigma_multiplier for a in selected_assets], dtype=np.float64
    )

    # ── Monte Carlo simulation ───────────────────────────────────────────────

    rng = np.random.default_rng(random_seed)

    # Shock tensor: (N, A, T)
    shocks = rng.standard_normal((n_iterations, n_assets, n_years))

    # Apply multiplicative shocks: (N, A, T)
    shocked_cfs = cf_matrix[np.newaxis, :, :] * (
        1.0 + sigma_vec[np.newaxis, :, np.newaxis] * shocks
    )

    # Present-value of each shocked CF: (N, A)
    pv_matrix = (shocked_cfs / discount_matrix[np.newaxis, :, :]).sum(axis=2)

    # NPV per asset per iteration: (N, A)
    npv_per_asset = pv_matrix - capex_vec[np.newaxis, :]

    # Total portfolio NPV per iteration: (N,)
    total_npv: np.ndarray = npv_per_asset.sum(axis=1)

    deficit_count = int((total_npv < 0.0).sum())

    return MonteCarloResult(
        n_iterations=n_iterations,
        deficit_count=deficit_count,
        deficit_probability=deficit_count / n_iterations,
        mean_total_npv=float(total_npv.mean()),
        std_total_npv=float(total_npv.std()),
        p5_total_npv=float(np.percentile(total_npv, 5)),
        p95_total_npv=float(np.percentile(total_npv, 95)),
        all_npv_samples=total_npv,
    )
