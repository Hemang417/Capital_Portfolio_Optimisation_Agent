import csv
import json
import os
from datetime import datetime
from typing import List

from models import ScenarioRunResult

_OUTPUT_DIR = "output"


def _ensure_output_dir() -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _scenario_to_dict(r: ScenarioRunResult) -> dict:
    opt = r.optimization
    mc = r.monte_carlo
    mult = opt.total_npv / opt.total_capital_deployed if opt.total_capital_deployed else 0.0
    return {
        "scenario_id": r.scenario.id,
        "scenario_name": r.scenario.name,
        "description": r.scenario.description,
        "modifiers": {
            "cash_flow_modifier": r.scenario.cash_flow_modifier,
            "discount_rate_delta": r.scenario.discount_rate_delta,
            "capex_modifier": r.scenario.capex_modifier,
            "risk_sigma_multiplier": r.scenario.risk_sigma_multiplier,
            "eligible_sectors": r.scenario.eligible_sectors,
        },
        "optimization": {
            "selected_asset_ids": [a.id for a in opt.selected_assets],
            "selected_asset_names": [a.name for a in opt.selected_assets],
            "total_capital_deployed_gbp": opt.total_capital_deployed,
            "total_npv_gbp": opt.total_npv,
            "remaining_budget_gbp": opt.remaining_budget,
            "return_multiplier": mult,
            "profitability_indices": opt.profitability_indices,
        },
        "monte_carlo": {
            "n_iterations": mc.n_iterations,
            "deficit_count": mc.deficit_count,
            "deficit_probability": mc.deficit_probability,
            "mean_total_npv_gbp": mc.mean_total_npv,
            "std_total_npv_gbp": mc.std_total_npv,
            "p5_total_npv_gbp": mc.p5_total_npv,
            "p95_total_npv_gbp": mc.p95_total_npv,
        },
    }


def save_results_json(results: List[ScenarioRunResult]) -> str:
    outdir = _ensure_output_dir()
    path = os.path.join(outdir, f"results_{_ts()}.json")
    payload = [_scenario_to_dict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)
    print(f"  [file] JSON results saved -> {path}")
    return path


def save_wave1_csv(results: List[ScenarioRunResult]) -> str:
    outdir = _ensure_output_dir()
    path = os.path.join(outdir, f"wave1_roadmap_{_ts()}.csv")
    fieldnames = [
        "scenario_id", "scenario_name",
        "capital_deployed_gbp", "total_npv_gbp", "return_multiplier",
        "mc_mean_npv_gbp", "mc_std_npv_gbp", "mc_p5_npv_gbp", "mc_p95_npv_gbp",
        "deficit_probability", "n_selected_assets", "selected_asset_ids",
        "eligible_sectors",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            opt = r.optimization
            mc = r.monte_carlo
            mult = opt.total_npv / opt.total_capital_deployed if opt.total_capital_deployed else 0.0
            writer.writerow({
                "scenario_id": r.scenario.id,
                "scenario_name": r.scenario.name,
                "capital_deployed_gbp": round(opt.total_capital_deployed, 2),
                "total_npv_gbp": round(opt.total_npv, 2),
                "return_multiplier": round(mult, 4),
                "mc_mean_npv_gbp": round(mc.mean_total_npv, 2),
                "mc_std_npv_gbp": round(mc.std_total_npv, 2),
                "mc_p5_npv_gbp": round(mc.p5_total_npv, 2),
                "mc_p95_npv_gbp": round(mc.p95_total_npv, 2),
                "deficit_probability": mc.deficit_probability,
                "n_selected_assets": len(opt.selected_assets),
                "selected_asset_ids": "|".join(a.id for a in opt.selected_assets),
                "eligible_sectors": "|".join(r.scenario.eligible_sectors),
            })
    print(f"  [file] CSV roadmap saved  -> {path}")
    return path
