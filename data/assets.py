import os
from dataclasses import dataclass
from typing import List

import pandas as pd

PIPELINE_FILE = "investment_pipeline.xlsx"


@dataclass
class Asset:
    id: str
    name: str
    sector: str
    capital_required: float
    annual_cash_flows: List[float]
    discount_rate: float
    risk_sigma: float


def get_asset_pool() -> List[Asset]:
    """
    Load the investment pipeline from investment_pipeline.xlsx.

    To use your firm's real data:
      1. Open investment_pipeline.xlsx
      2. Replace rows with your actual investment projects
      3. Run: python main.py  (no code changes needed)

    Required columns:
      id, name, sector, capex,
      Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8,
      discount_rate, risk_sigma
    """
    if not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError(
            f"'{PIPELINE_FILE}' not found in the project root.\n"
            f"Run: python create_sample_data.py  to generate a sample file."
        )

    df = pd.read_excel(PIPELINE_FILE, dtype={"id": str})

    required_cols = {"id", "name", "sector", "capex",
                     "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8",
                     "discount_rate", "risk_sigma"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"'{PIPELINE_FILE}' is missing columns: {missing}")

    assets = []
    for _, row in df.iterrows():
        cash_flows = [float(row[f"Y{t}"]) for t in range(1, 9)]
        assets.append(Asset(
            id=str(row["id"]),
            name=str(row["name"]),
            sector=str(row["sector"]),
            capital_required=float(row["capex"]),
            annual_cash_flows=cash_flows,
            discount_rate=float(row["discount_rate"]),
            risk_sigma=float(row["risk_sigma"]),
        ))

    return assets
