"""
Shared fixtures for the Capital Portfolio Optimisation Agent test suite.
"""
import os
import sys
import pytest

# Ensure project root is on the path so all modules resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Session-scoped real data fixtures (loaded once per test run) ──────────────

@pytest.fixture(scope="session")
def assets():
    from data.assets import get_asset_pool
    return get_asset_pool()


@pytest.fixture(scope="session")
def scenarios():
    from data.scenarios import get_all_scenarios
    return get_all_scenarios()


@pytest.fixture(scope="session")
def base_scenario(scenarios):
    return next(s for s in scenarios if s.id == 1)


# ── Lightweight synthetic assets (no Excel dependency) ────────────────────────

@pytest.fixture
def simple_assets():
    """Three minimal synthetic assets for fast unit tests."""
    from data.assets import Asset
    return [
        Asset("T1", "High-PI Tech",       "Technology",    5_000_000,
              [1_800_000, 2_200_000, 2_600_000, 2_800_000,
               3_000_000, 3_000_000, 2_800_000, 2_400_000], 0.08, 0.10),
        Asset("T2", "Mid-PI Renewables",  "Renewables",    3_000_000,
              [700_000,  900_000, 1_100_000, 1_200_000,
               1_300_000, 1_300_000, 1_200_000, 1_000_000], 0.07, 0.12),
        Asset("T3", "Low-PI Infra",       "Infrastructure",10_000_000,
              [1_400_000, 1_800_000, 2_200_000, 2_400_000,
               2_600_000, 2_600_000, 2_400_000, 2_000_000], 0.09, 0.11),
    ]
