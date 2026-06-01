from dataclasses import dataclass

from data.scenarios import Scenario
from engine.optimizer import OptimizationResult
from engine.monte_carlo import MonteCarloResult


@dataclass
class ScenarioRunResult:
    scenario: Scenario
    optimization: OptimizationResult
    monte_carlo: MonteCarloResult
