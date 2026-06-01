import argparse
import sys

from agent.portfolio_agent import PortfolioAgent
from engine.optimizer import BUDGET_GBP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capital Portfolio Optimisation Agent — MSc Financial Data Analytics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=BUDGET_GBP,
        metavar="GBP",
        help="Capital budget in GBP",
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=10_000,
        metavar="N",
        help="Monte Carlo iteration count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Monte Carlo reproducibility",
    )
    args = parser.parse_args()

    agent = PortfolioAgent(
        budget=args.budget,
        mc_iterations=args.mc_iterations,
        random_seed=args.seed,
    )
    agent.run()


if __name__ == "__main__":
    sys.exit(main())
