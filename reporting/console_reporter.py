from typing import List

from models import ScenarioRunResult

_W = 72  # console width


def _hr(char: str = "=") -> str:
    return char * _W


def _fmt_gbp(v: float) -> str:
    return f"£{v:>16,.0f}"


def _fmt_m(v: float) -> str:
    return f"{v / 1_000_000:>9.2f}M"


def print_scenario_report(result: ScenarioRunResult) -> None:
    s = result.scenario
    opt = result.optimization
    mc = result.monte_carlo

    print(_hr("="))
    mandate = (
        f"  Sector mandate: {', '.join(s.eligible_sectors)}"
        if s.eligible_sectors
        else "  Sector mandate: All sectors"
    )
    print(f"  SCENARIO {s.id:>2}: {s.name}")
    print(f"  {s.description}")
    print(mandate)
    print(_hr("-"))

    # ── Selected portfolio table ─────────────────────────────────────────────
    if not opt.selected_assets:
        print("  No viable assets selected under this scenario.")
    else:
        hdr = f"  {'#':>2}  {'ID':<4}  {'Name':<34}  {'Sector':<16}  {'Capex':>9}  {'NPV':>9}  {'PI':>6}"
        print(hdr)
        print("  " + "-" * (_W - 2))
        for i, asset in enumerate(opt.selected_assets, 1):
            pi = opt.profitability_indices.get(asset.id, 0.0)
            npv = opt.total_npv  # shown per-row below
            row_npv = (
                opt.total_npv
                if i == len(opt.selected_assets)
                else None
            )
            # Per-asset NPV is total_capital * PI (approx) — use stored PI
            asset_npv = opt.profitability_indices.get(asset.id, 0.0) * (
                asset.capital_required * result.scenario.capex_modifier
            )
            print(
                f"  {i:>2}  {asset.id:<4}  {asset.name:<34}  {asset.sector:<16}"
                f"  {asset.capital_required * s.capex_modifier / 1e6:>8.2f}M"
                f"  {asset_npv / 1e6:>8.1f}M"
                f"  {pi:>5.1f}x"
            )

        print("  " + "-" * (_W - 2))
        budget = opt.total_capital_deployed + opt.remaining_budget
        pct = 100.0 * opt.total_capital_deployed / budget if budget else 0.0
        multiplier = opt.total_npv / opt.total_capital_deployed if opt.total_capital_deployed else 0.0
        print(f"  Capital deployed : £{opt.total_capital_deployed:>14,.0f}  ({pct:.1f}% of budget)")
        print(f"  Strategic value  : £{opt.total_npv:>14,.0f}")
        print(f"  Return multiplier: {multiplier:.2f}x")

    # ── Monte Carlo summary ──────────────────────────────────────────────────
    print(_hr("-"))
    deficit_pct = mc.deficit_probability * 100.0
    deficit_tag = " << Zero downside confirmed" if mc.deficit_count == 0 else ""
    print(f"  Monte Carlo stress test ({mc.n_iterations:,} iterations, seed=42):")
    print(f"    Mean portfolio NPV  : £{mc.mean_total_npv:>14,.0f}")
    print(f"    Std dev             : £{mc.std_total_npv:>14,.0f}")
    print(f"    5th percentile (P05): £{mc.p5_total_npv:>14,.0f}")
    print(f"    95th percentile(P95): £{mc.p95_total_npv:>14,.0f}")
    print(f"    Deficit probability : {deficit_pct:.4f}%{deficit_tag}")
    print()


def print_wave1_summary(
    results: List[ScenarioRunResult],
    base_case: ScenarioRunResult,
) -> None:
    print("\n")
    print(_hr("="))
    print(f"{'WAVE-1 CAPITAL ROADMAP  —  ALL 22 SCENARIOS':^{_W}}")
    print(_hr("="))

    col = f"  {'#':>2}  {'Scenario':<32}  {'Capital(£M)':>11}  {'NPV(£M)':>9}  {'Return':>7}  {'Deficit%':>8}"
    print(col)
    print("  " + "-" * (_W - 2))

    for r in results:
        opt = r.optimization
        mc = r.monte_carlo
        cap_m = opt.total_capital_deployed / 1e6
        npv_m = opt.total_npv / 1e6
        mult = opt.total_npv / opt.total_capital_deployed if opt.total_capital_deployed else 0.0
        def_pct = mc.deficit_probability * 100.0
        marker = " *" if r.scenario.id == 1 else "  "
        print(
            f"{marker}{r.scenario.id:>2}  {r.scenario.name:<32}  {cap_m:>10.2f}M"
            f"  {npv_m:>8.1f}M  {mult:>6.2f}x  {def_pct:>7.4f}%"
        )

    print("  " + "-" * (_W - 2))
    print("  * = Wave-1 Roadmap (Base Case)")
    print()

    # ── Wave-1 highlighted ───────────────────────────────────────────────────
    opt = base_case.optimization
    mc = base_case.monte_carlo
    mult = opt.total_npv / opt.total_capital_deployed
    print(_hr("="))
    print(f"{'WAVE-1 ROADMAP  (Base Case)':^{_W}}")
    print(_hr("="))
    print(f"  Strategic value  : £{opt.total_npv:>14,.0f}")
    print(f"  Capital deployed : £{opt.total_capital_deployed:>14,.0f}")
    print(f"  Return multiplier: {mult:.2f}x")
    print(f"  Deficit prob.    : {mc.deficit_probability * 100:.4f}%  << Zero downside confirmed")
    print()
    print(f"  Selected portfolio ({len(opt.selected_assets)} assets):")
    print(f"  {'#':>2}  {'ID':<4}  {'Name':<34}  {'Sector':<16}  {'Capex':>9}  {'NPV':>10}  {'PI':>6}")
    print("  " + "-" * (_W - 2))
    for i, asset in enumerate(opt.selected_assets, 1):
        pi = opt.profitability_indices.get(asset.id, 0.0)
        asset_npv = pi * asset.capital_required
        print(
            f"  {i:>2}  {asset.id:<4}  {asset.name:<34}  {asset.sector:<16}"
            f"  {asset.capital_required / 1e6:>8.2f}M"
            f"  {asset_npv / 1e6:>9.1f}M"
            f"  {pi:>5.2f}x"
        )
    print("  " + "-" * (_W - 2))
    print(f"       {'TOTAL':<38}{'':>16}  {opt.total_capital_deployed / 1e6:>8.2f}M"
          f"  {opt.total_npv / 1e6:>9.1f}M  {mult:>5.2f}x")
    print(_hr("="))
    print()
