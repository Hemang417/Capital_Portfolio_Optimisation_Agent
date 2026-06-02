"""
main.py — CLI entry point for the Capital Portfolio Optimisation Agent.

Invokes the LangGraph multi-agent workflow.  The human_approval_node
interrupt is handled interactively: the CLI prints the results summary,
prompts the user, then resumes (or terminates) the graph.
"""

import argparse
import sys
import uuid

from engine.optimizer import BUDGET_GBP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capital Portfolio Optimisation Agent — LangGraph Workflow",
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
        help="Random seed for Monte Carlo reproducibility (passed via budget offset)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Natural language query for agentic RAG mode. "
            "Requires GROQ_API_KEY environment variable. "
            'Example: --query "What if there is a tech crash with rising rates?"'
        ),
    )
    args = parser.parse_args()

    from langgraph.checkpoint.memory import MemorySaver
    from workflow.graph import build_graph

    graph  = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "user_intent":   args.query if args.query else "run_all",
        "budget_gbp":    args.budget,
        "mc_iterations": args.mc_iterations,
    }

    print("\n" + "=" * 72)
    print("  Capital Portfolio Optimisation Agent — LangGraph Workflow")
    print(f"  Budget: £{args.budget:,.0f}  |  MC iterations: {args.mc_iterations:,}")
    if args.query:
        print(f"  Mode: Agentic RAG  |  Query: \"{args.query}\"")
    else:
        print("  Mode: Run All 22 Scenarios")
    print("=" * 72 + "\n")

    _run_with_interrupt(graph, initial_state, config)


# ── Graph runner with interrupt handling ──────────────────────────────────────

def _run_with_interrupt(graph, initial_state: dict, config: dict) -> None:
    """
    Stream the graph to completion, pausing at human_approval_node to prompt
    the user, then resuming or terminating based on their answer.
    """
    from langgraph.types import Command

    # ── First pass: run until graph pauses at interrupt ───────────────────────
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        _print_node_update(event)

    # ── Check whether we are paused at an interrupt ───────────────────────────
    snapshot = graph.get_state(config)
    if not _has_interrupt(snapshot):
        # Graph ran to END without pausing (e.g. human_approval_node was bypassed)
        _print_report_paths(graph.get_state(config).values)
        return

    # ── Display interrupt payload ─────────────────────────────────────────────
    payload = _extract_interrupt_payload(snapshot)
    _print_approval_summary(payload)

    user_input = input("\n  Proceed to report generation? (yes/no): ").strip()
    approved   = user_input.lower() in ("yes", "y")

    if not approved:
        print("\n  Workflow terminated by user. No report generated.\n")
        return

    # ── Resume graph after approval ───────────────────────────────────────────
    print("\n  Resuming — generating reports…\n")
    for event in graph.stream(
        Command(resume=user_input), config=config, stream_mode="updates"
    ):
        _print_node_update(event)

    _print_report_paths(graph.get_state(config).values)


# ── Helper functions ──────────────────────────────────────────────────────────

def _has_interrupt(snapshot) -> bool:
    """Return True when the graph is paused at an interrupt checkpoint."""
    try:
        if not snapshot.next:
            return False
        for task in snapshot.tasks or []:
            if getattr(task, "interrupts", None):
                return True
    except Exception:
        pass
    return False


def _extract_interrupt_payload(snapshot) -> dict:
    """Pull the interrupt() payload dict from the graph snapshot."""
    try:
        for task in snapshot.tasks or []:
            interrupts = getattr(task, "interrupts", None)
            if interrupts:
                return interrupts[0].value
    except Exception:
        pass
    return {}


def _print_approval_summary(payload: dict) -> None:
    """Pretty-print the human approval checkpoint to the CLI."""
    print("\n" + "=" * 72)
    print("  HUMAN APPROVAL CHECKPOINT")
    print("=" * 72)

    res = payload.get("results", {})
    print(f"\n  Key Results:")
    print(f"    Strategic Value     : £{res.get('total_npv_m', 0):,.1f}M")
    print(f"    Return Multiplier   : {res.get('return_multiplier', 0):.2f}×")
    print(f"    Deficit Probability : {res.get('deficit_probability_pct', 0):.4f}%")

    flags = payload.get("risk_flags", [])
    if flags:
        print(f"\n  Risk Flags ({len(flags)}):")
        for f in flags:
            print(f"    [{f['severity']:6}]  {f['description'][:80]}")

    sug = payload.get("top_suggestion")
    if sug:
        print(f"\n  Top Suggestion [{sug.get('urgency', '—')}]:")
        print(f"    {sug.get('action', '')}")


def _print_node_update(event: dict) -> None:
    """Emit a brief line for each completed node."""
    for node_name in event:
        label = node_name.replace("_", " ").title()
        print(f"  ✓ {label}")


def _print_report_paths(state_values: dict) -> None:
    """Print the report file paths from final state."""
    paths = state_values.get("report_paths") or {}
    if not paths:
        return
    print("\n" + "=" * 72)
    print("  REPORTS GENERATED")
    print("=" * 72)
    for fmt, path in paths.items():
        print(f"  {fmt.upper():4}: {path}")
    print()


if __name__ == "__main__":
    sys.exit(main())
