"""
workflow/graph.py — LangGraph StateGraph definition.

Topology
--------
    START
      → data_validation_node
      → scenario_retrieval_node
      → financial_engine_node
      → commentary_node  ┐  (parallel super-step)
      → risk_flag_node   ┘
      → human_approval_node
      → report_generation_node   (only when human_approved == True)
      → END

The graph requires a checkpointer to support interrupt() in
human_approval_node.  A MemorySaver is used by default.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from workflow.nodes import (
    commentary_node,
    data_validation_node,
    financial_engine_node,
    human_approval_node,
    report_generation_node,
    risk_flag_node,
    scenario_retrieval_node,
)
from workflow.state import WorkflowState


def _route_after_approval(state: WorkflowState) -> str:
    """
    Conditional edge from human_approval_node.
    Returns the next node name: report generation if approved, else END.
    """
    return "report_generation_node" if state.get("human_approved") else END


def build_graph(checkpointer=None) -> "CompiledGraph":
    """
    Build and compile the portfolio optimisation StateGraph.

    Args:
        checkpointer: LangGraph checkpointer instance.  Required for
                      interrupt() support.  Defaults to MemorySaver().

    Returns:
        A compiled CompiledGraph ready for .stream() or .invoke().
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("data_validation_node",    data_validation_node)
    builder.add_node("scenario_retrieval_node", scenario_retrieval_node)
    builder.add_node("financial_engine_node",   financial_engine_node)
    builder.add_node("commentary_node",         commentary_node)
    builder.add_node("risk_flag_node",          risk_flag_node)
    builder.add_node("human_approval_node",     human_approval_node)
    builder.add_node("report_generation_node",  report_generation_node)

    # ── Sequential edges ──────────────────────────────────────────────────────
    builder.add_edge(START,                      "data_validation_node")
    builder.add_edge("data_validation_node",     "scenario_retrieval_node")
    builder.add_edge("scenario_retrieval_node",  "financial_engine_node")

    # ── Parallel fan-out: financial_engine_node → {commentary, risk_flag} ─────
    builder.add_edge("financial_engine_node",    "commentary_node")
    builder.add_edge("financial_engine_node",    "risk_flag_node")

    # ── Parallel fan-in: {commentary, risk_flag} → human_approval_node ────────
    builder.add_edge("commentary_node",          "human_approval_node")
    builder.add_edge("risk_flag_node",           "human_approval_node")

    # ── Conditional edge from human_approval_node ─────────────────────────────
    builder.add_conditional_edges(
        "human_approval_node",
        _route_after_approval,
        {
            "report_generation_node": "report_generation_node",
            END: END,
        },
    )

    builder.add_edge("report_generation_node",   END)

    return builder.compile(checkpointer=checkpointer)


# ── Module-level default instance ─────────────────────────────────────────────
# Import this in app.py for the Streamlit tab.  main.py creates its own
# instance with a fresh MemorySaver per invocation.
portfolio_graph = build_graph()
