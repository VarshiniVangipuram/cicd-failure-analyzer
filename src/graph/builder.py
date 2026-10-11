"""Assemble the LangGraph pipeline.

Linear graph for the MVP:

    parse_log -> retrieve_context -> analyze_failure -> END

The structure is intentionally a graph (not a function) so nodes can be
reordered, parallelized, or gated behind conditionals later — e.g. a
"search_similar_past_failures" node or a retry loop on low confidence.
"""
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    analyze_failure_node,
    parse_log_node,
    retrieve_context_node,
)
from src.graph.state import AnalysisState


def build_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("parse_log", parse_log_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("analyze_failure", analyze_failure_node)

    graph.add_edge(START, "parse_log")
    graph.add_edge("parse_log", "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_failure")
    graph.add_edge("analyze_failure", END)

    return graph.compile()
