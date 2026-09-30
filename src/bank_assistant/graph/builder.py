"""Root workflow: START → triage → one allowlisted route → END."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.trace import Tracer

from bank_assistant.graph.routing import Route, select_route
from bank_assistant.observability.tracing import trace_node
from bank_assistant.state.schemas import AssistantState
from bank_assistant.workflows.responses import clarify, handoff, out_of_scope

Node = Callable[[AssistantState], dict[str, Any]]

TRIAGE = "triage"


@dataclass(frozen=True)
class GraphNodes:
    triage: Node
    faq: Node
    mortgage: Node
    account: Node
    card_block: Node


def build_graph(
    nodes: GraphNodes, min_confidence: float, tracer: Tracer | None = None
) -> CompiledStateGraph:
    handlers: dict[Route, Node] = {
        Route.FAQ: nodes.faq,
        Route.MORTGAGE: nodes.mortgage,
        Route.ACCOUNT: nodes.account,
        Route.CARD_BLOCK: nodes.card_block,
        Route.HANDOFF: handoff,
        Route.OUT_OF_SCOPE: out_of_scope,
        Route.CLARIFY: clarify,
    }
    graph = StateGraph(AssistantState)

    def instrument(name: str, node: Node) -> Node:
        return trace_node(name, node, tracer) if tracer else node

    graph.add_node(TRIAGE, instrument(TRIAGE, nodes.triage))
    graph.add_edge(START, TRIAGE)
    graph.add_conditional_edges(
        TRIAGE, partial(select_route, min_confidence=min_confidence), list(handlers)
    )
    for route, handler in handlers.items():
        graph.add_node(route, instrument(route.value, handler))
        graph.add_edge(route, END)
    return graph.compile()
