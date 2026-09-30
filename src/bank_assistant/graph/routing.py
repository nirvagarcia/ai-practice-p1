"""Pure routing: maps a validated triage result to an allowlisted route. No model or tool calls."""

from enum import StrEnum

from bank_assistant.domain.schemas import Intent
from bank_assistant.state.schemas import AssistantState


class Route(StrEnum):
    FAQ = "faq"
    MORTGAGE = "mortgage"
    ACCOUNT = "account"
    CARD_BLOCK = "card_block"
    HANDOFF = "handoff"
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFY = "clarify"


_ROUTE_BY_INTENT = {
    Intent.FAQ: Route.FAQ,
    Intent.MORTGAGE: Route.MORTGAGE,
    Intent.ACCOUNT: Route.ACCOUNT,
    Intent.CARD_BLOCK: Route.CARD_BLOCK,
    Intent.COMPLAINT: Route.HANDOFF,
    Intent.OUT_OF_SCOPE: Route.OUT_OF_SCOPE,
}


def select_route(state: AssistantState, min_confidence: float) -> Route:
    intent = state.get("intent")
    if intent is None or state.get("confidence", 0.0) < min_confidence:
        return Route.CLARIFY
    return _ROUTE_BY_INTENT[intent]
