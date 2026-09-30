"""Serializable LangGraph execution state. Holds data only: no clients, locks or providers."""

from enum import StrEnum
from typing import Any

from langgraph.graph import MessagesState

from bank_assistant.domain.schemas import Intent


class Outcome(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    NEEDS_INFO = "needs_info"
    HANDOFF = "handoff"
    OUT_OF_SCOPE = "out_of_scope"
    FAILED = "failed"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class AssistantState(MessagesState, total=False):
    request_id: str
    customer_id: str
    intent: Intent | None
    confidence: float
    outcome: Outcome
    response: str
    source_ids: list[str]


def latest_user_text(state: AssistantState) -> str:
    return str(state["messages"][-1].text)


def reply(outcome: Outcome, text: str, **extra: Any) -> dict[str, Any]:
    return {"outcome": outcome, "response": text, **extra}
