import pytest
from langchain_core.exceptions import OutputParserException

from bank_assistant.domain.schemas import Intent
from bank_assistant.graph.routing import Route, select_route
from bank_assistant.workflows.triage import LlmTriage, TriageDecision
from tests.fakes import FakeStructuredLLM, user_state


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        (Intent.FAQ, Route.FAQ),
        (Intent.MORTGAGE, Route.MORTGAGE),
        (Intent.ACCOUNT, Route.ACCOUNT),
        (Intent.CARD_BLOCK, Route.CARD_BLOCK),
        (Intent.COMPLAINT, Route.HANDOFF),
        (Intent.OUT_OF_SCOPE, Route.OUT_OF_SCOPE),
    ],
)
def test_every_intent_maps_to_an_allowlisted_route(intent: Intent, route: Route) -> None:
    assert select_route({"intent": intent, "confidence": 0.9}, min_confidence=0.6) is route


@pytest.mark.parametrize(
    "state",
    [
        {"intent": Intent.CARD_BLOCK, "confidence": 0.3},
        {"intent": None, "confidence": 0.0},
        {},
    ],
)
def test_low_confidence_or_missing_intent_asks_for_clarification(state: dict) -> None:
    assert select_route(state, min_confidence=0.6) is Route.CLARIFY


def test_triage_node_returns_structured_decision() -> None:
    llm = FakeStructuredLLM(TriageDecision(intent=Intent.ACCOUNT, confidence=0.8))

    assert LlmTriage(llm)(user_state("¿Mi saldo?")) == {
        "intent": Intent.ACCOUNT,
        "confidence": 0.8,
    }


def test_unparseable_triage_output_fails_closed() -> None:
    def broken(_text: str) -> TriageDecision:
        raise OutputParserException("not json")

    assert LlmTriage(FakeStructuredLLM(broken))(user_state("???")) == {
        "intent": None,
        "confidence": 0.0,
    }
