import pytest
from langgraph.errors import GraphRecursionError

from bank_assistant.domain.schemas import Intent
from bank_assistant.graph.builder import GraphNodes, build_graph
from bank_assistant.state.schemas import Outcome, reply
from tests.fakes import user_state


def fixed_triage(intent: Intent | None, confidence: float = 0.9):
    return lambda _state: {"intent": intent, "confidence": confidence}


def echo(name: str):
    return lambda _state: reply(Outcome.ANSWERED, name)


def graph_for(triage):
    nodes = GraphNodes(
        triage=triage,
        faq=echo("faq"),
        mortgage=echo("mortgage"),
        account=echo("account"),
        card_block=echo("card_block"),
    )
    return build_graph(nodes, min_confidence=0.6)


@pytest.mark.parametrize(
    ("intent", "outcome", "response_contains"),
    [
        (Intent.FAQ, Outcome.ANSWERED, "faq"),
        (Intent.MORTGAGE, Outcome.ANSWERED, "mortgage"),
        (Intent.ACCOUNT, Outcome.ANSWERED, "account"),
        (Intent.CARD_BLOCK, Outcome.ANSWERED, "card_block"),
        (Intent.COMPLAINT, Outcome.HANDOFF, "agente"),
        (Intent.OUT_OF_SCOPE, Outcome.OUT_OF_SCOPE, "Solo puedo"),
        (None, Outcome.NEEDS_INFO, "No he entendido"),
    ],
)
def test_each_intent_terminates_in_its_workflow(
    intent: Intent | None, outcome: Outcome, response_contains: str
) -> None:
    state = graph_for(fixed_triage(intent)).invoke(user_state("mensaje"))

    assert state["outcome"] is outcome
    assert response_contains in state["response"]


def test_low_confidence_never_reaches_a_sensitive_workflow() -> None:
    state = graph_for(fixed_triage(Intent.CARD_BLOCK, 0.2)).invoke(user_state("tarjeta"))

    assert state["outcome"] is Outcome.NEEDS_INFO


def test_max_graph_steps_is_enforced() -> None:
    with pytest.raises(GraphRecursionError):
        graph_for(fixed_triage(Intent.FAQ)).invoke(user_state("x"), {"recursion_limit": 1})
