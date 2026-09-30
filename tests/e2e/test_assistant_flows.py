"""Deterministic end-to-end flows through the application service, with offline fakes."""

from pathlib import Path

import pytest

from bank_assistant.application.service import FAILURE_TEXT, AssistantService
from bank_assistant.domain.schemas import Intent, Principal
from bank_assistant.graph.builder import GraphNodes, build_graph
from bank_assistant.state.schemas import AssistantState, Outcome, latest_user_text, reply
from bank_assistant.tools.account_api import InMemoryAccountGateway
from bank_assistant.workflows.account.workflow import AccountWorkflow
from bank_assistant.workflows.card_block.workflow import request_card_block
from bank_assistant.workflows.mortgage.workflow import MortgageRequest, MortgageWorkflow
from tests.fakes import FakeStructuredLLM

PRINCIPAL = Principal(customer_id="CUST-0001")
KEYWORDS = (
    ("hipoteca", Intent.MORTGAGE),
    ("saldo", Intent.ACCOUNT),
    ("tarjeta", Intent.CARD_BLOCK),
    ("reclamación", Intent.COMPLAINT),
    ("horario", Intent.FAQ),
)


def keyword_triage(state: AssistantState) -> dict:
    text = latest_user_text(state).lower()
    intent = next((i for word, i in KEYWORDS if word in text), Intent.OUT_OF_SCOPE)
    return {"intent": intent, "confidence": 0.9}


def grounded_faq(_state: AssistantState) -> dict:
    return reply(Outcome.ANSWERED, "De 8:30 a 14:30.", source_ids=["horarios#0"])


def service(mocks_dir: Path, triage=keyword_triage, max_steps: int = 12) -> AssistantService:
    request = MortgageRequest(loan_amount=200_000, annual_rate_percent=5, term_years=30)
    nodes = GraphNodes(
        triage=triage,
        faq=grounded_faq,
        mortgage=MortgageWorkflow(FakeStructuredLLM(request)),
        account=AccountWorkflow(InMemoryAccountGateway.from_fixture(mocks_dir / "accounts.json")),
        card_block=request_card_block,
    )
    return AssistantService(build_graph(nodes, min_confidence=0.6), max_steps)


@pytest.mark.parametrize(
    ("message", "intent", "outcome", "expected_text"),
    [
        ("¿Qué horario tenéis?", Intent.FAQ, Outcome.ANSWERED, "8:30"),
        ("Simula una hipoteca", Intent.MORTGAGE, Outcome.ANSWERED, "1073.64 €"),
        ("¿Cuál es mi saldo?", Intent.ACCOUNT, Outcome.ANSWERED, "2450.75 EUR"),
        ("Bloquea mi tarjeta", Intent.CARD_BLOCK, Outcome.HANDOFF, "ninguna acción"),
        ("Quiero poner una reclamación", Intent.COMPLAINT, Outcome.HANDOFF, "agente"),
        ("Escríbeme un poema", Intent.OUT_OF_SCOPE, Outcome.OUT_OF_SCOPE, "Solo puedo"),
    ],
)
def test_flows(mocks_dir: Path, message, intent, outcome, expected_text) -> None:
    result = service(mocks_dir).ask(message, PRINCIPAL)

    assert (result.intent, result.outcome) == (intent, outcome)
    assert expected_text in result.text
    assert len(result.request_id) == 32


def test_faq_sources_are_exposed(mocks_dir: Path) -> None:
    assert service(mocks_dir).ask("horario", PRINCIPAL).source_ids == ("horarios#0",)


def test_internal_errors_are_not_leaked(mocks_dir: Path) -> None:
    def exploding_triage(_state: AssistantState) -> dict:
        raise RuntimeError("secret internal detail")

    result = service(mocks_dir, triage=exploding_triage).ask("hola", PRINCIPAL)

    assert result.outcome is Outcome.FAILED
    assert result.text == FAILURE_TEXT


def test_step_budget_exhaustion_fails_safely(mocks_dir: Path) -> None:
    result = service(mocks_dir, max_steps=1).ask("¿Cuál es mi saldo?", PRINCIPAL)

    assert result.outcome is Outcome.FAILED
