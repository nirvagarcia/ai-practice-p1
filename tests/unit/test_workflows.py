from pathlib import Path

import pytest

from bank_assistant.state.schemas import Outcome
from bank_assistant.tools.account_api import InMemoryAccountGateway
from bank_assistant.workflows.account.workflow import NOT_FOUND, AccountWorkflow
from bank_assistant.workflows.card_block.workflow import request_card_block
from bank_assistant.workflows.mortgage.workflow import (
    OUT_OF_RANGE,
    MortgageRequest,
    MortgageWorkflow,
)
from tests.fakes import FakeStructuredLLM, user_state


def mortgage(amount: float | None, rate: float | None, years: int | None) -> MortgageWorkflow:
    request = MortgageRequest(loan_amount=amount, annual_rate_percent=rate, term_years=years)
    return MortgageWorkflow(FakeStructuredLLM(request))


@pytest.fixture
def account_workflow(mocks_dir: Path) -> AccountWorkflow:
    return AccountWorkflow(InMemoryAccountGateway.from_fixture(mocks_dir / "accounts.json"))


def test_mortgage_payment_is_computed_in_python() -> None:
    update = mortgage(200_000, 5, 30)(user_state("hipoteca"))

    assert update["outcome"] is Outcome.ANSWERED
    assert "1073.64 €" in update["response"]


def test_mortgage_asks_for_missing_parameters() -> None:
    update = mortgage(90_000, None, None)(user_state("hipoteca de 90.000"))

    assert update["outcome"] is Outcome.NEEDS_INFO
    assert "tipo de interés" in update["response"] and "plazo" in update["response"]


def test_mortgage_rejects_out_of_range_values() -> None:
    update = mortgage(200_000, 5, 90)(user_state("hipoteca a 90 años"))

    assert update == {"outcome": Outcome.NEEDS_INFO, "response": OUT_OF_RANGE}


def test_account_lists_only_the_principals_accounts(account_workflow: AccountWorkflow) -> None:
    update = account_workflow(user_state("¿Cuál es mi saldo?"))

    assert update["outcome"] is Outcome.ANSWERED
    assert "ACC-0001" in update["response"] and "2450.75 EUR" in update["response"]
    assert "ACC-0002" not in update["response"]


@pytest.mark.parametrize("message", ["Saldo de ACC-0002", "Saldo de ACC-9999"])
def test_foreign_and_missing_accounts_are_indistinguishable(
    account_workflow: AccountWorkflow, message: str
) -> None:
    assert account_workflow(user_state(message)) == {
        "outcome": Outcome.ABSTAINED,
        "response": NOT_FOUND,
    }


def test_card_block_request_hands_off_without_side_effects() -> None:
    assert request_card_block(user_state("Bloquea CARD-0001"))["outcome"] is Outcome.HANDOFF
