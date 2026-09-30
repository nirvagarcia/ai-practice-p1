from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from bank_assistant.domain.errors import InvalidStateError, NotFoundError
from bank_assistant.domain.schemas import CardStatus, MortgageInput
from bank_assistant.tools.account_api import InMemoryAccountGateway
from bank_assistant.tools.audit import InMemoryAuditLog
from bank_assistant.tools.card_api import InMemoryCardGateway
from bank_assistant.tools.mortgage_calculator import calculate_payment

FIXED_NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
def cards(mocks_dir: Path) -> InMemoryCardGateway:
    return InMemoryCardGateway.from_fixture(mocks_dir / "cards.json", clock=lambda: FIXED_NOW)


@pytest.fixture
def accounts(mocks_dir: Path) -> InMemoryAccountGateway:
    return InMemoryAccountGateway.from_fixture(mocks_dir / "accounts.json")


@pytest.mark.parametrize(
    ("amount", "rate", "years", "expected"),
    [
        (200_000, "5", 30, Decimal("1073.64")),
        (100_000, "0", 10, Decimal("833.33")),
    ],
)
def test_mortgage_payment_matches_reference(amount, rate, years, expected) -> None:
    quote = calculate_payment(
        MortgageInput(loan_amount=amount, annual_rate_percent=Decimal(rate), term_years=years)
    )

    assert quote.monthly_payment == expected
    assert quote.total_interest == quote.total_paid - quote.input.loan_amount


def test_account_balance_comes_from_fixture(accounts: InMemoryAccountGateway) -> None:
    account = accounts.get_account("ACC-0001")

    assert account.balance == Decimal("2450.75")
    assert len(account.transactions) == 3


def test_unknown_account_is_an_error_not_invented_data(accounts: InMemoryAccountGateway) -> None:
    with pytest.raises(NotFoundError):
        accounts.get_account("ACC-9999")


def test_account_adapter_scopes_lookup_to_owner(accounts: InMemoryAccountGateway) -> None:
    with pytest.raises(NotFoundError):
        accounts.get_account_for_owner("ACC-0002", "CUST-0001")


def test_unknown_card_is_denied(cards: InMemoryCardGateway) -> None:
    with pytest.raises(NotFoundError):
        cards.block_card("CARD-9999", "key-1")


def test_block_card_is_idempotent(cards: InMemoryCardGateway) -> None:
    first = cards.block_card("CARD-0001", "key-1")
    replay = cards.block_card("CARD-0001", "key-1")

    assert first == replay
    assert first.processed_at == FIXED_NOW
    assert cards.get_card("CARD-0001").status is CardStatus.BLOCKED


def test_idempotency_key_cannot_be_reused_for_another_card(cards: InMemoryCardGateway) -> None:
    cards.block_card("CARD-0001", "key-1")

    with pytest.raises(InvalidStateError):
        cards.block_card("CARD-0003", "key-1")


def test_already_blocked_card_with_new_key_is_rejected(cards: InMemoryCardGateway) -> None:
    with pytest.raises(InvalidStateError):
        cards.block_card("CARD-0002", "key-2")


def test_audit_log_records_minimal_event() -> None:
    log = InMemoryAuditLog(clock=lambda: FIXED_NOW)

    event = log.record(
        actor_id="CUST-0001", action="block_card", resource_id="CARD-0001", outcome="executed"
    )

    assert log.events == (event,)
    assert event.occurred_at == FIXED_NOW
