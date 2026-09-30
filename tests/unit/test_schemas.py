from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from bank_assistant.domain.schemas import (
    Account,
    ApprovalDecision,
    Card,
    CardBlockProposal,
    Intent,
    MortgageInput,
)


def test_unknown_intent_is_rejected() -> None:
    with pytest.raises(ValueError):
        Intent("wire_transfer")


@pytest.mark.parametrize(
    "overrides",
    [
        {"loan_amount": Decimal("-1")},
        {"loan_amount": Decimal("0")},
        {"annual_rate_percent": Decimal("-0.5")},
        {"term_years": 0},
        {"term_years": 41},
    ],
)
def test_mortgage_input_rejects_out_of_domain_values(overrides: dict) -> None:
    data = {"loan_amount": 200_000, "annual_rate_percent": 3, "term_years": 30} | overrides
    with pytest.raises(ValidationError):
        MortgageInput(**data)


def test_invalid_approval_decision_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ApprovalDecision(action_id="act-1", approver_id="ops-1", decision="maybe")


@pytest.mark.parametrize("card_id", ["1234", "CARD-12", "card-0001"])
def test_malformed_card_id_is_rejected(card_id: str) -> None:
    with pytest.raises(ValidationError):
        Card(card_id=card_id, owner_id="CUST-0001", last4="1234", status="active")


def test_unexpected_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        Card(card_id="CARD-0001", owner_id="CUST-0001", last4="1234", status="active", cvv="123")


@pytest.mark.parametrize(
    "model",
    [
        MortgageInput(loan_amount=150_000, annual_rate_percent=Decimal("2.75"), term_years=25),
        Account(account_id="ACC-0009", owner_id="CUST-0009", currency="EUR", balance="10.50"),
        CardBlockProposal(
            action_id="act-1", card_id="CARD-0001", requested_by="CUST-0001", reason="Robo"
        ),
    ],
)
def test_contracts_round_trip_through_json(model: BaseModel) -> None:
    assert type(model).model_validate_json(model.model_dump_json()) == model
