"""Business contracts. Framework-free: no LangChain, LangGraph, OpenAI or storage imports."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CustomerId = Annotated[str, StringConstraints(pattern=r"^CUST-\d{4}$")]
AccountId = Annotated[str, StringConstraints(pattern=r"^ACC-\d{4}$")]
CardId = Annotated[str, StringConstraints(pattern=r"^CARD-\d{4}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Money = Annotated[Decimal, Field(max_digits=14, decimal_places=2)]


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Intent(StrEnum):
    FAQ = "faq"
    MORTGAGE = "mortgage"
    ACCOUNT = "account"
    CARD_BLOCK = "card_block"
    COMPLAINT = "complaint"
    OUT_OF_SCOPE = "out_of_scope"


class CardStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"


class ActionStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class Principal(DomainModel):
    customer_id: CustomerId


class MortgageInput(DomainModel):
    loan_amount: Annotated[Money, Field(gt=0, le=10_000_000)]
    annual_rate_percent: Annotated[Decimal, Field(ge=0, le=30)]
    term_years: Annotated[int, Field(ge=1, le=40)]


class MortgageQuote(DomainModel):
    input: MortgageInput
    monthly_payment: Money
    total_paid: Money
    total_interest: Money


class Transaction(DomainModel):
    transaction_id: str
    booked_on: date
    description: str
    amount: Money


class Account(DomainModel):
    account_id: AccountId
    owner_id: CustomerId
    currency: CurrencyCode
    balance: Money
    transactions: tuple[Transaction, ...] = ()


class Card(DomainModel):
    card_id: CardId
    owner_id: CustomerId
    last4: Annotated[str, StringConstraints(pattern=r"^\d{4}$")]
    status: CardStatus


class CardBlockResult(DomainModel):
    card_id: CardId
    status: CardStatus
    idempotency_key: str
    processed_at: datetime


class CardBlockProposal(DomainModel):
    action_id: str
    card_id: CardId
    requested_by: CustomerId
    reason: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    status: ActionStatus = ActionStatus.PENDING_APPROVAL


class ApprovalDecision(DomainModel):
    action_id: str
    approver_id: str
    decision: Literal["approve", "reject"]
