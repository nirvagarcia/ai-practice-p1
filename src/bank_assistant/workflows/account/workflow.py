"""Account lookup: deterministic, scoped to the principal's own accounts. No LLM involved."""

import re

from opentelemetry.trace import Tracer

from bank_assistant.domain.errors import NotFoundError
from bank_assistant.domain.schemas import Account, Principal
from bank_assistant.guardrails.authorization import (
    AuthorizationPolicy,
    ToolAction,
    WorkflowName,
)
from bank_assistant.observability.metrics import RuntimeMetrics
from bank_assistant.observability.tracing import get_tracer, record_error, start_span
from bank_assistant.state.schemas import AssistantState, Outcome, latest_user_text, reply
from bank_assistant.tools.account_api import AccountGateway

NOT_FOUND = "No encuentro esa cuenta entre tus productos."

_ACCOUNT_ID = re.compile(r"\bACC-\d{4}\b")
_RECENT_TRANSACTIONS = 3


class AccountWorkflow:
    def __init__(
        self,
        accounts: AccountGateway,
        tracer: Tracer | None = None,
        metrics: RuntimeMetrics | None = None,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._accounts = accounts
        self._tracer = tracer or get_tracer()
        self._metrics = metrics
        self._authorization = authorization or AuthorizationPolicy()

    def __call__(self, state: AssistantState) -> dict:
        customer_id = state["customer_id"]
        requested = dict.fromkeys(_ACCOUNT_ID.findall(latest_user_text(state)))
        with start_span(self._tracer, "tool.get_account") as span:
            span.set_attribute("tool.name", "get_account")
            try:
                accounts = (
                    [
                        self._accounts.get_account_for_owner(account_id, customer_id)
                        for account_id in requested
                    ]
                    if requested
                    else list(self._accounts.accounts_of(customer_id))
                )
            except NotFoundError as error:
                record_error(span, error)
                if self._metrics is not None:
                    self._metrics.record_tool("get_account", "not_found")
                return reply(Outcome.ABSTAINED, NOT_FOUND)
            span.set_attribute("tool.result_count", len(accounts))
            if self._metrics is not None:
                self._metrics.record_tool("get_account", "success")
        # Foreign and missing accounts get the same answer so existence is not leaked.
        principal = Principal(customer_id=customer_id)
        if any(
            not self._authorization.authorize(
                principal,
                WorkflowName.ACCOUNT,
                ToolAction.GET_ACCOUNT,
                {"account_id": account.account_id},
                resource_owner_id=account.owner_id,
            ).allowed
            for account in accounts
        ):
            return reply(Outcome.ABSTAINED, NOT_FOUND)

        if not accounts:
            return reply(Outcome.ABSTAINED, "No tienes cuentas registradas.")
        return reply(Outcome.ANSWERED, "\n\n".join(_describe(account) for account in accounts))


def _describe(account: Account) -> str:
    recent = sorted(account.transactions, key=lambda t: t.booked_on, reverse=True)
    lines = [f"Cuenta {account.account_id}: saldo {account.balance} {account.currency}."]
    lines += [
        f"- {t.booked_on} {t.description}: {t.amount} {account.currency}"
        for t in recent[:_RECENT_TRANSACTIONS]
    ]
    return "\n".join(lines)
