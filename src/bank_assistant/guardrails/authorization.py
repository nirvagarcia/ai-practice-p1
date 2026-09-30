"""Pure tool allowlist, argument, ownership and approval checks; never calls a model."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from bank_assistant.domain.schemas import ApprovalDecision, MortgageInput, Principal


class WorkflowName(StrEnum):
    FAQ = "faq"
    MORTGAGE = "mortgage"
    ACCOUNT = "account"
    CARD_BLOCK = "card_block"


class ToolAction(StrEnum):
    CALCULATE_PAYMENT = "calculate_payment"
    GET_ACCOUNT = "get_account"
    BLOCK_CARD = "block_card"


class AuthorizationCode(StrEnum):
    ALLOWED = "allowed"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_ARGUMENTS = "invalid_arguments"
    OWNER_REQUIRED = "owner_required"
    NOT_OWNER = "not_owner"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_MISMATCH = "approval_mismatch"
    APPROVER_NOT_ALLOWED = "approver_not_allowed"
    APPROVAL_REJECTED = "approval_rejected"


class AuthorizationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    code: AuthorizationCode


_ALLOWED = {
    WorkflowName.MORTGAGE: {ToolAction.CALCULATE_PAYMENT},
    WorkflowName.ACCOUNT: {ToolAction.GET_ACCOUNT},
    WorkflowName.CARD_BLOCK: {ToolAction.BLOCK_CARD},
    WorkflowName.FAQ: set(),
}
_CARD_ID = re.compile(r"^CARD-\d{4}$")
_ACCOUNT_ID = re.compile(r"^ACC-\d{4}$")


class AuthorizationPolicy:
    def __init__(self, approver_ids: frozenset[str] = frozenset()) -> None:
        self._approver_ids = approver_ids

    def authorize(
        self,
        principal: Principal,
        workflow: WorkflowName,
        action: ToolAction,
        arguments: dict[str, Any],
        resource_owner_id: str | None = None,
        action_id: str | None = None,
        approval: ApprovalDecision | None = None,
    ) -> AuthorizationDecision:
        if action not in _ALLOWED[workflow]:
            return _decision(AuthorizationCode.TOOL_NOT_ALLOWED)
        if not _valid_arguments(action, arguments):
            return _decision(AuthorizationCode.INVALID_ARGUMENTS)
        if action in {ToolAction.GET_ACCOUNT, ToolAction.BLOCK_CARD}:
            if resource_owner_id is None:
                return _decision(AuthorizationCode.OWNER_REQUIRED)
            if principal.customer_id != resource_owner_id:
                return _decision(AuthorizationCode.NOT_OWNER)
        if action is ToolAction.BLOCK_CARD:
            if approval is None:
                return _decision(AuthorizationCode.APPROVAL_REQUIRED)
            if action_id is None or approval.action_id != action_id:
                return _decision(AuthorizationCode.APPROVAL_MISMATCH)
            if approval.approver_id not in self._approver_ids:
                return _decision(AuthorizationCode.APPROVER_NOT_ALLOWED)
            if approval.decision != "approve":
                return _decision(AuthorizationCode.APPROVAL_REJECTED)
        return _decision(AuthorizationCode.ALLOWED)


def _valid_arguments(action: ToolAction, arguments: dict[str, Any]) -> bool:
    try:
        if action is ToolAction.CALCULATE_PAYMENT:
            MortgageInput.model_validate(arguments)
        elif action is ToolAction.GET_ACCOUNT:
            return bool(_ACCOUNT_ID.fullmatch(arguments.get("account_id", "")))
        elif action is ToolAction.BLOCK_CARD:
            return bool(_CARD_ID.fullmatch(arguments.get("card_id", "")))
    except (TypeError, ValueError):
        return False
    return True


def _decision(code: AuthorizationCode) -> AuthorizationDecision:
    return AuthorizationDecision(allowed=code is AuthorizationCode.ALLOWED, code=code)
