import pytest

from bank_assistant.domain.schemas import ApprovalDecision, Principal
from bank_assistant.guardrails.authorization import (
    AuthorizationCode,
    AuthorizationPolicy,
    ToolAction,
    WorkflowName,
)

CUSTOMER = Principal(customer_id="CUST-0001")
OWNED_ACCOUNT = {"account_id": "ACC-0001"}
OWNED_CARD = {"card_id": "CARD-0001"}


def test_mortgage_calculation_is_allowlisted_only_in_its_workflow() -> None:
    policy = AuthorizationPolicy()
    args = {"loan_amount": 100_000, "annual_rate_percent": 3, "term_years": 20}

    assert policy.authorize(
        CUSTOMER, WorkflowName.MORTGAGE, ToolAction.CALCULATE_PAYMENT, args
    ).allowed
    denied = policy.authorize(CUSTOMER, WorkflowName.FAQ, ToolAction.CALCULATE_PAYMENT, args)
    assert denied.code is AuthorizationCode.TOOL_NOT_ALLOWED


def test_invalid_arguments_are_denied_even_for_allowlisted_tool() -> None:
    decision = AuthorizationPolicy().authorize(
        CUSTOMER,
        WorkflowName.MORTGAGE,
        ToolAction.CALCULATE_PAYMENT,
        {"loan_amount": -1, "annual_rate_percent": 3, "term_years": 20},
    )

    assert decision.code is AuthorizationCode.INVALID_ARGUMENTS


def test_account_access_requires_ownership() -> None:
    policy = AuthorizationPolicy()

    assert policy.authorize(
        CUSTOMER,
        WorkflowName.ACCOUNT,
        ToolAction.GET_ACCOUNT,
        OWNED_ACCOUNT,
        resource_owner_id="CUST-0001",
    ).allowed
    assert (
        policy.authorize(
            CUSTOMER,
            WorkflowName.ACCOUNT,
            ToolAction.GET_ACCOUNT,
            OWNED_ACCOUNT,
            resource_owner_id="CUST-0002",
        ).code
        is AuthorizationCode.NOT_OWNER
    )
    assert (
        policy.authorize(CUSTOMER, WorkflowName.ACCOUNT, ToolAction.GET_ACCOUNT, OWNED_ACCOUNT).code
        is AuthorizationCode.OWNER_REQUIRED
    )


@pytest.mark.parametrize(
    ("approval", "action_id", "approvers", "expected"),
    [
        (None, "action-1", frozenset({"OPS-1"}), AuthorizationCode.APPROVAL_REQUIRED),
        (
            ApprovalDecision(action_id="action-2", approver_id="OPS-1", decision="approve"),
            "action-1",
            frozenset({"OPS-1"}),
            AuthorizationCode.APPROVAL_MISMATCH,
        ),
        (
            ApprovalDecision(action_id="action-1", approver_id="OPS-2", decision="approve"),
            "action-1",
            frozenset({"OPS-1"}),
            AuthorizationCode.APPROVER_NOT_ALLOWED,
        ),
        (
            ApprovalDecision(action_id="action-1", approver_id="OPS-1", decision="reject"),
            "action-1",
            frozenset({"OPS-1"}),
            AuthorizationCode.APPROVAL_REJECTED,
        ),
    ],
)
def test_card_block_requires_matching_approval_from_allowlisted_approver(
    approval, action_id: str, approvers: frozenset[str], expected: AuthorizationCode
) -> None:
    decision = AuthorizationPolicy(approvers).authorize(
        CUSTOMER,
        WorkflowName.CARD_BLOCK,
        ToolAction.BLOCK_CARD,
        OWNED_CARD,
        resource_owner_id="CUST-0001",
        action_id=action_id,
        approval=approval,
    )

    assert decision.code is expected


def test_approved_card_block_is_allowed_only_for_owner() -> None:
    policy = AuthorizationPolicy(frozenset({"OPS-1"}))
    approval = ApprovalDecision(action_id="action-1", approver_id="OPS-1", decision="approve")

    assert policy.authorize(
        CUSTOMER,
        WorkflowName.CARD_BLOCK,
        ToolAction.BLOCK_CARD,
        OWNED_CARD,
        resource_owner_id="CUST-0001",
        action_id="action-1",
        approval=approval,
    ).allowed
    assert (
        policy.authorize(
            CUSTOMER,
            WorkflowName.CARD_BLOCK,
            ToolAction.BLOCK_CARD,
            OWNED_CARD,
            resource_owner_id="CUST-0002",
            action_id="action-1",
            approval=approval,
        ).code
        is AuthorizationCode.NOT_OWNER
    )


def test_card_action_is_not_allowlisted_in_faq() -> None:
    decision = AuthorizationPolicy().authorize(
        CUSTOMER, WorkflowName.FAQ, ToolAction.BLOCK_CARD, OWNED_CARD
    )

    assert decision.code is AuthorizationCode.TOOL_NOT_ALLOWED
