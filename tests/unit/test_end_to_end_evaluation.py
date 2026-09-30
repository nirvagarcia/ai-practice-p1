from bank_assistant.application.service import AssistantReply
from bank_assistant.config.settings import Settings
from bank_assistant.domain.schemas import Intent
from bank_assistant.state.schemas import Outcome
from evaluation.common.datasets import EndToEndCase
from evaluation.common.runner import Measured
from evaluation.end_to_end.run import case_record, summarize


def test_end_to_end_report_tracks_outcome_latency_tokens_and_unknown_cost() -> None:
    case = EndToEndCase(
        case_id="e2e-test",
        dataset_version="test-v1",
        split="dev",
        message="Synthetic message",
        customer_id="CUST-0001",
        expected_intent=Intent.FAQ,
        expected_outcome=Outcome.ANSWERED,
    )
    result = AssistantReply(
        request_id="request-1",
        intent=Intent.FAQ,
        outcome=Outcome.ANSWERED,
        text="synthetic answer",
    )
    record = case_record(case, Measured(result, None, 20.0, None, None), Settings(_env_file=None))

    assert record["outcome_correct"] and record["completed"]
    assert record["estimated_cost_usd"] is None
    summary = summarize([record])
    assert summary["outcome_accuracy"] == 1.0
    assert summary["tokens_per_request"] == {"input": None, "output": None}


def test_end_to_end_report_handles_unknown_intent_after_failure() -> None:
    case = EndToEndCase(
        case_id="e2e-fail",
        dataset_version="test-v1",
        split="dev",
        message="Synthetic message",
        customer_id="CUST-0001",
        expected_intent=Intent.FAQ,
        expected_outcome=Outcome.ANSWERED,
    )
    record = case_record(
        case, Measured(None, "RuntimeError", 5.0, None, None), Settings(_env_file=None)
    )

    assert summarize([record])["intent"]["accuracy"] == 0.0
