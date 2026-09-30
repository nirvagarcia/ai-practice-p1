from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from bank_assistant.config.settings import Settings
from bank_assistant.observability.budget import BudgetExceededError, RequestBudgetCallback
from bank_assistant.observability.metrics import RuntimeMetrics, create_meter_provider


def test_budget_rejects_call_before_provider_start() -> None:
    settings = Settings(_env_file=None, max_tokens_per_request=10, max_total_tokens_per_request=12)
    callback = RequestBudgetCallback(settings)

    with pytest.raises(BudgetExceededError, match="token budget"):
        callback.on_chat_model_start({}, [[HumanMessage("x" * 20)]], run_id=uuid4())

    assert callback.usage_metadata == {}


def test_budget_reserves_each_call_and_collects_reported_usage() -> None:
    settings = Settings(_env_file=None, max_tokens_per_request=10, max_total_tokens_per_request=25)
    callback = RequestBudgetCallback(settings)
    run_id = uuid4()
    callback.on_chat_model_start({}, [[HumanMessage("short")]], run_id=run_id)

    with pytest.raises(BudgetExceededError):
        callback.on_chat_model_start({}, [[HumanMessage("second")]], run_id=uuid4())

    callback.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="ok",
                            response_metadata={"model_name": "test-model"},
                            usage_metadata={
                                "input_tokens": 6,
                                "output_tokens": 4,
                                "total_tokens": 10,
                            },
                        )
                    )
                ]
            ]
        ),
        run_id=run_id,
    )

    assert callback.input_tokens == 6
    assert callback.output_tokens == 4


def test_pricing_requires_both_rates_and_a_version() -> None:
    with pytest.raises(ValueError, match="both input and output"):
        Settings(_env_file=None, input_cost_usd_per_million_tokens=1)
    with pytest.raises(ValueError, match="PRICING_VERSION"):
        Settings(
            _env_file=None,
            input_cost_usd_per_million_tokens=1,
            output_cost_usd_per_million_tokens=2,
        )
    with pytest.raises(ValueError, match="Token prices"):
        Settings(_env_file=None, max_estimated_cost_usd=0.1)


def test_metrics_record_outcome_errors_tokens_and_duration() -> None:
    reader = InMemoryMetricReader()
    provider = create_meter_provider(reader)
    metrics = RuntimeMetrics(provider)
    metrics.record_request("failed", 12.5, 8, 3, 0.0001)
    provider.force_flush()

    scopes = reader.get_metrics_data().resource_metrics[0].scope_metrics
    points = {
        metric.name: metric.data.data_points[0] for scope in scopes for metric in scope.metrics
    }
    assert points["assistant.requests"].value == 1
    assert points["assistant.errors"].value == 1
    assert points["assistant.request.duration"].count == 1
    assert points["llm.input.tokens"].sum == 8
    assert points["llm.output.tokens"].sum == 3
    provider.shutdown()
