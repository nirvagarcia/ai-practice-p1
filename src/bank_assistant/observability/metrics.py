"""Low-cardinality OpenTelemetry runtime metrics with a local console exporter."""

import atexit
from threading import Lock

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

_LOCK = Lock()
_PROVIDER: MeterProvider | None = None


def create_meter_provider(reader: MetricReader | None = None) -> MeterProvider:
    reader = reader or PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=60_000
    )
    return MeterProvider(
        resource=Resource.create({"service.name": "bank-assistant-lab"}),
        metric_readers=[reader],
    )


def configure_metrics() -> "RuntimeMetrics":
    global _PROVIDER
    with _LOCK:
        if _PROVIDER is None:
            from opentelemetry import metrics

            _PROVIDER = create_meter_provider()
            metrics.set_meter_provider(_PROVIDER)
            atexit.register(_PROVIDER.shutdown)
    return RuntimeMetrics(_PROVIDER)


class RuntimeMetrics:
    def __init__(self, provider: MeterProvider) -> None:
        meter = provider.get_meter("bank_assistant")
        self._requests = meter.create_counter("assistant.requests")
        self._errors = meter.create_counter("assistant.errors")
        self._duration = meter.create_histogram("assistant.request.duration", unit="ms")
        self._input_tokens = meter.create_histogram("llm.input.tokens", unit="{token}")
        self._output_tokens = meter.create_histogram("llm.output.tokens", unit="{token}")
        self._estimated_cost = meter.create_histogram("llm.estimated.cost", unit="USD")
        self._retrievals = meter.create_counter("rag.retrievals")
        self._tools = meter.create_counter("assistant.tools")
        self._retries = meter.create_counter("assistant.retries")

    def record_request(
        self,
        outcome: str,
        duration_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
        estimated_cost_usd: float | None,
    ) -> None:
        attrs = {"assistant.outcome": outcome}
        self._requests.add(1, attrs)
        if outcome == "failed":
            self._errors.add(1, attrs)
        self._duration.record(duration_ms, attrs)
        if input_tokens is not None:
            self._input_tokens.record(input_tokens)
        if output_tokens is not None:
            self._output_tokens.record(output_tokens)
        if estimated_cost_usd is not None:
            self._estimated_cost.record(estimated_cost_usd)

    def record_retrieval(self, count: int) -> None:
        self._retrievals.add(1, {"rag.result_count": count})

    def record_tool(self, name: str, outcome: str) -> None:
        self._tools.add(1, {"tool.name": name, "tool.outcome": outcome})

    def record_retry(self, operation: str) -> None:
        self._retries.add(1, {"retry.operation": operation})
