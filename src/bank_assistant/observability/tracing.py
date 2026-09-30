"""OpenTelemetry setup and safe workflow-node spans."""

from collections.abc import Callable
from threading import Lock
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

T = TypeVar("T")
_LOCK = Lock()
_PROVIDER: TracerProvider | None = None


def create_tracer_provider(exporter: SpanExporter | None = None) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "bank-assistant-lab"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
    return provider


def configure_tracing() -> Tracer:
    """Install one process-wide console exporter; repeated calls reuse the provider."""
    global _PROVIDER
    with _LOCK:
        if _PROVIDER is None:
            provider = create_tracer_provider()
            trace.set_tracer_provider(provider)
            _PROVIDER = provider
    return _PROVIDER.get_tracer("bank_assistant")


def get_tracer() -> Tracer:
    return trace.get_tracer("bank_assistant")


def start_span(tracer: Tracer, name: str):
    return tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False)


def trace_node(
    name: str, node: Callable[..., dict[str, Any]], tracer: Tracer
) -> Callable[..., dict[str, Any]]:
    def invoke(state: Any) -> dict[str, Any]:
        with start_span(tracer, f"assistant.{name}") as span:
            try:
                result = node(state)
            except Exception as error:
                span.set_attribute("error.type", type(error).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            for key in ("intent", "outcome"):
                if value := result.get(key):
                    span.set_attribute(f"assistant.{key}", str(value))
            if "source_ids" in result:
                span.set_attribute("rag.source_count", len(result["source_ids"]))
            return result

    return invoke


def record_error(span: Any, error: BaseException) -> None:
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(Status(StatusCode.ERROR))


def record_result(span: Any, outcome: Any, intent: Any, source_count: int) -> None:
    if intent is not None:
        span.set_attribute("assistant.intent", str(intent))
    span.set_attribute("assistant.outcome", str(outcome))
    span.set_attribute("rag.source_count", source_count)


def record_usage(
    span: Any,
    input_tokens: int | None,
    output_tokens: int | None,
    estimated_cost_usd: float | None,
) -> None:
    if input_tokens is not None:
        span.set_attribute("llm.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("llm.output_tokens", output_tokens)
    if estimated_cost_usd is not None:
        span.set_attribute("llm.estimated_cost_usd", estimated_cost_usd)
