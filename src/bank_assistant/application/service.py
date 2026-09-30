"""Application boundary used by CLI, evaluation and (later) HTTP. Hides graph internals."""

import logging
from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.trace import Tracer
from pydantic import BaseModel, ConfigDict

from bank_assistant.config.settings import Settings
from bank_assistant.domain.schemas import Intent, Principal
from bank_assistant.guardrails.injection_detection import detect_injection
from bank_assistant.guardrails.input_policy import InputPolicy
from bank_assistant.guardrails.moderation import ModerationResult, Moderator
from bank_assistant.guardrails.output_policy import OutputPolicy
from bank_assistant.observability.budget import RequestBudgetCallback
from bank_assistant.observability.metrics import RuntimeMetrics
from bank_assistant.observability.tracing import (
    get_tracer,
    record_error,
    record_result,
    record_usage,
    start_span,
)
from bank_assistant.state.schemas import Outcome

logger = logging.getLogger(__name__)

FAILURE_TEXT = "No he podido completar tu solicitud. Inténtalo de nuevo más tarde."


class AssistantReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    intent: Intent | None
    outcome: Outcome
    text: str
    source_ids: tuple[str, ...] = ()


class AssistantService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        max_steps: int,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
        metrics: RuntimeMetrics | None = None,
        input_policy: InputPolicy | None = None,
        output_policy: OutputPolicy | None = None,
        moderator: Moderator | None = None,
        moderation_fail_closed: bool = True,
    ) -> None:
        self._graph = graph
        self._max_steps = max_steps
        self._tracer = tracer or get_tracer()
        self._settings = settings
        self._metrics = metrics
        self._input_policy = input_policy or InputPolicy()
        self._output_policy = output_policy or OutputPolicy()
        self._moderator = moderator
        self._moderation_fail_closed = moderation_fail_closed

    def ask(self, message: str, principal: Principal) -> AssistantReply:
        started = perf_counter()
        request_id = uuid4().hex
        input_decision = self._input_policy.inspect(message)
        usage = (
            RequestBudgetCallback(self._settings)
            if self._settings is not None
            else UsageMetadataCallbackHandler()
        )
        initial = {
            "messages": [HumanMessage(input_decision.text)],
            "request_id": request_id,
            "customer_id": principal.customer_id,
        }
        with start_span(self._tracer, "assistant.request") as request_span:
            request_span.set_attribute("request.id", request_id)
            request_span.set_attribute("guardrail.input.allowed", input_decision.allowed)
            if not input_decision.allowed:
                request_span.set_attribute(
                    "guardrail.input.reason", input_decision.reason or "rejected"
                )
                reply = AssistantReply(
                    request_id=request_id,
                    intent=None,
                    outcome=Outcome.REJECTED,
                    text=(
                        "No puedo procesar una entrada vacía, demasiado larga o con caracteres "
                        "no permitidos."
                    ),
                )
                self._record_metrics(reply.outcome.value, started, usage)
                record_result(request_span, reply.outcome, reply.intent, 0)
                return reply
            signal = detect_injection(input_decision.text)
            request_span.set_attribute("security.injection_suspected", signal.detected)
            if self._moderator is not None:
                try:
                    moderation = self._moderator.check(input_decision.text)
                except Exception as error:
                    moderation = ModerationResult(flagged=False, error_type=type(error).__name__)
                request_span.set_attribute("moderation.flagged", moderation.flagged)
                request_span.set_attribute("moderation.failed", moderation.error_type is not None)
                if moderation.flagged or (
                    moderation.error_type is not None and self._moderation_fail_closed
                ):
                    reply = AssistantReply(
                        request_id=request_id,
                        intent=None,
                        outcome=Outcome.BLOCKED,
                        text="No puedo procesar esta solicitud según la política de contenido.",
                    )
                    self._record_metrics(reply.outcome.value, started, usage)
                    record_result(request_span, reply.outcome, reply.intent, 0)
                    return reply
            with start_span(self._tracer, "graph.invoke") as graph_span:
                graph_span.set_attribute("graph.max_steps", self._max_steps)
                try:
                    state = self._graph.invoke(
                        initial,
                        {"recursion_limit": self._max_steps, "callbacks": [usage]},
                    )
                except Exception as error:  # boundary: users never see internals
                    record_error(graph_span, error)
                    record_error(request_span, error)
                    self._record_usage(graph_span, usage)
                    self._record_usage(request_span, usage)
                    logger.error("request %s failed: %s", request_id, type(error).__name__)
                    self._record_metrics("failed", started, usage)
                    return AssistantReply(
                        request_id=request_id,
                        intent=None,
                        outcome=Outcome.FAILED,
                        text=FAILURE_TEXT,
                    )
                output_decision = self._output_policy.inspect(state["response"])
                reply = AssistantReply(
                    request_id=request_id,
                    intent=state.get("intent"),
                    outcome=state["outcome"] if output_decision.allowed else Outcome.BLOCKED,
                    text=output_decision.text,
                    source_ids=(
                        tuple(state.get("source_ids", ())) if output_decision.allowed else ()
                    ),
                )
                graph_span.set_attribute("guardrail.output.allowed", output_decision.allowed)
                graph_span.set_attribute("guardrail.output.redacted", output_decision.redacted)
                record_result(graph_span, reply.outcome, reply.intent, len(reply.source_ids))
                record_result(request_span, reply.outcome, reply.intent, len(reply.source_ids))
                self._record_usage(graph_span, usage)
                self._record_usage(request_span, usage)
                self._record_metrics(reply.outcome.value, started, usage)
                return reply

    def _record_metrics(
        self, outcome: str, started: float, usage: UsageMetadataCallbackHandler
    ) -> None:
        if self._metrics is None:
            return
        input_tokens, output_tokens = _token_totals(usage)
        cost = usage.estimated_cost_usd if isinstance(usage, RequestBudgetCallback) else None
        self._metrics.record_request(
            outcome,
            (perf_counter() - started) * 1000,
            input_tokens,
            output_tokens,
            cost,
        )

    @staticmethod
    def _record_usage(span, usage: UsageMetadataCallbackHandler) -> None:
        input_tokens, output_tokens = _token_totals(usage)
        cost = usage.estimated_cost_usd if isinstance(usage, RequestBudgetCallback) else None
        record_usage(span, input_tokens, output_tokens, cost)


def _token_totals(usage: UsageMetadataCallbackHandler) -> tuple[int | None, int | None]:
    totals = list(usage.usage_metadata.values())
    input_tokens = sum(item.get("input_tokens", 0) for item in totals) if totals else None
    output_tokens = sum(item.get("output_tokens", 0) for item in totals) if totals else None
    return input_tokens, output_tokens
