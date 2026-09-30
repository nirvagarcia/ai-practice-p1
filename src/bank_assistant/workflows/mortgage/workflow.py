"""Mortgage simulation: LLM extracts parameters, Python computes the payment."""

from langchain_core.language_models import BaseChatModel
from opentelemetry.trace import Tracer
from pydantic import BaseModel, ValidationError

from bank_assistant.domain.schemas import MortgageInput, MortgageQuote, Principal
from bank_assistant.guardrails.authorization import (
    AuthorizationPolicy,
    ToolAction,
    WorkflowName,
)
from bank_assistant.observability.metrics import RuntimeMetrics
from bank_assistant.observability.tracing import get_tracer, record_error, start_span
from bank_assistant.state.schemas import AssistantState, Outcome, latest_user_text, reply
from bank_assistant.tools.mortgage_calculator import calculate_payment

EXTRACTION_PROMPT = """Extrae los parámetros de una simulación hipotecaria del mensaje del cliente.
Usa null para cualquier dato que no aparezca explícitamente; nunca inventes valores.
loan_amount en euros, annual_rate_percent como porcentaje (3,5 % -> 3.5), term_years en años."""

OUT_OF_RANGE = (
    "Los datos están fuera de rango: importe hasta 10.000.000 €, tipo entre 0 y 30 % "
    "y plazo entre 1 y 40 años."
)

_FIELD_LABELS = {
    "loan_amount": "el importe del préstamo",
    "annual_rate_percent": "el tipo de interés anual",
    "term_years": "el plazo en años",
}


class MortgageRequest(BaseModel):
    loan_amount: float | None
    annual_rate_percent: float | None
    term_years: int | None


class MortgageWorkflow:
    def __init__(
        self,
        llm: BaseChatModel,
        tracer: Tracer | None = None,
        model_name: str | None = None,
        metrics: RuntimeMetrics | None = None,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._extractor = llm.with_structured_output(MortgageRequest)
        self._tracer = tracer or get_tracer()
        self._model_name = model_name
        self._metrics = metrics
        self._authorization = authorization or AuthorizationPolicy()

    def __call__(self, state: AssistantState) -> dict:
        with start_span(self._tracer, "llm.mortgage_extract") as span:
            span.set_attribute("llm.provider", "openai")
            if self._model_name:
                span.set_attribute("llm.model", self._model_name)
            try:
                request: MortgageRequest = self._extractor.invoke(
                    [("system", EXTRACTION_PROMPT), ("human", latest_user_text(state))]
                )
            except Exception as error:
                record_error(span, error)
                raise
        values = request.model_dump()
        if missing := [label for field, label in _FIELD_LABELS.items() if values[field] is None]:
            return reply(
                Outcome.NEEDS_INFO, f"Para simular la hipoteca necesito {', '.join(missing)}."
            )
        try:
            mortgage_input = MortgageInput.model_validate(values)
        except ValidationError:
            return reply(Outcome.NEEDS_INFO, OUT_OF_RANGE)
        authorization = self._authorization.authorize(
            Principal(customer_id=state["customer_id"]),
            WorkflowName.MORTGAGE,
            ToolAction.CALCULATE_PAYMENT,
            mortgage_input.model_dump(),
        )
        if not authorization.allowed:
            return reply(Outcome.BLOCKED, "No está autorizada la operación solicitada.")
        with start_span(self._tracer, "tool.calculate_payment") as span:
            span.set_attribute("tool.name", "calculate_payment")
            quote = calculate_payment(mortgage_input)
            if self._metrics is not None:
                self._metrics.record_tool("calculate_payment", "success")
        return reply(Outcome.ANSWERED, _describe(quote))


def _describe(quote: MortgageQuote) -> str:
    data = quote.input
    return (
        f"Cuota mensual estimada: {quote.monthly_payment} € para {data.loan_amount} € "
        f"a {data.term_years} años al {data.annual_rate_percent} %. "
        f"Total pagado: {quote.total_paid} € (intereses: {quote.total_interest} €). "
        "Simulación orientativa, no vinculante."
    )
