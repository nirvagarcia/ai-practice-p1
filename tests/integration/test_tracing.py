from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from bank_assistant.application.service import AssistantService
from bank_assistant.domain.schemas import Intent, Principal
from bank_assistant.graph.builder import GraphNodes, build_graph
from bank_assistant.observability.tracing import create_tracer_provider
from bank_assistant.rag.retriever import Retriever
from bank_assistant.rag.schemas import RetrievedChunk
from bank_assistant.state.schemas import AssistantState, Outcome, reply
from bank_assistant.tools.account_api import InMemoryAccountGateway
from bank_assistant.workflows.account.workflow import AccountWorkflow
from bank_assistant.workflows.faq.baseline import BaselineFaq, FaqDraft
from bank_assistant.workflows.mortgage.workflow import MortgageRequest, MortgageWorkflow
from tests.fakes import FakeStructuredLLM, user_state

SECRET_TEXT = "private prompt sentinel 739"


class FixedIndex:
    def search(self, _query: str, _k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="horarios#0",
                document_id="horarios",
                source="horarios.md",
                chunking_version="test",
                text="Las oficinas abren a las 8:30.",
                score=0.9,
            )
        ]


def test_trace_has_safe_hierarchical_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = create_tracer_provider(exporter)
    tracer = provider.get_tracer("test")
    faq = BaselineFaq(
        Retriever(FixedIndex(), top_k=2, tracer=tracer),
        FakeStructuredLLM(
            FaqDraft(answerable=True, answer="A las 8:30", source_ids=["horarios#0"])
        ),
        tracer,
        "test-model",
    )
    nodes = GraphNodes(
        triage=lambda _: {"intent": Intent.FAQ, "confidence": 0.9},
        faq=faq,
        mortgage=lambda _: reply(Outcome.ANSWERED, "mortgage"),
        account=lambda _: reply(Outcome.ANSWERED, "account"),
        card_block=lambda _: reply(Outcome.HANDOFF, "handoff"),
    )
    graph = build_graph(nodes, min_confidence=0.6, tracer=tracer)
    result = AssistantService(graph, max_steps=12, tracer=tracer).ask(
        SECRET_TEXT, Principal(customer_id="CUST-0001")
    )

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert result.outcome is Outcome.ANSWERED
    assert {
        "assistant.request",
        "graph.invoke",
        "assistant.triage",
        "assistant.faq",
        "rag.retrieve",
        "llm.generate",
    } <= names
    assert all(span.parent is not None for span in spans if span.name != "assistant.request")
    assert any(
        span.name == "rag.retrieve" and span.attributes["rag.result_count"] == 1 for span in spans
    )
    assert any(
        span.name == "llm.generate" and span.attributes["llm.model"] == "test-model"
        for span in spans
    )
    assert "assistant.outcome" in next(s for s in spans if s.name == "assistant.request").attributes
    assert all(SECRET_TEXT not in repr(span.attributes) for span in spans)
    assert all(SECRET_TEXT not in repr(span.events) for span in spans)
    provider.shutdown()


def test_failed_request_records_error_type_without_exception_payload() -> None:
    exporter = InMemorySpanExporter()
    provider = create_tracer_provider(exporter)
    tracer = provider.get_tracer("test")

    def fail(_state: AssistantState) -> dict:
        raise RuntimeError(SECRET_TEXT)

    nodes = GraphNodes(
        triage=fail,
        faq=lambda _: {},
        mortgage=lambda _: {},
        account=lambda _: {},
        card_block=lambda _: {},
    )
    result = AssistantService(
        build_graph(nodes, min_confidence=0.6, tracer=tracer), max_steps=12, tracer=tracer
    ).ask(SECRET_TEXT, Principal(customer_id="CUST-0001"))

    spans = exporter.get_finished_spans()
    failed = [span for span in spans if span.status.status_code.name == "ERROR"]
    assert result.outcome is Outcome.FAILED
    assert failed
    assert all(span.attributes.get("error.type") == "RuntimeError" for span in failed)
    assert all(not span.events for span in failed)
    assert all(SECRET_TEXT not in repr(span.attributes) for span in spans)
    provider.shutdown()


def test_provider_uses_lab_resource() -> None:
    provider = create_tracer_provider()

    assert isinstance(provider, TracerProvider)
    assert provider.resource.attributes["service.name"] == "bank-assistant-lab"
    provider.shutdown()


def test_tool_spans_record_names_and_counts_without_arguments(mocks_dir) -> None:
    exporter = InMemorySpanExporter()
    provider = create_tracer_provider(exporter)
    tracer = provider.get_tracer("test")
    mortgage = MortgageWorkflow(
        FakeStructuredLLM(
            MortgageRequest(loan_amount=200_000, annual_rate_percent=5, term_years=30)
        ),
        tracer,
    )
    account = AccountWorkflow(
        InMemoryAccountGateway.from_fixture(mocks_dir / "accounts.json"), tracer
    )
    mortgage(user_state(SECRET_TEXT))
    account(user_state(SECRET_TEXT))

    spans = exporter.get_finished_spans()
    assert {"llm.mortgage_extract", "tool.calculate_payment", "tool.get_account"} <= {
        span.name for span in spans
    }
    account_span = next(span for span in spans if span.name == "tool.get_account")
    assert account_span.attributes["tool.result_count"] == 1
    assert all(SECRET_TEXT not in repr(span.attributes) for span in spans)
    provider.shutdown()
