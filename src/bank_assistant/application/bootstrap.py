"""Composition root: the only place that wires settings, providers, adapters and workflows."""

from opentelemetry.trace import Tracer

from bank_assistant.application.service import AssistantService
from bank_assistant.config.settings import Settings
from bank_assistant.graph.builder import GraphNodes, build_graph
from bank_assistant.guardrails.input_policy import InputPolicy
from bank_assistant.guardrails.moderation import OpenAIModerator
from bank_assistant.guardrails.output_policy import OutputPolicy
from bank_assistant.models.config import ModelRole
from bank_assistant.models.factory import (
    build_chat_model,
    build_embeddings,
    build_moderation_client,
)
from bank_assistant.observability.metrics import configure_metrics
from bank_assistant.observability.tracing import configure_tracing
from bank_assistant.rag.grader import ContextGrader
from bank_assistant.rag.query_rewriter import QueryRewriter
from bank_assistant.rag.retriever import Retriever
from bank_assistant.rag.vectorstore import ChromaVectorStore
from bank_assistant.tools.account_api import InMemoryAccountGateway
from bank_assistant.workflows.account.workflow import AccountWorkflow
from bank_assistant.workflows.card_block.workflow import request_card_block
from bank_assistant.workflows.faq.baseline import BaselineFaq
from bank_assistant.workflows.faq.crag import CorrectiveFaq
from bank_assistant.workflows.mortgage.workflow import MortgageWorkflow
from bank_assistant.workflows.triage import LlmTriage


def build_triage(settings: Settings, tracer: Tracer | None = None) -> LlmTriage:
    return LlmTriage(build_chat_model(ModelRole.ROUTER, settings), tracer, settings.router_model)


def build_faq(
    settings: Settings,
    tracer: Tracer | None = None,
    metrics=None,
    strategy: str | None = None,
) -> BaselineFaq | CorrectiveFaq:
    tracer = tracer or configure_tracing()
    store = ChromaVectorStore(settings.vectorstore_path, build_embeddings(settings))
    retriever = Retriever(store, settings.retrieval_top_k, tracer, metrics)
    generator = BaselineFaq(
        retriever,
        build_chat_model(ModelRole.GENERATION, settings),
        tracer,
        settings.generation_model,
    )
    selected = strategy or settings.rag_strategy
    if selected == "rag-v1":
        return generator
    if selected == "crag":
        return CorrectiveFaq(
            retriever,
            generator,
            ContextGrader(
                build_chat_model(ModelRole.GRADER, settings), tracer, settings.grader_model
            ),
            QueryRewriter(
                build_chat_model(ModelRole.ROUTER, settings), tracer, settings.router_model
            ),
            settings.max_retrieval_attempts,
            tracer,
        )
    raise ValueError(f"Unsupported RAG strategy: {selected}")


def build_assistant(settings: Settings) -> AssistantService:
    tracer = configure_tracing()
    metrics = configure_metrics()
    nodes = GraphNodes(
        triage=build_triage(settings, tracer),
        faq=build_faq(settings, tracer, metrics),
        mortgage=MortgageWorkflow(
            build_chat_model(ModelRole.GENERATION, settings),
            tracer,
            settings.generation_model,
            metrics,
        ),
        account=AccountWorkflow(
            InMemoryAccountGateway.from_fixture(settings.mocks_path / "accounts.json"),
            tracer,
            metrics,
        ),
        card_block=request_card_block,
    )
    graph = build_graph(nodes, settings.router_min_confidence, tracer)
    return AssistantService(
        graph,
        settings.max_graph_steps,
        tracer,
        settings,
        metrics,
        InputPolicy(settings.max_input_chars),
        OutputPolicy(settings.max_output_chars),
        OpenAIModerator(build_moderation_client(settings)) if settings.enable_moderation else None,
        settings.moderation_fail_closed,
    )
