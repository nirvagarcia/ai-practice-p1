"""Query-time retrieval returning chunk IDs, scores, text and latency."""

from time import perf_counter
from typing import Protocol

from opentelemetry.trace import Tracer

from bank_assistant.observability.metrics import RuntimeMetrics
from bank_assistant.observability.tracing import get_tracer, start_span
from bank_assistant.rag.schemas import RetrievalResult, RetrievedChunk


class VectorIndex(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class Retriever:
    def __init__(
        self,
        index: VectorIndex,
        top_k: int,
        tracer: Tracer | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self._index = index
        self._top_k = top_k
        self._tracer = tracer or get_tracer()
        self._metrics = metrics

    def retrieve(self, query: str) -> RetrievalResult:
        started = perf_counter()
        with start_span(self._tracer, "rag.retrieve") as span:
            span.set_attribute("rag.top_k", self._top_k)
            chunks = self._index.search(query, self._top_k)
            span.set_attribute("rag.result_count", len(chunks))
            if self._metrics is not None:
                self._metrics.record_retrieval(len(chunks))
        return RetrievalResult(
            query=query,
            chunks=tuple(chunks),
            duration_ms=(perf_counter() - started) * 1000,
        )
