"""Corrective RAG: bounded retrieve → grade → rewrite, then grounded generation or abstain."""

from typing import Protocol

from opentelemetry.trace import Tracer

from bank_assistant.observability.tracing import get_tracer, start_span
from bank_assistant.rag.grader import RelevanceGrade
from bank_assistant.rag.query_rewriter import QueryRewriter
from bank_assistant.rag.retriever import Retriever
from bank_assistant.rag.schemas import RetrievalResult
from bank_assistant.state.schemas import Outcome, latest_user_text, reply
from bank_assistant.workflows.faq.baseline import BaselineFaq, FaqResult, abstain_faq


class Grader(Protocol):
    def grade(self, question: str, retrieval: RetrievalResult) -> RelevanceGrade: ...


class CorrectiveFaq:
    def __init__(
        self,
        retriever: Retriever,
        generator: BaselineFaq,
        grader: Grader,
        rewriter: QueryRewriter,
        max_attempts: int,
        tracer: Tracer | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._retriever = retriever
        self._generator = generator
        self._grader = grader
        self._rewriter = rewriter
        self._max_attempts = max_attempts
        self._tracer = tracer or get_tracer()

    def answer(self, question: str) -> FaqResult:
        query = question
        retrieval = None
        for attempt in range(1, self._max_attempts + 1):
            retrieval = self._retriever.retrieve(query)
            if retrieval.chunks:
                try:
                    grade = self._grader.grade(question, retrieval)
                except Exception:
                    return abstain_faq(retrieval, attempt)
                if grade.relevant:
                    return self._generator.generate(question, retrieval, attempt)
            if attempt < self._max_attempts:
                try:
                    query = self._rewriter.rewrite(question)
                except Exception:
                    return abstain_faq(retrieval, attempt)

        if retrieval is None:
            raise RuntimeError("CRAG requires at least one retrieval attempt")
        return abstain_faq(retrieval, self._max_attempts)

    def __call__(self, state) -> dict:
        with start_span(self._tracer, "faq.crag") as span:
            result = self.answer(latest_user_text(state))
            span.set_attribute("rag.retrieval_attempts", result.retrieval_attempts)
            outcome = Outcome.ANSWERED if result.grounded else Outcome.ABSTAINED
            return reply(outcome, result.answer, source_ids=list(result.source_ids))
