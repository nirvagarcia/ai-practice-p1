"""RAGAS 0.4.3 adapter (async `metrics.collections`). Pinned: RAGAS APIs change between versions."""

import asyncio
import os
from dataclasses import dataclass

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from bank_assistant.config.settings import Settings


@dataclass(frozen=True)
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    reference: str


class RagasScorer:
    def __init__(self, settings: Settings) -> None:
        client = AsyncOpenAI(
            api_key=settings.require_openai_api_key().get_secret_value(),
            timeout=settings.llm_timeout_seconds,
        )
        llm = llm_factory(settings.evaluator_model, client=client)
        embeddings = embedding_factory("openai", settings.embedding_model, client=client)
        self._faithfulness = Faithfulness(llm=llm)
        self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self._context_precision = ContextPrecisionWithReference(llm=llm)
        self._context_recall = ContextRecall(llm=llm)

    def score_all(self, samples: list[RagasSample]) -> list[dict[str, float | str]]:
        return asyncio.run(self._score_all(samples))

    async def _score_all(self, samples: list[RagasSample]) -> list[dict[str, float | str]]:
        return [await self._score_safely(sample) for sample in samples]

    async def _score_safely(self, sample: RagasSample) -> dict[str, float | str]:
        try:
            return await self._score(sample)
        except Exception as error:  # one failing judge call must not discard the whole run
            return {"error": type(error).__name__}

    async def _score(self, s: RagasSample) -> dict[str, float | str]:
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            self._faithfulness.ascore(
                user_input=s.question, response=s.answer, retrieved_contexts=s.contexts
            ),
            self._answer_relevancy.ascore(user_input=s.question, response=s.answer),
            self._context_precision.ascore(
                user_input=s.question, reference=s.reference, retrieved_contexts=s.contexts
            ),
            self._context_recall.ascore(
                user_input=s.question, retrieved_contexts=s.contexts, reference=s.reference
            ),
        )
        return {
            "faithfulness": faithfulness.value,
            "answer_relevancy": relevancy.value,
            "context_precision": precision.value,
            "context_recall": recall.value,
        }
