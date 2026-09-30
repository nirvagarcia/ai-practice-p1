"""RAG baseline over a real on-disk Chroma index with offline fakes for embeddings and LLM."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from bank_assistant.rag.ingest import load_chunks
from bank_assistant.rag.retriever import Retriever
from bank_assistant.rag.vectorstore import ChromaVectorStore
from bank_assistant.workflows.faq.baseline import ABSTENTION, BaselineFaq, FaqDraft
from tests.fakes import FakeStructuredLLM


class KeywordEmbeddings(Embeddings):
    """Bag-of-keywords vectors: deterministic and good enough to rank topical documents."""

    VOCABULARY = ("horario", "oficinas", "tarjeta", "robo", "hipoteca", "cuenta", "descubierto")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in self.VOCABULARY] + [0.01]


@pytest.fixture
def store(tmp_path: Path, knowledge_base_dir: Path) -> ChromaVectorStore:
    store = ChromaVectorStore(tmp_path / "vectorstore", KeywordEmbeddings())
    store.rebuild(load_chunks(knowledge_base_dir))
    return store


def test_expected_document_is_in_top_k(store: ChromaVectorStore) -> None:
    result = Retriever(store, top_k=2).retrieve("¿Cuál es el horario de las oficinas?")

    assert result.chunks[0].chunk_id == "horarios#0"
    assert result.chunks[0].score >= result.chunks[-1].score
    assert result.duration_ms >= 0


def test_rebuild_replaces_previous_index(store: ChromaVectorStore) -> None:
    store.rebuild([])

    assert store.search("horario", k=3) == []


def test_answer_keeps_only_citations_that_were_retrieved(store: ChromaVectorStore) -> None:
    llm = FakeStructuredLLM(
        FaqDraft(answerable=True, answer="8:30-14:30", source_ids=["horarios#0", "x#9"])
    )

    result = BaselineFaq(Retriever(store, top_k=2), llm).answer("horario de oficinas")

    assert result.grounded
    assert result.source_ids == ("horarios#0",)


@pytest.mark.parametrize(
    "draft",
    [
        FaqDraft(answerable=False, answer="", source_ids=[]),
        FaqDraft(answerable=True, answer="Inventado", source_ids=["unknown#0"]),
    ],
)
def test_unsupported_answer_abstains(store: ChromaVectorStore, draft: FaqDraft) -> None:
    result = BaselineFaq(Retriever(store, top_k=2), FakeStructuredLLM(draft)).answer(
        "seguro de mascotas"
    )

    assert not result.grounded
    assert result.answer == ABSTENTION


def test_empty_index_abstains_without_calling_llm(tmp_path: Path) -> None:
    llm = FakeStructuredLLM(FaqDraft(answerable=True, answer="?", source_ids=[]))
    empty = ChromaVectorStore(tmp_path / "empty", KeywordEmbeddings())

    result = BaselineFaq(Retriever(empty, top_k=2), llm).answer("horario")

    assert result.answer == ABSTENTION
    assert llm.calls == 0
