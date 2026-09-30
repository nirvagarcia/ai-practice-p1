"""Chroma adapter persisted on local disk. Embeddings are computed by the injected model."""

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

from bank_assistant.rag.schemas import Chunk, RetrievedChunk


class ChromaVectorStore:
    def __init__(self, path: Path, embeddings: Embeddings, collection: str = "knowledge_base"):
        self._client = chromadb.PersistentClient(
            path=path, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._embeddings = embeddings
        self._name = collection
        self._collection = self._open()

    def _open(self) -> Collection:
        return self._client.get_or_create_collection(
            self._name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        """Replace the whole index so removed documents never linger."""
        self._client.delete_collection(self._name)
        self._collection = self._open()
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            metadatas=[chunk.model_dump(exclude={"chunk_id", "text"}) for chunk in chunks],
            embeddings=self._embeddings.embed_documents(texts),
        )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        size = self._collection.count()
        if size == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(query)],
            n_results=min(k, size),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(chunk_id=chunk_id, text=text, score=1 - distance, **metadata)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
