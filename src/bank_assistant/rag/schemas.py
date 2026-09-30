"""RAG data contracts shared by ingestion, storage, retrieval and evaluation."""

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source: str
    chunking_version: str
    text: str


class RetrievedChunk(Chunk):
    score: float  # cosine similarity: a ranking signal, not a calibrated probability


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    chunks: tuple[RetrievedChunk, ...]
    duration_ms: float
