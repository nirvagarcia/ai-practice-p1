"""Explicit, offline indexing command: `python -m bank_assistant.rag.ingest`."""

from pathlib import Path

from bank_assistant.config.settings import ConfigurationError, get_settings
from bank_assistant.models.factory import build_embeddings
from bank_assistant.rag.chunking import CHUNKING_VERSION, chunk_markdown
from bank_assistant.rag.schemas import Chunk
from bank_assistant.rag.vectorstore import ChromaVectorStore


def load_chunks(source_dir: Path) -> list[Chunk]:
    return [
        chunk
        for path in sorted(source_dir.glob("*.md"))
        for chunk in chunk_markdown(path.stem, path.name, path.read_text(encoding="utf-8"))
    ]


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(settings.knowledge_base_path)
    if not chunks:
        raise SystemExit(f"No Markdown documents found in {settings.knowledge_base_path}")
    try:
        embeddings = build_embeddings(settings)
    except ConfigurationError as error:
        raise SystemExit(str(error)) from None
    ChromaVectorStore(settings.vectorstore_path, embeddings).rebuild(chunks)
    print(f"Indexed {len(chunks)} chunks ({CHUNKING_VERSION}) into {settings.vectorstore_path}")


if __name__ == "__main__":
    main()
