from pathlib import Path

from bank_assistant.rag.chunking import CHUNKING_VERSION, chunk_markdown
from bank_assistant.rag.ingest import load_chunks

DOC = """# Tarjetas

## Bloqueo
Llama a emergencias.

## Reposición
Llega en 5 días.
"""


def test_each_section_keeps_document_title() -> None:
    chunks = chunk_markdown("tarjetas", "tarjetas.md", DOC)

    assert [c.chunk_id for c in chunks] == ["tarjetas#0", "tarjetas#1"]
    assert all(c.text.startswith("# Tarjetas") for c in chunks)
    assert "## Bloqueo" in chunks[0].text and "Reposición" not in chunks[0].text
    assert {c.chunking_version for c in chunks} == {CHUNKING_VERSION}


def test_document_without_sections_is_a_single_chunk() -> None:
    chunks = chunk_markdown("nota", "nota.md", "# Nota\nTexto plano.")

    assert len(chunks) == 1
    assert chunks[0].text == "# Nota\nTexto plano."


def test_chunking_is_deterministic() -> None:
    assert chunk_markdown("d", "d.md", DOC) == chunk_markdown("d", "d.md", DOC)


def test_knowledge_base_produces_unique_chunk_ids(knowledge_base_dir: Path) -> None:
    ids = [chunk.chunk_id for chunk in load_chunks(knowledge_base_dir)]

    assert ids
    assert len(ids) == len(set(ids))
