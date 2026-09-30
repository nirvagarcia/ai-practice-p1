"""Versioned Markdown chunking: one chunk per `##` section, prefixed with the document title."""

import re

from bank_assistant.rag.schemas import Chunk

CHUNKING_VERSION = "md-h2-v1"
_SECTION_START = re.compile(r"^(?=## )", re.MULTILINE)


def chunk_markdown(document_id: str, source: str, markdown: str) -> list[Chunk]:
    header, *sections = (part.strip() for part in _SECTION_START.split(markdown.strip()))
    bodies = ["\n\n".join(filter(None, (header, section))) for section in sections] or [header]
    return [
        Chunk(
            chunk_id=f"{document_id}#{index}",
            document_id=document_id,
            source=source,
            chunking_version=CHUNKING_VERSION,
            text=body,
        )
        for index, body in enumerate(bodies)
        if body
    ]
