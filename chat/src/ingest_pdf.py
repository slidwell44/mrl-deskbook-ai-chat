from pathlib import Path
from typing import Any, Mapping

import chromadb
from google import genai
from pypdf import PdfReader

from config import settings
from vector_store import VectorStore

client = genai.Client(api_key=settings.gemini.API_KEY)


def load_pdf_text(path: str) -> list[dict[str, Any]]:
    reader = PdfReader(path)
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text: str = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def simple_chunk(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end: int = start + max_chars
        chunk: str = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_pdf(
    pdf_path: str,
    doc_id: str | None = None,
    collection_name: str = "documents",
) -> None:
    doc_id = doc_id or Path(pdf_path).stem
    vs = VectorStore(collection_name=collection_name)
    pages: list[dict[str, Any]] = load_pdf_text(pdf_path)

    chunk_count: int = 0

    for page in pages:
        chunks: list[str] = simple_chunk(page["text"])

        page_texts: list[str] = []
        page_embeddings: list[list[float]] = []
        page_metadatas: list[
            Mapping[str, chromadb.SparseVector | bool | float | int | str | None]
        ] = []

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            embedding: genai.types.EmbedContentResponse = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk,
            )

            if not embedding.embeddings:
                continue
            if embedding.embeddings[0].values is None:
                continue

            embedding_values: list[float] = embedding.embeddings[0].values

            metadata: dict[str, Any] = {
                "doc_id": doc_id,
                "page": page["page_num"],
                "chunk_index": idx,
            }

            page_texts.append(chunk)
            page_embeddings.append(embedding_values)
            page_metadatas.append(metadata)
            chunk_count += 1

        if page_texts:
            vs.add_documents(page_texts, page_embeddings, page_metadatas)

    print(
        f"Ingested {chunk_count} chunks from {pdf_path} into collection {collection_name}"
    )


if __name__ == "__main__":
    pdf_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "MRL_Deskbook_2025.pdf"
    )
    ingest_pdf(str(pdf_path), doc_id="MRL Deskbook 2025")
