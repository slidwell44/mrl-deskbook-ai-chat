from pathlib import Path
from typing import Any, Dict, List

from google import genai
from pypdf import PdfReader

from src.config import settings
from src.vector_store import VectorStore

client = genai.Client(api_key=settings.gemini.API_KEY)


def load_pdf_text(path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def simple_chunk(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    # very simple char-based chunking
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
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
    pages = load_pdf_text(pdf_path)

    all_texts: List[str] = []
    all_embeddings: List[list[float]] = []
    all_metadata: List[Dict[str, Any]] = []

    for page in pages:
        chunks = simple_chunk(page["text"])
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding: genai.types.EmbedContentResponse = client.models.embed_content(
                model="gemini-embedding-001", contents=chunk
            )

            all_texts.append(chunk)
            all_embeddings.append(embedding)
            all_metadata.append(
                {
                    "doc_id": doc_id,
                    "page": page["page_num"],
                    "chunk_index": idx,
                }
            )

    vs.add_documents(all_texts, all_embeddings, all_metadata)
    print(
        f"Ingested {len(all_texts)} chunks from {pdf_path} into collection {collection_name}"
    )


if __name__ == "__main__":
    from pathlib import Path

    pdf_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "MRL_Deskbook_2025.pdf"
    )
    ingest_pdf(str(pdf_path), doc_id="MRL Deskbook 2025")
