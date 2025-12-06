from google import genai

from config import settings
from vector_store import VectorStore

client = genai.Client(api_key=settings.gemini.API_KEY)


def retrieve_context(query: str, k: int = 5) -> list[dict]:
    vs = VectorStore(collection_name="documents")

    embedding: genai.types.EmbedContentResponse = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )

    if not embedding.embeddings:
        return []
    if embedding.embeddings[0].values is None:
        return []

    embedding_values: list[float] = embedding.embeddings[0].values

    return vs.similarity_search(embedding_values, k=k)


def format_context(chunks: list[dict]) -> str:
    formatted: list[str] = []
    for c in chunks:
        meta = c["metadata"]
        snippet = c["text"].strip()
        formatted.append(
            f"[Page {meta['page']} | chunk {meta['chunk_index']}]\n{snippet}"
        )
    return "\n\n---\n\n".join(formatted)
