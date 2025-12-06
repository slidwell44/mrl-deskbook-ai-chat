from typing import Any, Mapping

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, collection_name: str = "documents") -> None:
        self.client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[
            Mapping[str, chromadb.SparseVector | bool | float | int | str | None]
        ],
    ) -> None:
        if not texts:
            return

        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Lengths must match: texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        ids: list[str] = [
            f"{m['doc_id']}_{m['page']}_{m['chunk_index']}" for m in metadatas
        ]

        self.collection.add(
            ids=ids,
            # pyrefly: ignore [bad-argument-type]
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        result: chromadb.QueryResult = self.collection.query(
            # pyrefly: ignore [bad-argument-type]
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[dict[str, Any]] = []

        if not result["ids"]:
            return docs

        for i in range(len(result["ids"][0])):
            docs.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i] if result["documents"] else "",
                    "metadata": result["metadatas"][0][i]
                    if result["metadatas"]
                    else {},
                    "distance": result["distances"][0][i]
                    if result.get("distances")
                    else None,
                }
            )
        return docs
