from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
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
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        ids = [f"{m['doc_id']}_{m['page']}_{m['chunk_index']}" for m in metadatas]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        # massage into a nicer format
        docs = []
        for i in range(len(result["ids"][0])):
            docs.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                }
            )
        return docs
