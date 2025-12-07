from pathlib import Path

from openai import OpenAI
from openai.pagination import SyncCursorPage
from openai.types import VectorStore
from openai.types.vector_stores.vector_store_file_batch import VectorStoreFileBatch

from config import settings

client = OpenAI(api_key=settings.openai.API_KEY)

vector_store_name = "MRL-Deskbook-AI-Chat-Store"


def create_vector_store_with_pdf() -> str:
    vector_stores: SyncCursorPage[VectorStore] = client.vector_stores.list()

    check: list[VectorStore] = [
        vs for vs in vector_stores.data if vs.name == vector_store_name
    ]
    if check:
        print("Vector store already exists:", check[0].name)
        return check[0].id

    vs: VectorStore = client.vector_stores.create(
        name=vector_store_name,
    )

    pdf_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "MRL_Deskbook_2025.pdf"
    )

    batch: VectorStoreFileBatch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=[open(pdf_path, "rb")],
    )

    print("Vector store:", vs.id)
    print("Batch status:", batch.status)
    print("File counts:", batch.file_counts)
    return vs.id


if __name__ == "__main__":
    create_vector_store_with_pdf()
