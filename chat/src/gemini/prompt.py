from google import genai

from config import settings
from gemini.retriever import format_context, retrieve_context

client = genai.Client(api_key=settings.gemini.API_KEY)


RAG_SYSTEM_PROMPT = """
You are a helpful assistant answering questions about a specific document.
Use ONLY the provided context to answer. If the answer is not in the context,
say you don't know and suggest what section of the document might contain it.
"""


def answer_question_rag(question: str) -> genai.types.GenerateContentResponse:
    chunks: list[dict] = retrieve_context(question, k=5)

    if not chunks:
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            "Context:\nNo relevant context could be retrieved from the document.\n\n"
            f"Question: {question}\n\n"
            "Say you don't know."
        )
    else:
        context: str = format_context(chunks)
        prompt: str = f"""
                    {RAG_SYSTEM_PROMPT}

                    Context:
                    {context}

                    Question: {question}

                    Answer, citing page numbers when helpful.
                    """

    response: genai.types.GenerateContentResponse = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
    )
    return response


if __name__ == "__main__":
    answer: genai.types.GenerateContentResponse = answer_question_rag(
        "What is the main objective of this document?"
    )
    print(answer.text)
