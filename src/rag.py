"""RAG orchestration placeholders.

This module ties retrieval and generation. For the POC we'll keep a small
`generate_summary` function which accepts a `retriever` callable and an `llm`
callable. Both are injected to keep the function testable and model-agnostic.
"""

from typing import Callable, List


def generate_summary(query: str, retriever: Callable[[str, int], List[str]], llm: Callable[[str], str]) -> str:
    """Generate a summary for `query` using `retriever` and `llm`.

    Expected:
    - `retriever(query, k)` -> List[str]  (relevant document chunks as strings)
    - `llm(prompt)` -> str                (text output from model)

    This function concatenates retrieved context into a prompt template and
    calls the `llm` to produce a summary.
    """
    snippets = retriever(query, 5)
    context = "\n---\n".join(snippets)
    prompt = f"You are a helpful summarization assistant.\n\nContext:\n{context}\n\nPlease provide a concise, accurate summary of the above context with respect to: {query}"
    return llm(prompt)
