"""Retrieval-augmented answering: search the notes, then ask a model.

The whole point of RAG is that the model answers from the retrieved passages
rather than from memory. That is enforced in two places — the system prompt
below, and the fact that nothing but the retrieved text is ever sent.

Which model writes the answer is decided in llm.py; this file does not care.
"""

from __future__ import annotations

from collections.abc import Iterator

from .llm import ProviderError, get_provider
from .store import Retrieved, search

SYSTEM_PROMPT = """You answer questions about a student's university course notes.

Rules:
- Answer only from the numbered sources given in the user's message.
- If the sources do not contain the answer, say so plainly. Never fill the gap
  from your own knowledge, and never guess.
- Cite the sources you used inline, like [1] or [2, 3], attached to the claim
  they support.
- Quote a definition verbatim when the exact wording matters; otherwise explain
  it in your own words.
- Be concise. This is a study aid, not an essay."""


def _format_sources(hits: list[Retrieved]) -> str:
    """Lay the retrieved chunks out as a numbered list the model can cite."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.source}, page {hit.page}\n{hit.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Retrieved]) -> str:
    return f"Sources:\n\n{_format_sources(hits)}\n\n---\n\nQuestion: {question}"


def answer(question: str, k: int = 5) -> tuple[Iterator[str], list[Retrieved]]:
    """Retrieve, then stream a grounded answer.

    Returns the token iterator and the hits it was grounded in, so the caller
    can show the sources alongside the answer.
    """
    hits = search(question, k=k)

    if not hits:
        def empty() -> Iterator[str]:
            yield (
                "There is nothing in the index yet. Upload some PDFs in the "
                "sidebar and index them first."
            )
        return empty(), []

    prompt = build_prompt(question, hits)

    def stream_tokens() -> Iterator[str]:
        # Backend errors are turned into readable text rather than a traceback:
        # a chat box is the wrong place to show a stack trace, and the common
        # causes (Ollama not started, model not pulled) have obvious fixes.
        try:
            provider = get_provider()
            yield from provider.stream(SYSTEM_PROMPT, prompt)
        except ProviderError as exc:
            yield f"\n\n**Could not reach the model.** {exc}"

    return stream_tokens(), hits
