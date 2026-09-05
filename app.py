"""Streamlit front end: upload notes, index them, ask questions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.ingest import load_pdf
from rag.store import add_chunks, reset, stats
from rag.answer import answer
from rag.llm import ProviderError, get_provider

load_dotenv()

st.set_page_config(page_title="Ask My Notes", page_icon="📚", layout="centered")


def load_secrets() -> None:
    """Copy any Streamlit secrets into the environment.

    Locally the keys come from .env; on Streamlit Cloud there is no .env, so
    whatever is configured under Settings -> Secrets is promoted to env vars
    and the backend selection in rag/llm.py works unchanged either way.
    """
    for name in ("LLM_BACKEND", "GEMINI_API_KEY", "GEMINI_MODEL",
                 "ANTHROPIC_API_KEY", "OLLAMA_HOST", "OLLAMA_MODEL"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets[name]
        except Exception:
            continue
        if value:
            os.environ[name] = str(value)


# ── sidebar: indexing ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Your notes")

    uploaded = st.file_uploader(
        "Add PDFs", type="pdf", accept_multiple_files=True,
        help="Lecture slides, notes, past papers — anything with a text layer.",
    )

    if uploaded and st.button("Index these files", type="primary"):
        progress = st.progress(0.0, text="Reading…")
        total_chunks = 0

        for i, upload in enumerate(uploaded):
            # pypdf needs a real path, so the upload is spilled to a temp file.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = Path(tmp.name)

            try:
                # Pass the real upload name so citations and chunk IDs are
                # based on it rather than on the throwaway temp filename.
                chunks = load_pdf(tmp_path, source_name=upload.name)
                total_chunks += add_chunks(chunks)
            finally:
                tmp_path.unlink(missing_ok=True)

            progress.progress(
                (i + 1) / len(uploaded), text=f"Indexed {upload.name}"
            )

        progress.empty()
        if total_chunks:
            st.success(f"Indexed {total_chunks} passages.")
        else:
            st.warning(
                "No text found. These PDFs are probably scans — they would "
                "need OCR first."
            )

    info = stats()
    st.metric("Passages indexed", info["chunks"])
    if info["documents"]:
        with st.expander(f"{len(info['documents'])} documents"):
            for doc in info["documents"]:
                st.caption(doc)
        if st.button("Clear index"):
            reset()
            st.rerun()

    st.divider()
    top_k = st.slider(
        "Passages per answer", 3, 10, 5,
        help="How many chunks get sent to the model as context.",
    )


# ── main: chat ─────────────────────────────────────────────────────────
st.title("Ask My Notes")
st.caption(
    "Answers come only from the PDFs you upload — with the page they came from."
)

load_secrets()

try:
    provider = get_provider()
    st.caption(f"Answering with **{provider.label}**")
except ProviderError as exc:
    st.error(str(exc))
    st.info(
        "**Free, no key:** install [Ollama](https://ollama.com), then run "
        "`ollama pull llama3.2` and `ollama serve`.\n\n"
        "**Free key, works when deployed:** get one at "
        "[aistudio.google.com](https://aistudio.google.com/apikey) and put "
        "`GEMINI_API_KEY=...` in your `.env`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {src['source']}, page {src['page']}**")
                    st.caption(src["text"])

if question := st.chat_input("Ask something about your notes…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        tokens, hits = answer(question, k=top_k)
        text = st.write_stream(tokens)

        sources = [
            {"source": h.source, "page": h.page, "text": h.text[:400] + "…"}
            for h in hits
        ]
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {src['source']}, page {src['page']}**")
                    st.caption(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )
