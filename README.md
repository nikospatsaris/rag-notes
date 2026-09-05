# Ask My Notes

A retrieval-augmented chatbot for university course notes. Drop in your lecture
PDFs, ask questions in plain language, and get answers that cite the document
and page they came from — or an honest "that isn't in your notes".

```
Q: What is the difference between TCP Tahoe and Reno?

A: Both halve ssthresh on congestion, but they differ in what happens to the
   congestion window. Tahoe drops cwnd to one MSS and re-enters slow start.
   Reno adds fast recovery, which halves cwnd instead of resetting it, because
   three duplicate ACKs prove later packets are still arriving [1].

   [1] networks-lecture-3.pdf, page 2
```

---

## The problem I solved

Revising from a term's worth of lecture PDFs means knowing *which* of forty
documents mentions the thing you half-remember. Full-text search only works if
you can guess the exact wording the lecturer used; you usually can't, because
the thing you're searching for is the thing you don't know yet.

The obvious fix — paste it into a chatbot — fails in a specific and dangerous
way. A general chatbot answers from what it memorised during training. Ask it
about *your* lecture on TCP congestion control and it will produce something
fluent, unsourced, and occasionally wrong. For revision that's worse than
useless: you can't tell the correct parts from the invented ones, and you have
no page to go and check.

So the requirement wasn't "answer questions". It was **answer questions in a
way I can verify**. That shapes the whole design:

- The model never answers from memory. Every question goes through retrieval
  first, and the model only ever sees passages pulled from the user's own PDFs.
- Every answer carries a document and page number, so any claim can be checked
  against the source in a few seconds.
- When the retrieved passages don't contain the answer, the system prompt
  requires the model to say so rather than fill the gap.

## How it works

```
PDF ─► extract text per page ─► split into overlapping chunks
                                          │
                                          ▼
                              embed locally (MiniLM, ONNX)
                                          │
                                          ▼
                              ChromaDB  (cosine similarity)
                                          │
   question ─► embed ─► top-k nearest chunks ─┘
                                          │
                                          ▼
              numbered sources + question ─► LLM ─► cited answer
```

**Chunking per page** (`rag/ingest.py`) — Text is split page by page into
~1000-character windows with 150 characters of overlap. Chunking per page
rather than over the whole document is what makes the citations trustworthy:
the page number is known exactly, not estimated from a character offset. The
overlap stops a definition that straddles a boundary from being lost by both
neighbours, and each window is trimmed back to a paragraph, sentence, or word
break so chunks don't begin mid-word and embed badly.

**Local embeddings** (`rag/store.py`) — `all-MiniLM-L6-v2` on onnxruntime, via
ChromaDB's default embedding function. Embedding costs nothing, needs no API
key, and works offline. Critically it avoids pulling in PyTorch: a ~2 GB
dependency doesn't fit in most free hosting tiers, and indexing is the one
operation that touches every page of every document.

Chroma defaults to squared L2 distance. Cosine is set explicitly, because it's
the right metric for normalised sentence embeddings.

**Stable chunk IDs** — Each chunk's ID is a SHA-256 hash of
`filename | page | text`, and writes are upserts. Indexing the same document
twice replaces those chunks instead of duplicating them, so re-indexing is
always safe to repeat — which matters, because the natural reaction to "did
that work?" is to click the button again.

**Grounded generation** (`rag/answer.py`, `rag/llm.py`) — The top-k chunks go
to the model as a numbered list, and the system prompt restricts it to those
sources. Responses stream token by token so the UI fills in immediately.

The model sits behind a one-method interface — give it a system prompt and a
user prompt, get back a stream of text — so nothing above `llm.py` knows or
cares which model is answering.

## Tools I used

| Tool | Why this one |
|---|---|
| **ChromaDB** | Embedded vector store — persists to a local directory with no server to run. A hosted vector DB would have meant an account, a key, and a network round trip per query, for a corpus that fits on disk. |
| **all-MiniLM-L6-v2** (ONNX) | Free, offline, ~80 MB. Good enough for prose at 384 dimensions, and no PyTorch. |
| **pypdf** | Pure Python PDF text extraction, no system binaries — so it installs the same on Windows and on a Linux host. |
| **Streamlit** | Chat UI, file upload, and streaming output in a few dozen lines. `st.write_stream` renders tokens as they arrive for free. |
| **Ollama / Gemini / Claude** | Three interchangeable backends behind one interface: Ollama for free local dev, Gemini's free tier for a deployed demo, Claude when answer quality matters most. |
| **reportlab** | Dev-only, used once to generate the sample PDFs in `samples/`. Not a runtime dependency. |

## Does it actually work?

Retrieval is the half of RAG that sets the ceiling: if the right passage never
comes back, no model can recover. So it's measured separately from generation,
against questions whose correct page is known:

```bash
python evaluate.py
```

```
precision@1: 7/7
worst in-domain distance:  0.688
best off-topic distance:   0.934
separation: +0.246 (a cutoff between them would work)
```

The second half of that is the more interesting result. Questions the notes
genuinely can't answer ("what is the recipe for carbonara?") come back at a
cosine distance clearly worse than every real hit. The gap means the system
could decline to answer *before* spending a model call — see below.

Caveat on the numbers: seven questions over five pages is a smoke test, not a
benchmark. It catches "retrieval is broken", not "retrieval is good".

## What I learned

**Retrieval quality is the whole game, and it needs its own metric.** My first
instinct was to judge the system by reading its answers. That conflates two
failures with different fixes — the passage never arrived, versus the model
mishandled a passage that did. Measuring precision@1 separately made the
distinction visible, and it's the number I'd actually iterate on.

**Chunking is a citation decision, not just a size decision.** Chunking across
the whole document is what most tutorials do, but then page numbers have to be
reconstructed from character offsets and drift by a page near boundaries.
Chunking per page costs a little retrieval quality on ideas that span a page
break, and buys exact citations. For a tool whose entire value proposition is
verifiability, that's the right trade.

**"I don't know" is a measurable property, not just a prompt instruction.** I'd
assumed refusal was purely the system prompt's job. Watching the distance
scores showed it's partly a retrieval property: off-topic questions are
*visibly* far away in embedding space before any model sees them.

**Pin your dependencies.** The first version of `requirements.txt` said
`chromadb>=0.5`, which now resolves to 1.5.9 — a major version the code was
written before. It happened to still work, but that was luck, not
compatibility. The `>=` was doing nothing except deferring a breakage to
whenever someone next cloned it.

**Choosing a model is mostly a deployment constraint.** The interesting
question wasn't which embedding model scores best, but which one still runs
where this has to run. Ruling out anything needing PyTorch decided it before
quality entered the discussion.

## What I'd improve next

**Refuse before calling the model.** The measured +0.246 gap between in-domain
and off-topic distances means a threshold could catch unanswerable questions at
retrieval time — saving a model call and removing the chance of the model
ignoring its instructions. I'd want a much larger question set before picking
the actual cutoff, since one badly-phrased real question landing above it would
be a worse failure than the one it prevents.

**Hybrid retrieval.** Right now it's pure vector similarity, which is
phrasing-sensitive in a specific way: exact identifiers embed poorly. A
question about "RFC 5681" or `ssthresh` can miss the page that defines it,
because rare tokens carry little semantic weight. BM25 running alongside the
vector search, with the two result sets fused, is the standard fix.

**Rewrite follow-up questions before retrieving.** Each question is embedded on
its own, so "and what about the second one?" retrieves nothing useful — the
pronoun carries the meaning and the embedding doesn't. Rewriting follow-ups
into standalone queries using the chat history would fix it.

**OCR for scanned PDFs.** Text is read from the PDF's text layer; a photo of a
page has none, and those pages are silently skipped. The UI warns when a
document yields zero chunks, but Tesseract would actually handle it.

**Adaptive chunk size.** 1000 characters is a compromise picked once and
applied to everything. Dense material wants smaller chunks, flowing prose
larger ones.

**A real evaluation set.** Seven questions is enough to catch a broken index
and nothing more. I'd want ~50 questions across more documents, plus a check on
the generated answers themselves — specifically, how often a cited page
actually supports the claim attached to it.

## Running it

```bash
git clone https://github.com/nikospatsaris/ask-my-notes
cd ask-my-notes
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

### Pick a model backend

`rag/llm.py` chooses automatically: whatever `LLM_BACKEND` names, else a
running Ollama, else whichever API key is set.

| Backend | Cost | Key | Works on free hosting |
|---|---|---|---|
| **Ollama** | free | none | no — needs a local machine |
| **Gemini** | free tier | free key | yes |
| **Anthropic** | paid | paid key | yes |

Fully offline, no key — install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
```

Or a free key that also works deployed, from
[aistudio.google.com](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
```

Then set `GEMINI_API_KEY` in the new `.env`. Indexing is free either way,
because embeddings run locally.

### Run

```bash
streamlit run app.py
```

The first run downloads the ~80 MB embedding model and caches it. Two sample
lecture PDFs are included in `samples/` if you want to try it before adding
your own notes.

## Deploying

Push to GitHub, then on [share.streamlit.io](https://share.streamlit.io) point
a new app at `app.py`. Add `GEMINI_API_KEY` under **Settings → Secrets** — the
app promotes Streamlit secrets to environment variables, so backend selection
works unchanged whether it's running locally or deployed.

Ollama can't be used for a deployed demo: free hosting has no GPU and no way to
run the model server, which is why the Gemini path exists.

## Project layout

```
app.py              Streamlit UI — upload, index, chat
evaluate.py         retrieval quality check against samples/
rag/ingest.py       PDF → cleaned, overlapping chunks
rag/store.py        embedding + ChromaDB persistence and search
rag/answer.py       retrieval → grounded prompt → streamed answer
rag/llm.py          pluggable backends (Ollama / Gemini / Claude)
samples/            two generated lecture PDFs, for trying it out
```

## Known limitations

- **Scanned PDFs are skipped.** No text layer, no chunks.
- **Retrieval is pure vector similarity**, so unusually-phrased questions and
  rare identifiers can miss.
- **No conversation memory in retrieval** — follow-up questions retrieve poorly.
- **Chunk size is fixed** at 1000 characters regardless of the material.
