"""Measure retrieval quality against the sample notes.

Retrieval is the half of RAG that decides whether a correct answer is even
possible: if the right passage never comes back, no model can save it. This
script indexes samples/ and checks that a set of questions retrieves the page
that actually contains the answer.

    python evaluate.py
"""

from __future__ import annotations

import sys

from rag.ingest import load_folder
from rag.store import add_chunks, search, stats

# (question, expected source, expected page) — the page a human would cite.
CASES = [
    ("What is the difference between TCP Tahoe and Reno?", "networks-lecture-3.pdf", 2),
    ("Why does slow start double the window?", "networks-lecture-3.pdf", 1),
    ("What problem does BBR solve?", "networks-lecture-3.pdf", 3),
    ("What causes congestion collapse?", "networks-lecture-3.pdf", 1),
    ("What is a partial dependency?", "databases-lecture-7.pdf", 1),
    ("When would you deliberately not use BCNF?", "databases-lecture-7.pdf", 2),
    ("Why might you denormalise a schema?", "databases-lecture-7.pdf", 2),
]

# Questions the notes genuinely cannot answer. These have no correct passage,
# so the useful signal is the distance: it should be clearly worse than the
# in-domain hits, which is what a "not in your notes" cutoff would key on.
OUT_OF_DOMAIN = [
    "What is the recipe for carbonara?",
    "Who won the 2022 World Cup?",
]


def main() -> int:
    print(f"indexing samples/ ... ", end="", flush=True)
    add_chunks(load_folder("samples"))
    print(f"{stats()['chunks']} passages\n")

    worst_in_domain = 0.0
    correct = 0

    for question, want_source, want_page in CASES:
        hits = search(question, k=3)
        if not hits:
            print(f"FAIL  (nothing retrieved)  {question}")
            continue

        top = hits[0]
        ok = top.source == want_source and top.page == want_page
        correct += ok
        worst_in_domain = max(worst_in_domain, top.distance)

        print(f"{'PASS' if ok else 'FAIL'}  {top.distance:.3f}  {question}")
        if not ok:
            print(f"        got {top.source} p{top.page}, "
                  f"wanted {want_source} p{want_page}")

    print(f"\nprecision@1: {correct}/{len(CASES)}")

    best_off_topic = min(search(q, k=1)[0].distance for q in OUT_OF_DOMAIN)
    print(f"worst in-domain distance:  {worst_in_domain:.3f}")
    print(f"best off-topic distance:   {best_off_topic:.3f}")
    print(
        f"separation: {best_off_topic - worst_in_domain:+.3f} "
        f"({'a cutoff between them would work' if best_off_topic > worst_in_domain else 'they overlap — a cutoff would misfire'})"
    )

    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
