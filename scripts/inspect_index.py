"""Inspect the built indexes and demonstrate why hybrid retrieval is needed.

Two jobs:

1. Assert the invariants indexing promises - dense and sparse describing the same corpus,
   metadata intact, scores on the scale the relevance floor assumes.
2. Run paired probe queries through each retriever separately. Conceptual paraphrases should
   favour the dense side; exact identifiers - district names, section numbers, figures -
   should favour BM25. This is the observed failure mode that motivates fusion, rather than
   fusion being adopted because it is fashionable.

Run:  uv run python scripts/inspect_index.py
"""

from __future__ import annotations

import sys

from citara.config import get_settings
from citara.embeddings import Embedder
from citara.indexing.builder import IndexManifest, load_chunks
from citara.indexing.sparse_index import SparseIndex
from citara.indexing.vector_store import VectorStore

# Queries chosen to separate the two retrievers, not to flatter either.
CONCEPTUAL = [
    "what should people do when water rises",
    "how are relief camps organised",
    "who decides to declare an emergency",
]
IDENTIFIER = [
    "Quetta",
    "Thatta district",
    "PKR 800 billion",
]


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{f' - {detail}' if detail else ''}")
    return passed


def main() -> int:
    settings = get_settings()
    data_dir = settings.paths.resolved(settings.paths.data_dir)
    manifest_path = settings.paths.resolved(settings.paths.index_manifest_path)
    if not manifest_path.exists():
        print("No index. Run: uv run python -m citara.indexing")
        return 2

    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    chunks = load_chunks(data_dir / "chunks.jsonl")
    sparse = SparseIndex.load(settings.paths.resolved(settings.paths.bm25_path))
    store = VectorStore(settings)
    embedder = Embedder(settings.embedding)

    try:
        dense_count = store.count()
        print(f"\nindex built {manifest.built_at:%Y-%m-%d %H:%M} | {manifest.embedding_model}")
        print(f"  dense {dense_count} | sparse {len(sparse)} | chunks {len(chunks)}")
        print(f"  documents {len(manifest.documents)} | dimension {manifest.dimension}")

        print("\ninvariants:")
        probe = store.search(embedder.embed_query("flood response"), k=3)
        results = [
            check(
                "dense and sparse cover the same corpus",
                dense_count == len(sparse) == len(chunks),
                f"dense={dense_count} sparse={len(sparse)} chunks={len(chunks)}",
            ),
            check(
                "embedding dimension matches the model",
                manifest.dimension == embedder.dimension,
                f"{manifest.dimension} vs {embedder.dimension}",
            ),
            check("search returns results", bool(probe)),
            check(
                "scores are cosine similarities in [0, 1]",
                all(-0.01 <= hit.score <= 1.01 for hit in probe),
                f"{[round(h.score, 3) for h in probe]}",
            ),
            check(
                "citations survive into the index",
                all(hit.metadata.get("citation") for hit in probe),
            ),
            check(
                "chunk count matches the manifest",
                manifest.chunk_count == len(chunks),
                f"{manifest.chunk_count} vs {len(chunks)}",
            ),
        ]

        # Approximate search is only worth its speed if it finds what exhaustive search would.
        # Brute force over the stored vectors is the ground truth at this corpus size.
        vectors_path = data_dir / "chunk_vectors.npy"
        if vectors_path.exists():
            import numpy as np

            vectors = np.load(vectors_path)
            if vectors.shape[0] == len(chunks):
                ids = [c.chunk_id for c in chunks]
                norms = np.linalg.norm(vectors, axis=1)
                top1 = 0
                overlap = 0
                for query in CONCEPTUAL + IDENTIFIER:
                    query_vector = embedder.embed_query(query)
                    exact = [ids[i] for i in np.argsort(-(vectors @ query_vector))[:5]]
                    approximate = [h.chunk_id for h in store.search(query_vector, k=5)]
                    top1 += bool(approximate) and approximate[0] == exact[0]
                    overlap += len(set(approximate) & set(exact))
                probes = len(CONCEPTUAL) + len(IDENTIFIER)
                recall_at_5 = overlap / (5 * probes)
                results.append(
                    check(
                        "vectors are unit-normalised",
                        bool(np.allclose(norms, 1.0, atol=1e-4)),
                        f"norms {norms.min():.4f}-{norms.max():.4f}",
                    )
                )
                # An approximate index is not required to reproduce brute force exactly; it is
                # required not to lose the best answer. Top-1 must be exact, and recall@5 is
                # held to a threshold rather than to equality, since the reranker downstream
                # sees roughly 24 candidates anyway.
                results.append(
                    check(
                        "approximate search preserves the best answer",
                        top1 == probes and recall_at_5 >= 0.95,
                        f"top-1 {top1}/{probes} exact, recall@5 {recall_at_5:.3f}",
                    )
                )

        print("\nconceptual queries (no shared keywords - dense should win):")
        for query in CONCEPTUAL:
            dense = store.search(embedder.embed_query(query), k=1)
            sparse_hits = sparse.search(query, k=1)
            print(f"  {query!r}")
            if dense:
                print(f"      dense  {dense[0].score:.3f}  {dense[0].metadata.get('citation')}")
                print(f"             {dense[0].content[:90].strip()}...")
            print(
                f"      sparse {'no match' if not sparse_hits else f'{sparse_hits[0].score:.2f}'}"
            )

        print("\nidentifier queries (exact tokens - BM25 should win):")
        by_id = {c.chunk_id: c for c in chunks}
        for query in IDENTIFIER:
            dense = store.search(embedder.embed_query(query), k=1)
            sparse_hits = sparse.search(query, k=1)
            print(f"  {query!r}")
            if dense:
                found = query.split()[0].lower() in dense[0].content.lower()
                print(
                    f"      dense  {dense[0].score:.3f}  {dense[0].metadata.get('citation')}"
                    f"  {'contains the term' if found else 'TERM ABSENT'}"
                )
            if sparse_hits:
                chunk = by_id.get(sparse_hits[0].chunk_id)
                found = chunk is not None and query.split()[0].lower() in chunk.content.lower()
                print(
                    f"      sparse {sparse_hits[0].score:.2f}  "
                    f"{chunk.citation if chunk else '?'}"
                    f"  {'contains the term' if found else 'TERM ABSENT'}"
                )
            else:
                print("      sparse no match")

        print()
        return 0 if all(results) else 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
