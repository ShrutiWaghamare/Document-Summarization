"""Multi-query retriever with coverage-based chunk selection."""

from typing import List
import os
import json


def load_chunks(build_dir: str = "build") -> List[str]:
    """Load chunks from build directory."""
    path = os.path.join(build_dir, "chunks.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"chunks.json not found in {build_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("chunks", [])


def load_index(build_dir: str = "build"):
    """Load FAISS index from build directory."""
    try:
        import faiss
    except Exception:
        faiss = None
    
    idx_path = os.path.join(build_dir, "index.faiss")
    if faiss and os.path.exists(idx_path):
        return faiss.read_index(idx_path)
    
    # fallback: try to build index from embeddings.npy
    emb_path = os.path.join(build_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        import numpy as np
        from .vectorstore import create_faiss_index
        vectors = np.load(emb_path)
        return create_faiss_index(vectors.tolist())
    
    raise FileNotFoundError("No FAISS index or embeddings found in build/")


def make_retriever(build_dir: str = "build"):
    """Create retriever function for a document."""
    chunks = load_chunks(build_dir)
    index = load_index(build_dir)

    from .embeddings import embed_texts
    from .vectorstore import search_index

    def retriever(query: str, k: int = 5) -> List[str]:
        q_vec = embed_texts([query])[0]
        ids, distances = search_index(index, q_vec, k=k)
        results = []
        for _id in ids:
            if 0 <= _id < len(chunks):
                results.append(chunks[_id])
        return results

    return retriever


def get_coverage_chunks(
    chunks: List[str],
    retriever,
    queries: List[str],
    total_k: int = 15
) -> List[str]:
    """Retrieve chunks from different document sections for better coverage."""

    Returns:
        List of representative chunks spread across the document
    \"\"\"
    if len(chunks) == 0:
        return []

    if len(chunks) <= 20:
        return list(chunks)

    num_groups = min(5, max(3, len(chunks) // 20))
    group_size = max(1, len(chunks) // num_groups)
    chunks_per_group = max(1, total_k // num_groups)

    all_retrieved = []
    seen = set()

    for query in queries:
        for chunk in retriever(query, k=5):
            key = chunk[:300]
            if key not in seen:
                seen.add(key)
                all_retrieved.append(chunk)

    grouped = {i: [] for i in range(num_groups)}
    for retrieved_chunk in all_retrieved[:total_k * 3]:
        for idx, chunk in enumerate(chunks):
            if chunk[:300] == retrieved_chunk[:300]:
                group_id = min(idx // group_size, num_groups - 1)
                if len(grouped[group_id]) < chunks_per_group:
                    grouped[group_id].append(chunk)
                break

    for group_id in range(num_groups):
        start = group_id * group_size
        if not grouped[group_id] and start < len(chunks):
            grouped[group_id].append(chunks[start])

    final_chunks = []
    for group_id in sorted(grouped):
        final_chunks.extend(grouped[group_id][:chunks_per_group])

    unique_chunks = []
    seen = set()
    for chunk in final_chunks:
        key = chunk[:300]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks[:total_k]

