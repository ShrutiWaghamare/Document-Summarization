"""FAISS vector store wrapper."""

from typing import List, Tuple
import numpy as np


def create_faiss_index(embeddings: List[List[float]]):
    try:
        import faiss
    except Exception:
        raise RuntimeError("faiss is required for the vector store. Install faiss-cpu.")

    arr = np.array(embeddings).astype('float32')
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)
    return index


def search_index(index, query_vec: List[float], k: int = 5) -> Tuple[List[int], List[float]]:
    import numpy as np
    q = np.array([query_vec]).astype('float32')
    D, I = index.search(q, k)
    return I[0].tolist(), D[0].tolist()
