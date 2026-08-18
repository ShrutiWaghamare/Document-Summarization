"""Generate embeddings using sentence-transformers."""

from typing import List
import numpy as np
import os


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """Encode texts to embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=False)
    # Ensure list of lists (JSON-serializable-friendly)
    return [list(map(float, v)) for v in np.array(vectors)]


def save_embeddings(vectors: List[List[float]], build_dir: str) -> str:
    """Save embeddings to .npy file."""
    os.makedirs(build_dir, exist_ok=True)
    emb_path = os.path.join(build_dir, "embeddings.npy")
    np.save(emb_path, np.array(vectors, dtype=np.float32))
    return emb_path


def load_embeddings(build_dir: str) -> List[List[float]]:
    """Load embeddings from .npy file."""
    emb_path = os.path.join(build_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    
    vectors = np.load(emb_path)
    return [list(map(float, v)) for v in vectors]
