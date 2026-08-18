"""Manage document-specific build directories and cached artifacts."""

import os
import json
import re
from typing import Dict, Optional


def get_document_name(pdf_path: str) -> str:
    """Sanitize filename to safe folder name."""
    basename = os.path.basename(pdf_path)
    # Remove .pdf extension
    name = os.path.splitext(basename)[0]
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def get_document_build_dir(pdf_path: str, base_build_dir: str = "build") -> str:
    """Get document-specific build directory (e.g., Data For Better Lives.pdf -> build/Data_For_Better_Lives)."""
    doc_name = get_document_name(pdf_path)
    build_dir = os.path.join(base_build_dir, doc_name)
    return build_dir


def ensure_build_dir(pdf_path: str, base_build_dir: str = "build") -> str:
    """Ensure document-specific build directory exists and return its path.
    
    Args:
        pdf_path: Path to the PDF file
        base_build_dir: Base build directory (default: "build")
    
    Returns:
        Path to document-specific build directory (created if needed)
    """
    build_dir = get_document_build_dir(pdf_path, base_build_dir)
    os.makedirs(build_dir, exist_ok=True)
    return build_dir


def get_metadata_path(build_dir: str) -> str:
    """Get the metadata.json file path for a build directory."""
    return os.path.join(build_dir, "metadata.json")


def save_metadata(
    build_dir: str,
    pdf_path: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> None:
    """Save metadata about the document and processing configuration.
    
    Args:
        build_dir: Build directory path
        pdf_path: Path to the original PDF
        chunk_size: Chunk size used
        overlap: Overlap between chunks
        embedding_model: Embedding model name
    """
    metadata = {
        "source_pdf": os.path.basename(pdf_path),
        "source_pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model
    }
    
    metadata_path = get_metadata_path(build_dir)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(build_dir: str) -> Optional[Dict]:
    """Load metadata about the document and processing configuration.
    
    Args:
        build_dir: Build directory path
    
    Returns:
        Metadata dictionary if exists, None otherwise
    """
    metadata_path = get_metadata_path(build_dir)
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def artifacts_exist(build_dir: str) -> bool:
    """Check if all required build artifacts exist for a document.
    
    Args:
        build_dir: Build directory path
    
    Returns:
        True if chunks.json, embeddings.npy, and index.faiss all exist
    """
    chunks_path = os.path.join(build_dir, "chunks.json")
    embeddings_path = os.path.join(build_dir, "embeddings.npy")
    index_path = os.path.join(build_dir, "index.faiss")
    
    return (
        os.path.exists(chunks_path) and
        os.path.exists(embeddings_path) and
        os.path.exists(index_path)
    )


def can_reuse_artifacts(
    build_dir: str,
    pdf_path: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> bool:
    """Check whether existing artifacts can be reused.
    
    Args:
        build_dir: Build directory path
        pdf_path: Current PDF path
        chunk_size: Current chunk size
        overlap: Current overlap
        embedding_model: Current embedding model
    
    Returns:
        True if artifacts exist and configuration matches
    """
    if not artifacts_exist(build_dir):
        return False
    
    metadata = load_metadata(build_dir)
    if metadata is None:
        return False
    
    # Check if configuration matches
    return (
        metadata.get("chunk_size") == chunk_size and
        metadata.get("overlap") == overlap and
        metadata.get("embedding_model") == embedding_model and
        os.path.basename(metadata.get("source_pdf", "")) == os.path.basename(pdf_path)
    )
