"""Split text into chunks with metadata."""

from typing import List, Dict, Union


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    
    return chunks


def split_text_with_metadata(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    page: int = 1
) -> List[Dict[str, Union[str, int]]]:
    """Split text into chunks with page/chunk metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    chunk_id = 0
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text,
            "page": page,
            "chunk_id": chunk_id,
            "start_char": start,
            "end_char": end
        })
        
        if end == text_len:
            break
        
        start = end - overlap
        chunk_id += 1
    
    return chunks


def split_pages_with_metadata(
    pages: List[Dict[str, object]],
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, Union[str, int]]]:
    """Split multiple pages into chunks while preserving page metadata.

    Args:
        pages: List of page dicts with "page" and "text" keys
        chunk_size: Target chunk size
        overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries with page information
    """
    all_chunks = []
    global_chunk_id = 0
    
    for page_dict in pages:
        page_num = page_dict.get("page", 1)
        text = page_dict.get("text", "")
        
        page_chunks = split_text_with_metadata(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            page=page_num
        )
        
        # Update global chunk IDs
        for chunk in page_chunks:
            chunk["chunk_id"] = global_chunk_id
            global_chunk_id += 1
        
        all_chunks.extend(page_chunks)
    
    return all_chunks

