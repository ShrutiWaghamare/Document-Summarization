# Document Summarization 

RAG-based document summarization using Qwen LLM, FAISS, and LangGraph. Reduces API calls by **96%** through intelligent batching.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Usage

**Small document (DOCX):**
```bash
python -m src.main --stage rag --docx "AI_in_Healthcare_5_Page_Test_Document.docx" --use-langgraph
```

**Large document (PDF):**
```bash
python -m src.main --stage rag --pdf "Data For Better Lives.pdf" --use-langgraph
```

**With model comparison:**
```bash
python -m src.main --stage rag --pdf "document.pdf" --compare
```

## CLI Options

```
--stage {extract|chunk|embed|rag|all}  Pipeline stage (default: rag)
--pdf <path>                            PDF file (mutually exclusive)
--docx <path>                           DOCX file (mutually exclusive)
--model <name>                          LLM model (default: Qwen2.5-1.5B)
--use-langgraph                         Enable LangGraph workflow
--compare                               Compare multiple models
--evaluate                              Run evaluation metrics
```

## Performance

| Metric | Small (DOCX) | Large (PDF) |
|--------|------------|-----------|
| Pages | 5 | 25 |
| Chunks | 14 | 136 |
| LLM Calls | 9 | 5 |
| Time | 25s | 50s |
| Quality | 8.6/10 | 8.1/10 |

## Architecture

```
Input → Extract → Chunk → Embed → Index → Retrieve → Batch → Summarize → Reduce → Final → Evaluate → Output
```

**6-Node LangGraph DAG:**
1. load_document → Initialize pipeline
2. chunk_document → Split text, build FAISS index
3. retrieve_chunks → Multi-query retrieval
4. summarize_chunks → Batch processing (adaptive)
5. combine_summaries → Hierarchical combination
6. evaluate_summary → Quality scoring

## Project Structure

```
src/
├── main.py                   CLI orchestration
├── pdf_loader.py            PDF extraction
├── docx_loader.py           DOCX extraction
├── text_splitter.py         Chunking
├── embeddings.py            Vectorization
├── vectorstore.py           FAISS indexing
├── retriever.py             Multi-query retrieval
├── llm.py                   Qwen summarization
├── langchain_pipeline.py    LangChain RAG
├── workflow.py              LangGraph workflow
├── doc_build_manager.py     Build artifact management
├── evaluation.py            Quality scoring (5 metrics)
└── model_comparison.py      Multi-model benchmarking
```

## Requirements

- Python 3.8+
- 8GB RAM minimum
- 15GB disk (includes Qwen model)
- GPU optional (CUDA 11.8+ for 2x speedup)

## Tech Stack

- **LLM:** Qwen 2.5-1.5B-Instruct (open-source)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS
- **Orchestration:** LangGraph
- **Framework:** LangChain
- **Input:** PyPDF2, python-docx

## Output

Summary includes:
- Plain text summary (no markdown)
- 5-metric quality scores (Relevance, Faithfulness, Coverage, Conciseness, Coherence)
- Execution statistics



