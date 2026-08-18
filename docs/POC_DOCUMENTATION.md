# Document Summarization POC Documentation

## 1. Objective

This project is a lightweight proof-of-concept for building a document summarization system using retrieval-augmented generation (RAG) with open-source LLMs. The goal is to summarize long PDF documents while keeping the solution runnable on a local laptop and usable for a fast MVP.

The original problem statement was:

> Design and develop a document summarization solution using RAG, LangChain, and LangGraph. Use open-source LLMs and perform a comparative analysis of multiple models to identify the best-performing model, along with justification for the selection.

---

## 2. What we actually built

### 2.1 Core pipeline

The implemented POC follows this flow:

1. PDF is loaded
2. Text is extracted from the document
3. Text is split into chunks
4. Chunks are converted into embeddings
5. Embeddings are stored in a local FAISS vector index
6. Relevant chunks are retrieved using query-based similarity search
7. A small open-source language model summarizes the selected chunks
8. Partial summaries are combined into a final document summary

This is the standard RAG design: retrieve relevant context first, then generate a response from that context.

### 2.2 Current implementation files

- [README.md](../README.md) — project overview and problem framing
- [src/main.py](../src/main.py) — command-line runner for extraction, embedding, and RAG stages
- [src/pdf_loader.py](../src/pdf_loader.py) — PDF text extraction logic
- [src/text_splitter.py](../src/text_splitter.py) — chunking logic for document sections
- [src/embeddings.py](../src/embeddings.py) — embedding generation logic
- [src/vectorstore.py](../src/vectorstore.py) — FAISS-based vector storage and search
- [src/retriever.py](../src/retriever.py) — retrieval abstraction for top-k chunk selection
- [src/llm.py](../src/llm.py) — Qwen-based summarization wrapper
- [src/workflow.py](../src/workflow.py) — minimal LangGraph workflow stub

---

## 3. RAG, LangChain, and LangGraph status

### 3.1 RAG

RAG is used in the project in a real way:

- chunks are embedded
- a FAISS index is used for retrieval
- relevant chunks are selected before summarization
- the LLM summarizes from selected content rather than the full document

This is the main technical contribution of the project.

### 3.2 LangChain

LangChain was added to the requirements as a dependency, but the actual codebase does not yet use a full LangChain chain or agent orchestration in the live flow. In other words, the project is not yet a full LangChain-native application.

Current status:

- dependency installed: yes
- active, production-style LangChain usage: not yet

This is still a good POC, but the LangChain layer is not central to the current implementation.

### 3.3 LangGraph

A minimal LangGraph example exists in [src/workflow.py](../src/workflow.py). It shows the idea of a graph-based summarization workflow, but it is not the main path used by the actual summarization pipeline.

Current status:

- graph stub present: yes
- full production workflow: not yet

So the architecture can be honestly summarized as:

- RAG: implemented and used
- LangChain: partially planned / dependency only
- LangGraph: minimal proof-of-concept stub

---

## 4. Technology stack and models used so far

### 4.1 Core tech stack used in this project

The project currently uses the following stack:

- Python as the primary implementation language
- PyPDF2 for PDF extraction and text loading
- sentence-transformers for embedding generation
- FAISS for local vector search
- Hugging Face transformers for model loading and generation
- Qwen instruction-tuned model for summarization
- LangChain as a planned / dependency-based integration layer
- LangGraph as a minimal workflow prototype

#### Summary of actual files and usage

- [src/pdf_loader.py](../src/pdf_loader.py): PDF extraction using PyPDF2
- [src/text_splitter.py](../src/text_splitter.py): chunking logic
- [src/embeddings.py](../src/embeddings.py): embedding generation with sentence-transformers
- [src/vectorstore.py](../src/vectorstore.py): FAISS index creation and search
- [src/retriever.py](../src/retriever.py): chunk retrieval wrapper
- [src/llm.py](../src/llm.py): summarization wrapper using Hugging Face models
- [src/workflow.py](../src/workflow.py): LangGraph workflow prototype

### 4.2 Models used so far

#### Model 1: BART-based summarization model

A BART-style summarization model was attempted earlier in the project as a quick baseline:

- example: sshleifer/distilbart-cnn-12-6

Why it was used:

- easy to run as a summarization baseline
- widely used for extractive and abstractive summarization
- lightweight compared to large LLMs

Why it was not kept as the final model:

- the summaries were too generic and weak for this specific document
- quality was lower than goal requirements
- the model was not tuned strongly enough for long contextual reports

#### Model 2: Qwen/Qwen2.5-1.5B-Instruct

This is the model currently used in the working CPU-friendly POC.

Why it was chosen:

- open-source and instruction-tuned
- runs reasonably well on CPU for a local prototype
- suitable for summarization with RAG
- more capable than a general summarization-only model for document instruction tasks

Trade-offs:

- still limited on very long documents
- slower than a smaller generative model in some cases
- may drift into metadata or boilerplate without a strong prompt

### 4.3 Key differences between model families

#### BART-style summarization models

Pros:

- fast
- good as baseline models
- relatively compact

Cons:

- weaker instruction following for diverse document content
- may produce generic summaries
- less robust for long-form reasoning

#### Qwen instruction models

Pros:

- stronger instruction following
- better for prompt-based summarization
- better generalization to report-style documents
- good open-source option for local experimentation

Cons:

- higher memory usage than BART-like models
- slower on CPU
- quality still depends on chunking, prompt design, and retrieval quality

### 4.4 Models that can still be used next

For future improvements, the following are the most practical options:

#### Option 1: Qwen/Qwen2.5-3B-Instruct

Good for:

- stronger summary quality
- more robust reasoning on relevant document chunks

Trade-offs:

- more memory and compute
- may still be heavy for a laptop without GPU

#### Option 2: Qwen/Qwen2.5-7B-Instruct

Good for:

- stronger general instruction-following
- better long-document understanding

Trade-offs:

- heavy on CPU
- usually needs a stronger machine or GPU for comfortable use

#### Option 3: Mistral-based models

Good for:

- strong open-source instruction performance
- good summarization quality in many benchmark settings

Trade-offs:

- often more resource-intensive than the 1.5B model
- not always better than a tuned Qwen model for this workload

#### Option 4: Phi-3 / Phi-3.5 models

Good for:

- compact and efficient models
- quick testing on limited hardware

Trade-offs:

- quality may be lower than larger models on demanding long-document tasks

#### Option 5: LLaMA-based instruct models

Good for:

- strong open-source support
- good benchmark performance

Trade-offs:

- resource-hungry depending on size
- may require more careful prompt tuning

### 4.5 Best practical model choices for a laptop POC

For a local hardware-constrained demo, the best practical path is:

1. keep Qwen/Qwen2.5-1.5B-Instruct for the current MVP
2. test Qwen/Qwen2.5-3B-Instruct for a stronger but still practical upgrade
3. if GPU is available, consider 7B-class models for higher-quality summaries
4. keep BART only as a baseline, not as the main final model

---

## 5. What happens when the PDF has fewer pages

When the PDF is short, the summarization problem is much easier.

### Recommended approach for small documents

- use all or most of the pages
- use a moderate chunk size (e.g., 800-1200 tokens)
- use a single pass summary from the full content
- use a stronger instruction prompt
- avoid retrieval if the document is small enough

### Why this works

For short documents:

- there is less need for retrieval
- the full document fits into the context window more easily
- summarization quality is usually better
- there is less risk of missing important sections

### Good practical rule

If the document is under 10-20 pages, a direct summarization pipeline usually works better than a retrieval-heavy one.

---

## 5. What happens when the PDF has many pages

When the document is long, the main problem is not just summarization quality, but computational cost and context dilution.

### Common approaches for large documents

#### Approach A: Retrieval-first summarization

- split document into chunks
- embed each chunk
- retrieve top-k relevant chunks for a task or query
- summarize only the retrieved content

This is the approach used in this project.

Advantages:

- faster than processing every page
- lower memory usage
- easy to implement
- works well for local POCs

Disadvantages:

- depends heavily on retrieval quality
- can miss cross-section context
- may omit important information outside the retrieved set

#### Approach B: Map-reduce summarization

- summarize each chunk individually
- then combine those chunk summaries into a final summary

This is also used in the current project.

Advantages:

- scalable for long documents
- reduces context overload
- works with moderate hardware

Disadvantages:

- may lose nuance between chunks
- repeated summarization can flatten important details
- quality can drop if chunk boundaries are poor

#### Approach C: Hierarchical summarization

- summarize sections first
- then summarize the section summaries into a final summary

This is a stronger version of map-reduce and often provides better document-level coherence.

Advantages:

- better structure
- better retention of major themes
- more natural final summary

Disadvantages:

- more expensive than simple retrieval
- more complex orchestration

#### Approach D: Query-focused summarization

- asks the user for the summary target: business summary, risks, recommendations, etc.
- retrieves only the sections relevant to that question

Advantages:

- higher quality for targeted summaries
- faster than general-purpose document summarization

Disadvantages:

- not ideal for full-document synthesis

#### Approach E: Reranking + retrieval

- retrieve a larger set of chunks
- rerank them with a stronger cross-encoder or semantic ranker
- keep the best chunks for final summarization

Advantages:

- better retrieval quality than plain vector search
- more accurate final summary

Disadvantages:

- more complexity
- requires reranker model or additional compute

#### Approach F: Long-context LLMs

- use a model with a larger context window
- feed in longer chunks or whole documents

Advantages:

- better global understanding
- fewer retrieval misses

Disadvantages:

- very expensive for CPU or local machines
- memory-intensive
- not always practical for POC or laptop projects

---

## 6. Practical recommendation for this laptop-based POC

For a local 1-hour MVP, the best realistic setup is:

- document chunking
- embedding + FAISS retrieval
- small set of topic-focused search queries
- 15-25 relevant chunks only
- map-reduce summarization
- strong prompt instructions

This is what the project already does in its latest form.

Why this is a good compromise:

- avoids summarizing all 136 chunks
- reduces CPU load
- keeps the flow representative of real RAG
- works on a typical laptop without CUDA
- demonstrates the architecture clearly

---

## 7. Evaluation strategy for the POC

To evaluate a summarization system in a principled way, use several criteria instead of relying on only one final text.

### 7.1 Core evaluation dimensions

#### Relevance

Does the summary include the most important information from the document?

#### Factual consistency

Are the claims in the summary supported by the source text?

#### Coverage

Does the summary include the major themes, issues, findings, and recommendations?

#### Conciseness

Is the summary brief without dropping key facts?

#### Fluency

Is the writing coherent and readable?

#### Hallucination rate

Does the model invent content or drift into unsupported statements?

#### Latency

How long does summarization take?

#### Resource usage

How much RAM and CPU does the pipeline consume?

### 7.2 Practical POC scoring rubric

Use a 1-to-5 score for each metric:

- 5 = excellent
- 4 = good
- 3 = acceptable
- 2 = weak
- 1 = poor

Suggested rubric:

- relevance: 25%
- factual consistency: 25%
- coverage: 20%
- conciseness: 15%
- fluency: 10%
- latency/resource fit: 5%

This is more realistic than simply looking at one text output and saying it is “good.”

### 7.3 Qualitative evaluation method

For each model or configuration, compare:

- model output
- document key facts
- whether the output omits metadata and focuses on findings
- whether it preserves the report’s main argument

This is especially useful when benchmarking open-source models on CPU.

---

## 8. How to do better next

### 8.1 Prompt engineering

Use tighter prompts such as:

- focus only on core findings
- ignore acknowledgements and publication metadata
- preserve major facts and statistics
- output 3-5 sentences only
- do not invent content

This alone can improve output quality significantly.

### 8.2 Better retrieval

Use:

- multiple queries instead of one
- topic-focused retrieval groups
- deduplication of chunks
- higher-K retrieval with reranking

This is a strong improvement path for long documents.

### 8.3 Use a stronger model

The current model is usable, but for better quality the next step is a stronger small model or a GPU-enabled environment.

Possible improvements:

- use a larger instruction-tuned model
- run on CUDA if available
- compare several open-source models

### 8.4 Fine-tuning

Fine-tuning is useful only if:

- you have a labeled summarization dataset
- the domain is narrow and repetitive
- the same style of documents appears repeatedly

For a generic PDF summarizer, prompt engineering and retrieval quality usually provide more value than fine-tuning in a short POC.

### 8.5 Parameter-efficient tuning

If fine-tuning is required, a better path is:

- LoRA
- QLoRA
- small domain-specific dataset
- low-rank adapters on an instruction model

This is more practical than full fine-tuning on a laptop.

---

## 9. Complexity and trade-offs

### Low complexity

- direct summary on short documents
- small chunk count
- single model

Pros:

- fast
- easy to debug
- works on weak hardware

Cons:

- weaker coverage for long documents
- more hallucination risk

### Medium complexity

- RAG + retrieval + map-reduce
- multiple topic queries
- top-k relevant chunk selection

Pros:

- better for long documents
- more realistic architecture
- good balance between performance and cost

Cons:

- more code and orchestration
- retrieval mistakes can hurt output

### High complexity

- reranking
- hierarchical summarization
- multiple models
- fine-tuning
- evaluation loops

Pros:

- best quality potential
- strong research-grade pipeline

Cons:

- more engineering effort
- higher compute cost
- slower iteration cycle

---

## 10. Best final POC story

The strongest honest summary of the current project is:

> This POC demonstrates a local, retrieval-augmented document summarization pipeline using open-source LLMs. It uses a FAISS vector store for chunk retrieval, a small instruction-tuned model for summarization, and a map-reduce flow to combine section summaries into a final summary. While LangChain and LangGraph are included as conceptual and scaffold components, the active pipeline is a practical RAG implementation designed to run on a laptop within a limited-time MVP.

That is accurate and defensible.

---

## 11. Recommendation for the next phase

If the goal is a stronger final project, the next best step is:

1. keep the RAG architecture
2. add a stronger model comparison
3. add evaluation scoring
4. add retrieval reranking
5. optionally integrate LangChain more cleanly
6. optionally add LangGraph as a workflow orchestrator

This gives a path from MVP to a more research-grade system without overshooting the project timeline.
