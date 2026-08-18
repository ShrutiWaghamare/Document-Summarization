"""CLI entrypoint for document summarization (RAG, LangGraph, evaluation)."""

import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser(description="Document summarization POC with RAG, LangChain, and LangGraph")
    parser.add_argument("--stage", choices=["extract", "chunk", "embed", "rag", "all"], default="rag",
                       help="Pipeline stage to run")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf", help="Path to PDF file for local testing")
    input_group.add_argument("--docx", help="Path to DOCX file for local testing")

    parser.add_argument("--model", help="Hugging Face model name (default: Qwen 1.5B)",
                       default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--use-langgraph", action="store_true",
                       help="Use LangGraph workflow instead of direct pipeline")
    parser.add_argument("--compare", action="store_true",
                       help="Run model comparison (Qwen 1.5B vs 3B)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation metrics on generated summary")
    args = parser.parse_args()
    # NOTE: --use-langchain was removed. LangChain (splitter, FAISS, embeddings)
    # now runs automatically inside the LangGraph nodes when --use-langgraph is
    # passed, so there's no separate flag needed for it anymore.

    input_path = args.pdf or args.docx
    if not input_path or not os.path.exists(input_path):
        print("Provide a valid --pdf or --docx path (e.g. data/sample.pdf or data/sample.docx)")
        return

    if args.stage in ("extract", "all"):
        if args.pdf:
            from .pdf_loader import load_pdf_text
            text = load_pdf_text(input_path)
        else:
            from .docx_loader import load_docx_text
            text = load_docx_text(input_path)
        print("Extracted text length:", len(text))

    if args.stage in ("chunk", "all"):
        if args.pdf:
            from .pdf_loader import load_pdf_text
            text = load_pdf_text(input_path)
        else:
            from .docx_loader import load_docx_text
            text = load_docx_text(input_path)
        from .text_splitter import split_text
        chunks = split_text(text)
        print("Created", len(chunks), "chunks. Example chunk:\n")
        print(chunks[0][:500])

    # NOTE: fixed - "embed" now also runs as part of "all", and always uses the
    # same per-document build directory as the "rag" stage so artifacts are
    # reusable between stages instead of writing to a shared, unscoped "build/" folder.
    if args.stage in ("embed", "all"):
        if args.pdf:
            from .pdf_loader import load_pdf_text
            text = load_pdf_text(input_path)
        else:
            from .docx_loader import load_docx_text
            text = load_docx_text(input_path)
        from .text_splitter import split_text
        from .embeddings import embed_texts, save_embeddings
        from .vectorstore import create_faiss_index
        from .doc_build_manager import ensure_build_dir, save_metadata
        import numpy as np

        chunks = split_text(text)
        print(f"Creating embeddings for {len(chunks)} chunks...")
        vectors = embed_texts(chunks)

        doc_build_dir = ensure_build_dir(input_path)

        # save embeddings (also written as .npy for inspection)
        save_embeddings(vectors, doc_build_dir)
        emb_path = os.path.join(doc_build_dir, "embeddings.npy")
        np.save(emb_path, np.array(vectors, dtype=np.float32))

        # create faiss index
        try:
            index = create_faiss_index(vectors)
            try:
                import faiss
                faiss_write_path = os.path.join(doc_build_dir, "index.faiss")
                faiss.write_index(index, faiss_write_path)
                print("FAISS index written to:", faiss_write_path)
            except Exception:
                print("FAISS is not available to persist index; index built in-memory.")
        except Exception as e:
            print("Failed to create FAISS index:", e)

        # save chunks metadata
        chunks_path = os.path.join(doc_build_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as fh:
            json.dump({"source_file": input_path, "chunks": chunks}, fh, ensure_ascii=False)

        save_metadata(doc_build_dir, input_path)
        print(f"Saved chunks and embeddings to {doc_build_dir} (embeddings.npy, chunks.json)")

    if args.stage in ("rag", "all"):
        if args.pdf:
            from .pdf_loader import load_pdf_text
            text = load_pdf_text(input_path)
        else:
            from .docx_loader import load_docx_text
            text = load_docx_text(input_path)
        from .text_splitter import split_text
        from .doc_build_manager import (
            ensure_build_dir,
            can_reuse_artifacts,
            save_metadata,
        )

        print(f"Loading {os.path.splitext(os.path.basename(input_path))[1].lower().replace('.', '').upper() or 'DOCUMENT'} file...")
        print("Splitting document...")
        chunks = split_text(text)
        print(f"Total chunks: {len(chunks)}")

        # Get document-specific build directory
        doc_build_dir = ensure_build_dir(input_path)
        print(f"Using build directory: {doc_build_dir}")

        # Check if we can reuse artifacts
        should_reembed = not can_reuse_artifacts(doc_build_dir, input_path)

        # Create/update embeddings and FAISS index if needed
        if should_reembed:
            print("Creating embeddings and FAISS index...")
            from .embeddings import embed_texts, save_embeddings
            from .vectorstore import create_faiss_index

            vectors = embed_texts(chunks)
            save_embeddings(vectors, doc_build_dir)

            # Save FAISS index
            try:
                index = create_faiss_index(vectors)
                try:
                    import faiss
                    index_path = os.path.join(doc_build_dir, "index.faiss")
                    faiss.write_index(index, index_path)
                    print(f"FAISS index written to: {index_path}")
                except Exception:
                    print("FAISS write unavailable; index built in-memory.")
            except Exception as e:
                print(f"Failed to create FAISS index: {e}")

            # Save chunks metadata
            chunks_path = os.path.join(doc_build_dir, "chunks.json")
            with open(chunks_path, "w", encoding="utf-8") as fh:
                json.dump({"source_file": input_path, "chunks": chunks}, fh, ensure_ascii=False)

            # Save metadata
            save_metadata(doc_build_dir, input_path)
            print(f"Saved artifacts to {doc_build_dir}")
        else:
            print(f"Reusing existing artifacts from {doc_build_dir}")

        # Define retrieval queries
        queries = [
            "main purpose and overview of the report",
            "key findings and important facts",
            "data governance and data use",
            "challenges risks and gaps",
            "recommendations and conclusions"
        ]

        # Adaptive K based on document size. For small documents, summarize all chunks
        # rather than forcing aggressive retrieval that misses sections.
        total_chunks = len(chunks)
        if total_chunks <= 20:
            adaptive_k = total_chunks
            batch_size = min(4, max(1, total_chunks))
        elif total_chunks <= 50:
            adaptive_k = 10
            batch_size = 3
        elif total_chunks <= 100:
            adaptive_k = 12
            batch_size = 4
        elif total_chunks <= 200:
            adaptive_k = 15
            batch_size = 5
        else:
            adaptive_k = 15
            batch_size = 6

        adaptive_k = min(adaptive_k, total_chunks)
        batch_size = max(1, min(batch_size, total_chunks))

        print(f"Adaptive retrieval: {adaptive_k} chunks, batch size: {batch_size}")

        # Model comparison mode
        if args.compare:
            from .model_comparison import ModelComparison

            print("\n" + "="*70)
            print("RUNNING MODEL COMPARISON")
            print("="*70)

            comparator = ModelComparison(models=[
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct"
            ])

            results = comparator.compare_on_document(text, chunks, queries)
            comparator.print_comparison_summary()

            # Export results
            results_file = os.path.join(doc_build_dir, "comparison_results.json")
            comparator.export_results(results_file)
            print(f"\nResults saved to {results_file}")

        # LangGraph workflow mode
        elif args.use_langgraph:
            from .workflow import DocumentSummarizationWorkflow

            print("\n" + "="*70)
            print("RUNNING LANGGRAPH WORKFLOW")
            print("="*70)

            workflow = DocumentSummarizationWorkflow(model_name=args.model)
            result = workflow.run(text, queries, target_chunks=adaptive_k, batch_size=batch_size)

            print("\n" + "="*70)
            print("FINAL SUMMARY (from LangGraph workflow)")
            print("="*70)
            print(result.get("final_summary", "No summary generated."))

            if "evaluation_scores" in result and result["evaluation_scores"]:
                from .evaluation import print_evaluation_report
                print_evaluation_report(result["evaluation_scores"])

        # Standard RAG mode with adaptive retrieval and batching
        else:
            from .llm import Summarizer
            from .retriever import make_retriever, get_coverage_chunks

            print(f"\nRetrieving {adaptive_k} chunks with coverage strategy...")

            try:
                retriever = make_retriever(doc_build_dir)
                if total_chunks <= 20:
                    chunks_to_summarize = list(chunks)
                else:
                    chunks_to_summarize = get_coverage_chunks(chunks, retriever, queries, total_k=adaptive_k)
                print(f"Retrieved {len(chunks_to_summarize)} chunks")
            except Exception as e:
                print(f"Retriever failed: {e}. Using first {adaptive_k} chunks.")
                chunks_to_summarize = chunks[:adaptive_k]

            summarizer = Summarizer(model_name=args.model)

            # MAP STEP - Batch chunk summarization
            print(f"\nSummarizing {len(chunks_to_summarize)} chunks in {batch_size}-chunk batches...")
            partial_summaries = []

            for i, chunk in enumerate(chunks_to_summarize):
                summary = summarizer.summarize_with_instruction(
                    chunk,
                    instruction="Summarize this section in 2-3 sentences. Focus on main ideas, facts, and conclusions.",
                    max_length=120
                )
                partial_summaries.append(summary)
                if (i + 1) % batch_size == 0 or (i + 1) == len(chunks_to_summarize):
                    print(f"  Processed {i + 1}/{len(chunks_to_summarize)} chunks")

            # REDUCE STEP - Hierarchical reduction
            print(f"\nReducing {len(partial_summaries)} summaries...")
            groups = []

            for i in range(0, len(partial_summaries), batch_size):
                group = "\n\n".join(partial_summaries[i:i + batch_size])
                group_summary = summarizer.summarize_with_instruction(
                    group,
                    instruction="Combine these summaries into one concise summary. Remove repetition and preserve key information.",
                    max_length=200
                )
                groups.append(group_summary)
                print(f"  Group {len(groups)}: Combined {len(partial_summaries[i:i + batch_size])} summaries")

            # FINAL STEP - Generate comprehensive summary
            print("\nGenerating final summary...")
            combined_groups = "\n\n".join(groups)
            final = summarizer.summarize_final(combined_groups, max_length=400)

            print("\n" + "="*70)
            print("GENERATED DOCUMENT SUMMARY")
            print("="*70 + "\n")
            print(final)

            # Evaluation mode
            if args.evaluate:
                from .evaluation import evaluate_summary, print_evaluation_report

                print("\n" + "="*70)
                print("SUMMARY EVALUATION")
                print("="*70)

                scores = evaluate_summary(text, final)
                print_evaluation_report(scores)


if __name__ == "__main__":
    main()