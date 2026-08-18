"""LangGraph-based summarization workflow."""

import math
from typing import TypedDict, List, Dict

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    StateGraph = None
    START = None
    END = None


class DocumentSummaryState(TypedDict):
    """Workflow state."""
    document_text: str
    chunks: List[str]
    retrieved_chunks: List[str]
    partial_summaries: List[str]
    final_summary: str
    model_name: str
    queries: List[str]
    target_chunks: int
    batch_size: int
    evaluation_scores: Dict[str, float]


class DocumentSummarizationWorkflow:
    """LangGraph workflow for summarization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed. Install with: pip install langgraph")
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.graph = None
        # Created fresh each run in load_document_node; holds the LangChain
        # splitter/embeddings/vectorstore/retriever for this document.
        self.pipeline = None

    def load_document_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Load the document and initialize the LangChain pipeline."""
        print(f"[WORKFLOW] Loading document... (text length: {len(state['document_text'])})")
        from .langchain_pipeline import LangChainSummarizationPipeline

        self.pipeline = LangChainSummarizationPipeline(
            pdf_text=state["document_text"],
            embedding_model=self.embedding_model,
            llm_model=state["model_name"],
        )
        return state

    def chunk_document_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Split document into chunks and build the FAISS vector store (LangChain)."""
        print("[WORKFLOW] Chunking document + building vector store (LangChain)...")
        num_chunks = self.pipeline.setup()
        state["chunks"] = [c.page_content for c in self.pipeline.chunks]
        print(f"[WORKFLOW] Created {num_chunks} chunks")
        return state

    def retrieve_chunks_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Retrieve relevant chunks using LangChain's FAISS retriever."""
        print(f"[WORKFLOW] Retrieving relevant chunks using {len(state['queries'])} queries (LangChain)...")

        target_chunks = max(1, int(state.get("target_chunks", 0) or 0))
        if target_chunks <= 0:
            target_chunks = min(len(state["chunks"]), 20)

        if target_chunks >= len(state["chunks"]):
            state["retrieved_chunks"] = list(state["chunks"])
            print(f"[WORKFLOW] Retrieved {len(state['retrieved_chunks'])} chunks for summarization")
            return state

        k_per_query = max(1, math.ceil(target_chunks / max(1, len(state["queries"]))))

        try:
            retrieved = self.pipeline.retrieve_relevant_chunks(state["queries"], k_per_query=k_per_query)
            if len(retrieved) > target_chunks:
                retrieved = retrieved[:target_chunks]
            state["retrieved_chunks"] = retrieved if retrieved else state["chunks"][:target_chunks]
        except Exception as e:
            print(f"[WORKFLOW] Retriever failed: {e}. Using first chunks.")
            state["retrieved_chunks"] = state["chunks"][:target_chunks]

        print(f"[WORKFLOW] Retrieved {len(state['retrieved_chunks'])} chunks for summarization")
        return state

    def summarize_chunks_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Summarize chunks in adaptive batches."""
        batch_size = state.get("batch_size", 4)
        print(f"[WORKFLOW] Generating summaries for {len(state['retrieved_chunks'])} chunks (batch size: {batch_size})...")

        partial_summaries = []
        
        for i in range(0, len(state["retrieved_chunks"]), batch_size):
            batch = state["retrieved_chunks"][i:i + batch_size]
            combined_text = "\n---\n".join(batch)
            summary = self.pipeline.summarize_chunk(combined_text)
            partial_summaries.append(summary)
            print(f"[WORKFLOW] Processed batch {len(partial_summaries)} ({len(batch)} chunks)")

        state["partial_summaries"] = partial_summaries
        return state

    def combine_summaries_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Combine partial summaries into the final summary via LangChain."""
        print(f"[WORKFLOW] Combining {len(state['partial_summaries'])} partial summaries...")

        groups = []
        group_size = 10
        for i in range(0, len(state["partial_summaries"]), group_size):
            group = state["partial_summaries"][i:i + group_size]
            group_summary = self.pipeline.combine_summaries(group)
            groups.append(group_summary)

        final_summary = self.pipeline.generate_final_summary(groups)
        state["final_summary"] = final_summary
        print("[WORKFLOW] Generated final summary")
        return state

    def evaluate_summary_node(self, state: DocumentSummaryState) -> DocumentSummaryState:
        """Evaluate the quality of the generated summary."""
        try:
            from .evaluation import evaluate_summary
            print("[WORKFLOW] Evaluating summary quality...")
            scores = evaluate_summary(state["document_text"], state["final_summary"])

            print("[WORKFLOW] Evaluation scores:")
            for metric, score in scores.items():
                print(f"  {metric}: {score}/10")

            state["evaluation_scores"] = scores
        except Exception as e:
            print(f"[WORKFLOW] Evaluation skipped: {e}")
            state["evaluation_scores"] = {}

        return state

    def build_graph(self):
        """Build the LangGraph workflow."""
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed")

        graph = StateGraph(DocumentSummaryState)

        graph.add_node("load_document", self.load_document_node)
        graph.add_node("chunk_document", self.chunk_document_node)
        graph.add_node("retrieve_chunks", self.retrieve_chunks_node)
        graph.add_node("summarize_chunks", self.summarize_chunks_node)
        graph.add_node("combine_summaries", self.combine_summaries_node)
        graph.add_node("evaluate_summary", self.evaluate_summary_node)

        graph.add_edge(START, "load_document")
        graph.add_edge("load_document", "chunk_document")
        graph.add_edge("chunk_document", "retrieve_chunks")
        graph.add_edge("retrieve_chunks", "summarize_chunks")
        graph.add_edge("summarize_chunks", "combine_summaries")
        graph.add_edge("combine_summaries", "evaluate_summary")
        graph.add_edge("evaluate_summary", END)

        self.graph = graph.compile()
        return self.graph

    def run(self, document_text: str, queries: List[str] = None, target_chunks: int = None, batch_size: int = 4):
        """Run the complete workflow."""
        if self.graph is None:
            self.build_graph()

        if queries is None:
            queries = [
                "main purpose and overview of the report",
                "key findings and important statistics",
                "data governance and data usage",
                "challenges and risks",
                "recommendations and conclusions"
            ]

        initial_state = DocumentSummaryState(
            document_text=document_text,
            chunks=[],
            retrieved_chunks=[],
            partial_summaries=[],
            final_summary="",
            model_name=self.model_name,
            queries=queries,
            target_chunks=target_chunks or 0,
            batch_size=batch_size,
            evaluation_scores={}
        )

        print("[WORKFLOW] Starting document summarization workflow...")
        result = self.graph.invoke(initial_state)
        print("[WORKFLOW] Workflow completed.")

        return result