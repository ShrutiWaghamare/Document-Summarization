"""LangChain RAG components (embeddings, FAISS, retriever, LLM wrapper)."""

from typing import List, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun


class QwenLLMWrapper(LLM):
    """Qwen model wrapper for LangChain."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Qwen model through the Summarizer wrapper."""
        if not hasattr(self, "_summarizer") or self._summarizer is None:
            from .llm import Summarizer
            object.__setattr__(self, "_summarizer", Summarizer(model_name=self.model_name))
        return self._summarizer.summarize_with_instruction(
            prompt,
            instruction="Summarize this content.",
            max_length=kwargs.get("max_length", 300)
        )


class LangChainSummarizationPipeline:
    """LangChain-based RAG summarization pipeline."""

    def __init__(
        self,
        pdf_text: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.pdf_text = pdf_text
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = QwenLLMWrapper(model_name=llm_model)

        self.vectorstore = None
        self.retriever = None
        self.documents = []
        self.chunks = []

        # Load the summarization model ONCE and reuse it for every chunk,
        # group, and the final summary — instantiating a fresh Summarizer()
        # per call reloads the whole model from disk each time, which is slow
        # and can exhaust memory over many chunks.
        self._summarizer = None

    def _get_summarizer(self):
        if self._summarizer is None:
            from .llm import Summarizer
            self._summarizer = Summarizer(model_name=self.llm_model)
        return self._summarizer

    def setup(self):
        """Set up the pipeline by creating documents and vector store."""
        self.documents = [Document(page_content=self.pdf_text)]
        self.chunks = self.text_splitter.split_documents(self.documents)

        if self.chunks:
            self.vectorstore = FAISS.from_documents(
                self.chunks,
                self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        return len(self.chunks)

    def retrieve_relevant_chunks(self, queries: List[str], k_per_query: int = 4) -> List[str]:
        """Retrieve relevant chunks using multiple queries."""
        if not self.retriever:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        retrieved_chunks = []
        seen = set()

        for query in queries:
            docs = self.retriever.invoke(query)
            for doc in docs[:k_per_query]:
                chunk_key = doc.page_content[:300]
                if chunk_key not in seen:
                    seen.add(chunk_key)
                    retrieved_chunks.append(doc.page_content)

        return retrieved_chunks

    def summarize_chunk(self, chunk: str, instruction: str = None) -> str:
        """Summarize a single chunk."""
        summarizer = self._get_summarizer()
        return summarizer.summarize_with_instruction(
            chunk,
            instruction=instruction or "Summarize this section in 2-3 sentences.",
            max_length=120
        )

    def combine_summaries(self, summaries: List[str]) -> str:
        """Combine multiple chunk summaries into a cohesive summary."""
        summarizer = self._get_summarizer()

        combined_text = "\n\n".join(summaries)
        return summarizer.summarize_with_instruction(
            combined_text,
            instruction="Combine these section summaries into one cohesive summary. Remove repetition and preserve key information.",
            max_length=200
        )

    def generate_final_summary(self, intermediate_summaries: List[str]) -> str:
        """Generate the final document summary from intermediate summaries."""
        summarizer = self._get_summarizer()

        combined = "\n\n".join(intermediate_summaries)
        return summarizer.summarize_with_instruction(
            combined,
            instruction=(
                "Create the final comprehensive summary of the entire document. "
                "Include the main purpose, major findings, important facts, "
                "key challenges, and recommendations."
            ),
            max_length=350
        )

    def run_rag_pipeline(self, queries: List[str]) -> str:
        """Run the complete RAG pipeline standalone (without LangGraph)."""
        chunks_to_summarize = self.retrieve_relevant_chunks(queries, k_per_query=4)

        partial_summaries = []
        for chunk in chunks_to_summarize:
            partial_summaries.append(self.summarize_chunk(chunk))

        group_size = 10
        intermediate_summaries = []
        for i in range(0, len(partial_summaries), group_size):
            group = partial_summaries[i:i + group_size]
            intermediate_summaries.append(self.combine_summaries(group))

        return self.generate_final_summary(intermediate_summaries)