"""Compare multiple LLM models on same document."""

from typing import Dict, List, Tuple
from .llm import Summarizer
from .evaluation import evaluate_summary, print_comparison_table


class ModelComparison:
    """Run and compare multiple models."""

    def __init__(self, models: List[str] = None):
        """Initialize with list of models to compare."""
        if models is None:
            models = [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct"
            ]

        self.models = models
        self.results = {}

    def compare_on_document(
        self,
        document_text: str,
        chunks: List[str],
        retrieval_queries: List[str] = None
    ) -> Dict[str, Dict]:
        """Compare all models on same document."""
            retrieval_queries: Queries for chunk retrieval

        Returns:
            Dictionary mapping model names to results (summary + scores)
        """
        if retrieval_queries is None:
            retrieval_queries = [
                "main purpose and overview of the report",
                "key findings and important statistics",
                "data governance and data usage",
                "challenges and risks",
                "recommendations and conclusions"
            ]

        results = {}

        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"Running model: {model_name}")
            print(f"{'='*60}")

            try:
                summary = self._summarize_with_model(
                    model_name,
                    chunks,
                    retrieval_queries
                )

                # Evaluate the summary
                scores = evaluate_summary(document_text, summary)

                results[model_name] = {
                    "summary": summary,
                    "scores": scores,
                    "status": "success"
                }

                print(f"\nSummary from {model_name}:")
                print("-" * 60)
                print(summary)
                print("-" * 60)

                print(f"\nScores for {model_name}:")
                for metric, score in scores.items():
                    print(f"  {metric}: {score}/10")

            except Exception as e:
                print(f"ERROR with {model_name}: {e}")
                results[model_name] = {
                    "summary": "",
                    "scores": {},
                    "status": "error",
                    "error": str(e)
                }

        self.results = results
        return results

    def _summarize_with_model(
        self,
        model_name: str,
        chunks: List[str],
        retrieval_queries: List[str]
    ) -> str:
        """Summarize document chunks using a specific model."""
        from .retriever import make_retriever

        # Retrieve relevant chunks
        try:
            retriever = make_retriever("build")
            retrieved_chunks = []
            seen = set()
            K_PER_QUERY = 4
            MAX_CHUNKS = 20

            for query in retrieval_queries:
                for chunk in retriever(query, K_PER_QUERY):
                    key = chunk[:300]
                    if key not in seen:
                        seen.add(key)
                        retrieved_chunks.append(chunk)
                    if len(retrieved_chunks) >= MAX_CHUNKS:
                        break
                if len(retrieved_chunks) >= MAX_CHUNKS:
                    break

            chunks_to_use = retrieved_chunks if retrieved_chunks else chunks[:MAX_CHUNKS]
        except Exception:
            print("Retriever failed; using first chunks")
            chunks_to_use = chunks[:20]

        print(f"Summarizing {len(chunks_to_use)} chunks...")

        # Initialize model
        summarizer = Summarizer(model_name=model_name)

        # MAP: Summarize chunks
        partial_summaries = []
        for i, chunk in enumerate(chunks_to_use):
            summary = summarizer.summarize_with_instruction(
                chunk,
                instruction="Summarize this section in 2-3 sentences. Focus on main ideas and facts.",
                max_length=120
            )
            partial_summaries.append(summary)

        # REDUCE: Combine summaries
        groups = []
        group_size = 10

        for i in range(0, len(partial_summaries), group_size):
            group = "\n\n".join(partial_summaries[i:i + group_size])
            group_summary = summarizer.summarize_with_instruction(
                group,
                instruction="Combine these summaries into one concise summary. Remove repetition.",
                max_length=200
            )
            groups.append(group_summary)

        combined = "\n\n".join(groups)

        # Generate final summary
        final_summary = summarizer.summarize_with_instruction(
            combined,
            instruction=(
                "Create the final comprehensive summary. "
                "Include purpose, findings, facts, challenges, and recommendations."
            ),
            max_length=350
        )

        return final_summary

    def get_best_model(self) -> Tuple[str, Dict, float]:
        """Get the model with the highest overall score.

        Returns:
            Tuple of (model_name, results_dict, overall_score)
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")

        best_model = None
        best_score = 0.0
        best_results = None

        for model_name, result in self.results.items():
            if result["status"] == "success" and result["scores"]:
                overall = sum(result["scores"].values()) / len(result["scores"])
                if overall > best_score:
                    best_score = overall
                    best_model = model_name
                    best_results = result

        return best_model, best_results, best_score

    def print_comparison_summary(self) -> None:
        """Print a detailed comparison summary."""
        if not self.results:
            print("No results to compare.")
            return

        print("\n" + "="*70)
        print("FINAL COMPARISON RESULTS")
        print("="*70)

        # Create comparison table data
        comparison_data = {}
        for model_name, result in self.results.items():
            if result["status"] == "success":
                comparison_data[model_name] = result["scores"]

        if comparison_data:
            print_comparison_table(comparison_data)

        # Find and print best model
        try:
            best_model, best_result, best_score = self.get_best_model()
            print(f"\n🏆 BEST MODEL: {best_model}")
            print(f"   Overall Score: {best_score:.1f}/10")
            print(f"\n   Summary:")
            print("   " + "\n   ".join(best_result["summary"].split("\n")))
        except ValueError:
            print("No successful results to determine best model.")

        # Print error summary
        errors = {m: r for m, r in self.results.items() if r["status"] == "error"}
        if errors:
            print(f"\n⚠️  Models with errors: {len(errors)}")
            for model_name, result in errors.items():
                print(f"   {model_name}: {result.get('error', 'Unknown error')}")

    def export_results(self, filepath: str) -> None:
        """Export comparison results to a JSON file."""
        import json

        export_data = {}
        for model_name, result in self.results.items():
            export_data[model_name] = {
                "summary": result["summary"],
                "scores": result["scores"],
                "status": result["status"]
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {filepath}")
