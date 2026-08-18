"""Evaluate summary quality (5 metrics: relevance, faithfulness, coverage, conciseness, coherence)."""

from typing import Dict
import re


def evaluate_summary(original_text: str, summary: str) -> Dict[str, float]:
    """Score summary across 5 metrics."""

    scores = {
        "relevance": evaluate_relevance(original_text, summary),
        "faithfulness": evaluate_faithfulness(original_text, summary),
        "coverage": evaluate_coverage(original_text, summary),
        "conciseness": evaluate_conciseness(original_text, summary),
        "coherence": evaluate_coherence(summary)
    }

    return scores


def evaluate_relevance(original_text: str, summary: str) -> float:
    """Score summary vocabulary overlap with original text."""
    original_words = set(extract_key_terms(original_text))
    summary_words = set(extract_key_terms(summary))

    if not summary_words:
        return 1.0

    # NOTE: fixed — previously divided by len(original_words), which measures
    # what fraction of the ENTIRE document's vocabulary shows up in the
    # summary. That's structurally near-zero for any real document (a short
    # summary can never contain most of a long document's unique words) and
    # is really a coverage question, not a relevance one — coverage is
    # already measured separately below via evaluate_coverage(). Relevance
    # should instead ask: of the words the summary actually uses, how many
    # are grounded in the source document? So we divide by len(summary_words).
    overlap = len(original_words & summary_words)
    relevance_ratio = overlap / len(summary_words)

    # Convert to 1-10 scale
    score = 1 + (relevance_ratio * 9)
    return round(min(score, 10.0), 1)


def evaluate_faithfulness(original_text: str, summary: str) -> float:
    """Score how faithful the summary is to the original document.

    Checks if the summary contains factual information present in the original
    and doesn't make up false claims.
    """
    # Simple heuristic: check if summary length is reasonable compared to original
    # and doesn't introduce new proper nouns
    original_nouns = extract_proper_nouns(original_text)
    summary_nouns = extract_proper_nouns(summary)

    # Check for introduced nouns (potential hallucinations)
    new_nouns = set(summary_nouns) - set(original_nouns)

    # Penalize too many new nouns
    penalty = min(len(new_nouns) * 0.5, 3.0)

    # Check summary length reasonableness
    length_ratio = len(summary) / max(len(original_text), 1)
    if length_ratio > 0.5:
        # Summary is too long for a proper summary
        penalty += 2.0

    score = 10.0 - penalty
    return round(max(score, 1.0), 1)


def evaluate_coverage(original_text: str, summary: str) -> float:
    """Score how well the summary covers major themes in the document.

    Checks if important sections of the original are represented.
    """
    # Split original into paragraphs and check representation
    original_paragraphs = [p.strip() for p in original_text.split('\n\n') if len(p.strip()) > 50]

    if not original_paragraphs:
        return 5.0

    # Check how many major themes are mentioned
    covered_themes = 0
    for para in original_paragraphs[:5]:  # Check first 5 major paragraphs
        theme_words = extract_key_terms(para)
        summary_words = extract_key_terms(summary)

        if any(word in summary_words for word in theme_words):
            covered_themes += 1

    coverage_ratio = covered_themes / min(5, len(original_paragraphs))
    score = 1 + (coverage_ratio * 9)

    return round(min(score, 10.0), 1)


def evaluate_conciseness(original_text: str, summary: str) -> float:
    """Score how concise the summary is.

    A good summary should be significantly shorter than the original
    while retaining key information.
    """
    if len(original_text) == 0:
        return 5.0

    compression_ratio = len(summary) / len(original_text)

    # Ideal compression ratio: 10-20% of original
    if 0.10 <= compression_ratio <= 0.20:
        score = 10.0
    elif 0.05 <= compression_ratio < 0.10:
        score = 8.0
    elif 0.20 < compression_ratio <= 0.30:
        score = 7.0
    elif compression_ratio < 0.05:
        score = 5.0  # Too short, might miss info
    else:
        # compression_ratio > 0.30
        score = 3.0  # Too long for a summary

    return round(score, 1)


def evaluate_coherence(summary: str) -> float:
    """Score the coherence and readability of the summary.

    Checks for sentence structure quality, logical flow, and readability.
    """
    if not summary:
        return 1.0

    # Check sentence count
    sentences = re.split(r'[.!?]+', summary)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 2.0

    # Ideal: 3-7 sentences for a summary
    sentence_count = len(sentences)
    if 3 <= sentence_count <= 7:
        sentence_score = 10.0
    elif 2 <= sentence_count < 3 or 7 < sentence_count <= 10:
        sentence_score = 7.0
    else:
        sentence_score = 4.0

    # Check average sentence length (should be moderate)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    if 10 <= avg_sentence_length <= 25:
        length_score = 10.0
    elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 40:
        length_score = 7.0
    else:
        length_score = 4.0

    # Check for flow indicators (conjunctions, transitions)
    flow_words = [
        "however", "therefore", "furthermore", "moreover", "in addition",
        "consequently", "thus", "meanwhile", "subsequently", "meanwhile"
    ]
    flow_score = 7.0 if any(word in summary.lower() for word in flow_words) else 5.0

    # Average the scores
    coherence_score = (sentence_score + length_score + flow_score) / 3
    return round(coherence_score, 1)


def extract_key_terms(text: str) -> list:
    """Extract key terms from text (simple approach)."""
    # Remove common stop words and extract significant words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "of", "to", "for",
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "do", "does", "did", "will", "would", "could", "should", "may", "might"
    }

    words = re.findall(r'\b\w+\b', text.lower())
    key_terms = [w for w in words if w not in stop_words and len(w) > 3]
    return key_terms


def extract_proper_nouns(text: str) -> list:
    """Extract proper nouns from text (simple capitalization-based approach)."""
    words = text.split()
    proper_nouns = [w.strip('.,!?;:') for w in words if w and w[0].isupper()]
    return proper_nouns


def print_evaluation_report(scores: Dict[str, float]) -> None:
    """Pretty-print an evaluation report."""
    print("\n" + "="*50)
    print("SUMMARY EVALUATION REPORT")
    print("="*50)

    for metric, score in scores.items():
        bar_length = int(score)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        print(f"{metric.capitalize():15} [{bar}] {score:5.1f}/10")

    overall = sum(scores.values()) / len(scores)
    print("-"*50)
    print(f"{'Overall':15} {overall:5.1f}/10")
    print("="*50 + "\n")


def compare_summaries(summaries: Dict[str, str], original_text: str) -> Dict[str, Dict[str, float]]:
    """Compare multiple summaries and return scores for each.

    Args:
        summaries: Dictionary mapping model names to their summaries
        original_text: The original document text

    Returns:
        Dictionary mapping model names to their evaluation scores
    """
    results = {}

    for model_name, summary in summaries.items():
        scores = evaluate_summary(original_text, summary)
        results[model_name] = scores

    return results


def print_comparison_table(comparison_results: Dict[str, Dict[str, float]]) -> None:
    """Print a comparison table of model scores."""
    if not comparison_results:
        print("No results to compare.")
        return

    metrics = list(next(iter(comparison_results.values())).keys())
    models = list(comparison_results.keys())

    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)

    # Header
    header = "Model".ljust(20)
    for metric in metrics:
        header += metric.capitalize().ljust(12)
    header += "OVERALL".ljust(10)
    print(header)
    print("-"*70)

    # Rows
    for model in models:
        row = model.ljust(20)
        scores = comparison_results[model]
        for metric in metrics:
            row += f"{scores[metric]:.1f}".ljust(12)

        overall = sum(scores.values()) / len(scores)
        row += f"{overall:.1f}".ljust(10)
        print(row)

    print("="*70 + "\n")