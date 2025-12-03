"""
evaluate.py - Evaluation metrics for topic models

This module provides functions to evaluate topic model quality using
coherence scores, diversity metrics, and clustering metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import warnings


def compute_topic_diversity(
    topic_model,
    top_n: int = 10
) -> float:
    """
    Compute topic diversity (proportion of unique words across topics).
    
    Higher diversity indicates topics are more distinct from each other.
    
    Args:
        topic_model: Fitted BERTopic model
        top_n: Number of top words per topic to consider
    
    Returns:
        Diversity score between 0 and 1
    """
    topics = topic_model.get_topics()
    
    # Exclude outlier topic (-1)
    if -1 in topics:
        del topics[-1]
    
    all_words = []
    for topic_id, words in topics.items():
        topic_words = [word for word, _ in words[:top_n]]
        all_words.extend(topic_words)
    
    unique_words = set(all_words)
    diversity = len(unique_words) / len(all_words) if all_words else 0
    
    return diversity


def compute_coherence_scores(
    topic_model,
    documents: List[str],
    coherence_type: str = "c_npmi",
    top_n: int = 10
) -> Dict[str, float]:
    """
    Compute topic coherence using gensim.
    
    Args:
        topic_model: Fitted BERTopic model
        documents: List of document strings
        coherence_type: Type of coherence ("c_npmi", "c_v", "u_mass")
        top_n: Number of top words per topic
    
    Returns:
        Dictionary with overall and per-topic coherence
    """
    # Get topic words
    topics = topic_model.get_topics()
    
    # Exclude outlier topic
    if -1 in topics:
        del topics[-1]
    
    # Extract top words for each topic
    topic_words = []
    for topic_id in sorted(topics.keys()):
        words = [word for word, _ in topics[topic_id][:top_n]]
        topic_words.append(words)
    
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Create dictionary
    dictionary = Dictionary(tokenized_docs)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    
    # Compute coherence
    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        overall_coherence = coherence_model.get_coherence()
        per_topic_coherence = coherence_model.get_coherence_per_topic()
        
        return {
            "overall": overall_coherence,
            "per_topic": dict(enumerate(per_topic_coherence)),
            "coherence_type": coherence_type
        }
    except Exception as e:
        warnings.warn(f"Could not compute coherence: {e}")
        return {"overall": None, "per_topic": {}, "coherence_type": coherence_type}


def compute_silhouette(
    embeddings: np.ndarray,
    topics: List[int]
) -> float:
    """
    Compute silhouette score for clustering quality.
    
    Args:
        embeddings: Document embeddings
        topics: Topic assignments
    
    Returns:
        Silhouette score between -1 and 1
    """
    # Remove outliers (-1) for silhouette computation
    mask = np.array(topics) != -1
    if mask.sum() < 2:
        return 0.0
    
    filtered_embeddings = embeddings[mask]
    filtered_topics = np.array(topics)[mask]
    
    # Need at least 2 clusters
    if len(set(filtered_topics)) < 2:
        return 0.0
    
    return silhouette_score(filtered_embeddings, filtered_topics)


def compute_topic_sizes(topics: List[int]) -> Dict[int, int]:
    """
    Compute the size of each topic.
    
    Args:
        topics: List of topic assignments
    
    Returns:
        Dictionary mapping topic_id to document count
    """
    return dict(Counter(topics))


def compute_outlier_ratio(topics: List[int]) -> float:
    """
    Compute the ratio of documents assigned to outlier topic.
    
    Args:
        topics: List of topic assignments
    
    Returns:
        Outlier ratio between 0 and 1
    """
    n_outliers = sum(1 for t in topics if t == -1)
    return n_outliers / len(topics)


def evaluate_topic_model(
    topic_model,
    documents: List[str],
    embeddings: np.ndarray,
    topics: List[int],
    top_n_words: int = 10
) -> Dict:
    """
    Comprehensive evaluation of a topic model.
    
    Args:
        topic_model: Fitted BERTopic model
        documents: List of document strings
        embeddings: Document embeddings
        topics: Topic assignments
        top_n_words: Number of top words for coherence
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("Computing evaluation metrics...")
    
    results = {}
    
    # Topic diversity
    results["diversity"] = compute_topic_diversity(topic_model, top_n=top_n_words)
    print(f"  Topic Diversity: {results['diversity']:.4f}")
    
    # Coherence (NPMI)
    coherence = compute_coherence_scores(
        topic_model, documents, 
        coherence_type="c_npmi", 
        top_n=top_n_words
    )
    results["coherence_npmi"] = coherence["overall"]
    results["coherence_per_topic"] = coherence["per_topic"]
    if coherence["overall"]:
        print(f"  Coherence (NPMI): {coherence['overall']:.4f}")
    
    # Silhouette score
    results["silhouette"] = compute_silhouette(embeddings, topics)
    print(f"  Silhouette Score: {results['silhouette']:.4f}")
    
    # Topic sizes
    results["topic_sizes"] = compute_topic_sizes(topics)
    results["n_topics"] = len([t for t in set(topics) if t != -1])
    print(f"  Number of Topics: {results['n_topics']}")
    
    # Outlier ratio
    results["outlier_ratio"] = compute_outlier_ratio(topics)
    print(f"  Outlier Ratio: {results['outlier_ratio']:.2%}")
    
    return results


def compare_models(
    results_list: List[Dict],
    model_names: List[str]
) -> pd.DataFrame:
    """
    Compare multiple topic models.
    
    Args:
        results_list: List of evaluation result dictionaries
        model_names: Names for each model
    
    Returns:
        DataFrame comparing models
    """
    comparison = []
    for name, results in zip(model_names, results_list):
        comparison.append({
            "Model": name,
            "Topics": results.get("n_topics", 0),
            "Diversity": results.get("diversity", 0),
            "Coherence (NPMI)": results.get("coherence_npmi", None),
            "Silhouette": results.get("silhouette", 0),
            "Outlier %": results.get("outlier_ratio", 0) * 100
        })
    
    return pd.DataFrame(comparison)


def create_evaluation_report(
    results: Dict,
    topic_model,
    output_path: Optional[str] = None
) -> str:
    """
    Create a text evaluation report.
    
    Args:
        results: Evaluation results dictionary
        topic_model: Fitted BERTopic model
        output_path: Optional path to save report
    
    Returns:
        Report string
    """
    report = []
    report.append("=" * 60)
    report.append("TOPIC MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append("SUMMARY METRICS")
    report.append("-" * 40)
    report.append(f"Number of Topics: {results['n_topics']}")
    report.append(f"Topic Diversity: {results['diversity']:.4f}")
    if results['coherence_npmi']:
        report.append(f"Coherence (NPMI): {results['coherence_npmi']:.4f}")
    report.append(f"Silhouette Score: {results['silhouette']:.4f}")
    report.append(f"Outlier Ratio: {results['outlier_ratio']:.2%}")
    report.append("")
    
    report.append("TOP TOPICS BY SIZE")
    report.append("-" * 40)
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.head(10).iterrows():
        if row['Topic'] != -1:
            report.append(f"Topic {row['Topic']}: {row['Count']} docs - {row['Name']}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")
    
    return report_text


# Example usage
if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Available functions:")
    print("  - compute_topic_diversity()")
    print("  - compute_coherence_scores()")
    print("  - compute_silhouette()")
    print("  - evaluate_topic_model()")
    print("  - compare_models()")
