"""
topic_model.py - BERTopic pipeline wrapper

This module provides a wrapper around BERTopic for topic modeling
arXiv cs.AI abstracts with customizable components.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import os


def create_embedding_model(model_name: str = "all-mpnet-base-v2") -> SentenceTransformer:
    """
    Create sentence transformer embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model
            Options:
            - "all-mpnet-base-v2" (high quality, slower)
            - "all-MiniLM-L6-v2" (faster, good quality)
    
    Returns:
        SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def create_umap_model(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42
) -> UMAP:
    """
    Create UMAP dimensionality reduction model.
    
    Args:
        n_neighbors: Number of neighbors for local structure
        n_components: Target dimensions
        min_dist: Minimum distance between points
        metric: Distance metric
        random_state: Random seed for reproducibility
    
    Returns:
        UMAP model
    """
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=False
    )


def create_hdbscan_model(
    min_cluster_size: int = 15,
    min_samples: int = 10,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    prediction_data: bool = True
) -> HDBSCAN:
    """
    Create HDBSCAN clustering model.
    
    Args:
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood
        metric: Distance metric
        cluster_selection_method: "eom" or "leaf"
        prediction_data: Generate prediction data
    
    Returns:
        HDBSCAN model
    """
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data
    )


def create_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english",
    min_df: int = 5,
    max_df: float = 0.95
) -> CountVectorizer:
    """
    Create CountVectorizer for c-TF-IDF.
    
    Args:
        ngram_range: Range of n-grams to extract
        stop_words: Stop words to remove
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        CountVectorizer
    """
    return CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df
    )


def build_bertopic_model(
    embedding_model: str = "all-mpnet-base-v2",
    umap_params: Optional[Dict] = None,
    hdbscan_params: Optional[Dict] = None,
    vectorizer_params: Optional[Dict] = None,
    nr_topics: Optional[int] = None,
    top_n_words: int = 10,
    verbose: bool = True
) -> BERTopic:
    """
    Build complete BERTopic model with custom components.
    
    Args:
        embedding_model: Name of sentence-transformers model
        umap_params: Parameters for UMAP
        hdbscan_params: Parameters for HDBSCAN
        vectorizer_params: Parameters for CountVectorizer
        nr_topics: Number of topics to reduce to (None for auto)
        top_n_words: Number of words per topic
        verbose: Print progress
    
    Returns:
        Configured BERTopic model
    """
    # Set defaults
    umap_params = umap_params or {}
    hdbscan_params = hdbscan_params or {}
    vectorizer_params = vectorizer_params or {}
    
    # Create components
    sentence_model = create_embedding_model(embedding_model)
    umap_model = create_umap_model(**umap_params)
    hdbscan_model = create_hdbscan_model(**hdbscan_params)
    vectorizer_model = create_vectorizer(**vectorizer_params)
    ctfidf_model = ClassTfidfTransformer()
    
    # Build BERTopic
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
        top_n_words=top_n_words,
        verbose=verbose,
        calculate_probabilities=True
    )
    
    return topic_model


def fit_transform_model(
    topic_model: BERTopic,
    documents: List[str],
    embeddings: Optional[np.ndarray] = None
) -> Tuple[List[int], np.ndarray]:
    """
    Fit the topic model and transform documents.
    
    Args:
        topic_model: BERTopic model
        documents: List of document strings
        embeddings: Pre-computed embeddings (optional)
    
    Returns:
        Tuple of (topics, probabilities)
    """
    print(f"Fitting topic model on {len(documents)} documents...")
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude -1 (outliers)
    n_outliers = (np.array(topics) == -1).sum()
    
    print(f"\nResults:")
    print(f"  Number of topics: {n_topics}")
    print(f"  Outliers: {n_outliers} ({100*n_outliers/len(documents):.1f}%)")
    
    return topics, probs


def compute_embeddings(
    documents: List[str],
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Pre-compute document embeddings.
    
    Args:
        documents: List of document strings
        model_name: Sentence-transformers model name
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        NumPy array of embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """Save embeddings to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from file."""
    return np.load(path)


def save_model(topic_model: BERTopic, path: str) -> None:
    """Save BERTopic model."""
    os.makedirs(path, exist_ok=True)
    topic_model.save(path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
    print(f"Saved model to {path}")


def load_model(path: str, embedding_model: str = "all-mpnet-base-v2") -> BERTopic:
    """Load BERTopic model."""
    return BERTopic.load(path, embedding_model=embedding_model)


def get_topic_summary(topic_model: BERTopic) -> pd.DataFrame:
    """
    Get summary of all topics.
    
    Args:
        topic_model: Fitted BERTopic model
    
    Returns:
        DataFrame with topic info
    """
    return topic_model.get_topic_info()


def get_representative_docs(
    topic_model: BERTopic,
    topic_id: int,
    n_docs: int = 5
) -> List[str]:
    """
    Get representative documents for a topic.
    
    Args:
        topic_model: Fitted BERTopic model
        topic_id: Topic ID
        n_docs: Number of documents to return
    
    Returns:
        List of representative document strings
    """
    return topic_model.get_representative_docs(topic_id)[:n_docs]


# Example usage
if __name__ == "__main__":
    # Example with sample documents
    sample_docs = [
        "Deep learning neural networks for image classification",
        "Natural language processing with transformers and BERT",
        "Reinforcement learning in robotics applications",
        "Computer vision object detection algorithms",
        "Machine translation using attention mechanisms"
    ]
    
    # Build and fit model
    model = build_bertopic_model(
        embedding_model="all-MiniLM-L6-v2",  # Faster for testing
        hdbscan_params={"min_cluster_size": 2}
    )
    topics, probs = fit_transform_model(model, sample_docs)
    print(f"Topics: {topics}")
