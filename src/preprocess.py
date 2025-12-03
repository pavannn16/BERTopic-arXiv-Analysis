"""
preprocess.py - Text preprocessing utilities for arXiv abstracts

This module provides functions to clean and normalize text data from arXiv
abstracts for topic modeling with BERTopic.
"""

import re
import pandas as pd
from typing import List, Optional
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize a single text string.
    
    Args:
        text: Input text (title or abstract)
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    
    # Remove LaTeX math expressions (common in arXiv)
    text = re.sub(r'\$[^$]+\$', ' MATH ', text)  # Inline math
    text = re.sub(r'\\\[[^\]]+\\\]', ' MATH ', text)  # Display math
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' MATH ', text, flags=re.DOTALL)
    
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove arXiv references
    text = re.sub(r'arXiv:\d+\.\d+', '', text)
    
    # Remove common boilerplate phrases
    boilerplate = [
        r'^in this paper,?\s*',
        r'^in this work,?\s*',
        r'^we present\s*',
        r'^we propose\s*',
        r'^this paper presents\s*',
        r'^this work presents\s*',
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra punctuation but keep sentence structure
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_columns: List[str] = ["title", "abstract"],
    combine_columns: bool = True,
    output_column: str = "text",
    min_length: int = 50
) -> pd.DataFrame:
    """
    Preprocess a DataFrame of arXiv papers.
    
    Args:
        df: Input DataFrame with paper data
        text_columns: Columns to clean
        combine_columns: Whether to combine title and abstract
        output_column: Name of output text column
        min_length: Minimum text length to keep
    
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Clean each text column
    for col in text_columns:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(clean_text)
    
    # Combine title and abstract if requested
    if combine_columns and "title_clean" in df.columns and "abstract_clean" in df.columns:
        df[output_column] = df["title_clean"] + ". " + df["abstract_clean"]
    elif "abstract_clean" in df.columns:
        df[output_column] = df["abstract_clean"]
    elif "title_clean" in df.columns:
        df[output_column] = df["title_clean"]
    
    # Remove short texts
    initial_count = len(df)
    df = df[df[output_column].str.len() >= min_length]
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} documents with text length < {min_length}")
    
    # Remove duplicates based on arxiv_id
    if "arxiv_id" in df.columns:
        df = df.drop_duplicates(subset=["arxiv_id"])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Preprocessed {len(df)} documents")
    
    return df


def get_documents(df: pd.DataFrame, text_column: str = "text") -> List[str]:
    """
    Extract document list for BERTopic.
    
    Args:
        df: Preprocessed DataFrame
        text_column: Column containing text
    
    Returns:
        List of document strings
    """
    return df[text_column].tolist()


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save preprocessed data to CSV.
    
    Args:
        df: Preprocessed DataFrame
        path: Output file path
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved preprocessed data to {path}")


def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load preprocessed data from CSV.
    
    Args:
        path: Input file path
    
    Returns:
        DataFrame with preprocessed data
    """
    return pd.read_csv(path)


# Example usage
if __name__ == "__main__":
    # Example text cleaning
    sample_text = """
    In this paper, we propose a novel approach to neural network optimization.
    Our method achieves $O(n^2)$ complexity using \\textbf{gradient descent}.
    See our code at https://github.com/example/repo
    """
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
