"""
fetch_arxiv.py - Utilities for fetching arXiv cs.AI abstracts

This module provides functions to collect paper metadata (title, abstract, date)
from the arXiv API for the cs.AI (Artificial Intelligence) category.
"""

import arxiv
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from tqdm import tqdm
import time
import json
import os


def fetch_arxiv_papers(
    category: str = "cs.AI",
    max_results: int = 5000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_path: Optional[str] = None,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Fetch papers from arXiv API for a given category.
    
    Args:
        category: arXiv category (default: cs.AI)
        max_results: Maximum number of papers to fetch
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        save_path: Path to save raw JSON data (optional)
        batch_size: Number of papers per API request
    
    Returns:
        DataFrame with columns: arxiv_id, title, abstract, authors, date, url, categories
    """
    
    # Build search query
    query = f"cat:{category}"
    
    # Add date filter if provided
    if start_date and end_date:
        # Note: arXiv API date filtering is limited, we filter post-fetch
        pass
    
    print(f"Fetching up to {max_results} papers from arXiv category: {category}")
    
    # Configure search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    client = arxiv.Client(
        page_size=batch_size,
        delay_seconds=3.0,  # Be respectful to API
        num_retries=5
    )
    
    try:
        for result in tqdm(client.results(search), total=max_results, desc="Fetching papers"):
            paper = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title.replace("\n", " ").strip(),
                "abstract": result.summary.replace("\n", " ").strip(),
                "authors": ", ".join([author.name for author in result.authors]),
                "date": result.published.strftime("%Y-%m-%d"),
                "url": result.entry_id,
                "categories": ", ".join(result.categories),
                "primary_category": result.primary_category
            }
            
            # Filter by date if specified
            if start_date:
                paper_date = datetime.strptime(paper["date"], "%Y-%m-%d")
                filter_start = datetime.strptime(start_date, "%Y-%m-%d")
                if paper_date < filter_start:
                    continue
            
            if end_date:
                paper_date = datetime.strptime(paper["date"], "%Y-%m-%d")
                filter_end = datetime.strptime(end_date, "%Y-%m-%d")
                if paper_date > filter_end:
                    continue
            
            papers.append(paper)
            
            if len(papers) >= max_results:
                break
                
    except Exception as e:
        print(f"Error during fetch: {e}")
        print(f"Successfully fetched {len(papers)} papers before error")
    
    # Create DataFrame
    df = pd.DataFrame(papers)
    
    # Save raw data if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(papers, f, indent=2)
        print(f"Raw data saved to {save_path}")
    
    print(f"\nTotal papers fetched: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_arxiv_data(json_path: str) -> pd.DataFrame:
    """
    Load previously saved arXiv data from JSON file.
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        DataFrame with paper data
    """
    with open(json_path, 'r') as f:
        papers = json.load(f)
    return pd.DataFrame(papers)


def get_date_range(months_back: int = 12) -> tuple:
    """
    Get date range for fetching recent papers.
    
    Args:
        months_back: Number of months to look back
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


# Example usage
if __name__ == "__main__":
    # Fetch recent cs.AI papers
    start, end = get_date_range(months_back=12)
    df = fetch_arxiv_papers(
        category="cs.AI",
        max_results=3000,
        start_date=start,
        end_date=end,
        save_path="data/raw/arxiv_cs_ai.json"
    )
    print(df.head())
