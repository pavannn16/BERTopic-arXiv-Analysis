# 🤖 Topic Modeling arXiv cs.AI with BERTopic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Team:** Pavan Chauhan, Vedanta Nayak  
**Course:** CS5660 – Advanced Topics in AI  
**Institution:** California State University, Los Angeles  
**Date:** December 2025

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Results Summary](#-results-summary)
3. [Quick Start (Google Colab)](#-quick-start-google-colab---recommended)
4. [Local Setup](#-local-setup-alternative)
5. [Project Structure](#-project-structure)
6. [Pipeline Details](#-pipeline-details)
7. [Methodology](#-methodology)
8. [Evaluation Metrics](#-evaluation-metrics)
9. [References](#-references)

---

## 🎯 Project Overview

This project implements an **unsupervised topic modeling system** for arXiv cs.AI research paper abstracts using **BERTopic**. The pipeline automatically discovers, labels, and visualizes coherent research topics from **20,000 AI research papers**.

### Why BERTopic over LDA?
| Traditional (LDA) | BERTopic (Ours) |
|-------------------|-----------------|
| Bag-of-words representation | Semantic embeddings (SBERT) |
| Fixed number of topics | Automatic topic discovery |
| Word-level features only | Contextual understanding |
| Poor on short text | Excellent on abstracts |

### Key Components
| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Embeddings** | Sentence-BERT (`all-mpnet-base-v2`) | 768-dim semantic vectors |
| **Dim. Reduction** | UMAP (n_neighbors=15, n_components=5) | Preserve local structure |
| **Clustering** | HDBSCAN (min_cluster_size=20) | Density-based, handles outliers |
| **Topic Labels** | c-TF-IDF with bigrams | Interpretable keywords |

---

## 📊 Results Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Documents Processed** | 19,998 | After filtering short abstracts |
| **Topics Discovered** | 167 | Automatically determined |
| **Coherence (NPMI)** | **0.0798** | ✅ Good (positive = coherent) |
| **Topic Diversity** | **69.0%** | ✅ High variety |
| **Silhouette Score** | 0.0358 | Reasonable separation |
| **LDA Baseline** | -0.0869 | ❌ Poor coherence |
| **BERTopic Improvement** | **+0.167** | Significantly better than LDA |

---

## 🚀 Quick Start (Google Colab) - RECOMMENDED

**Best for evaluation - runs entirely in browser with free GPU, no installation required!**

### Step 1: Download Pre-collected Dataset (Optional but Faster)

> **📥 Dataset Download:** [Google Drive Link](https://drive.google.com/YOUR_LINK_HERE)
> 
> This saves ~15 minutes of API fetching. Place files in the `data/raw/` folder.

### Step 2: Open Notebooks in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook → GitHub tab**
3. Enter repository URL: `https://github.com/pavannn16/CS5660-BERTopic-arXiv`
4. Select a notebook to open

**Or upload directly:**
1. **File → Upload notebook**
2. Select `.ipynb` files from the `notebooks/` folder

### Step 3: Enable GPU Runtime

1. **Runtime → Change runtime type**
2. Select **GPU** (T4 is free, A100 with Colab Pro)
3. Click **Save**

### Step 4: Run Notebooks in Order

| Order | Notebook | What it Does | Time |
|-------|----------|--------------|------|
| 1️⃣ | `01_data_collection.ipynb` | Fetches 20K papers from arXiv | ~15 min |
| 2️⃣ | `02_preprocessing.ipynb` | Cleans text data | ~1 min |
| 3️⃣ | `03_topic_modeling.ipynb` | Trains BERTopic model | ~2 min |
| 4️⃣ | `04_evaluation.ipynb` | Computes metrics + LDA baseline | ~5 min |
| 5️⃣ | `05_visualization.ipynb` | Generates interactive plots | ~2 min |

**Total runtime: ~25 minutes with GPU**

---

## 💻 Local Setup (Alternative)

### Prerequisites
- Python 3.9+
- 16GB+ RAM
- GPU optional (CPU works but slower for notebook 03)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/pavannn16/CS5660-BERTopic-arXiv.git
cd CS5660-BERTopic-arXiv

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

### Run Pipeline
Open notebooks in Jupyter and run in order: `01` → `02` → `03` → `04` → `05`

---

## 📁 Project Structure

```
CS5660-BERTopic-arXiv/
│
├── 📓 notebooks/                    # Run these in order
│   ├── 01_data_collection.ipynb    # Fetch 20K arXiv papers
│   ├── 02_preprocessing.ipynb      # Text cleaning
│   ├── 03_topic_modeling.ipynb     # BERTopic training (GPU)
│   ├── 04_evaluation.ipynb         # Metrics & LDA comparison
│   └── 05_visualization.ipynb      # Interactive visualizations
│
├── 📦 src/                          # Reusable Python modules
│   ├── fetch_arxiv.py              # arXiv API utilities
│   ├── preprocess.py               # Text cleaning functions
│   ├── topic_model.py              # BERTopic wrapper
│   └── evaluate.py                 # Evaluation metrics
│
├── 📊 data/                         # Generated by notebooks
│   ├── raw/                        # arxiv_cs_ai_raw.csv
│   ├── processed/                  # arxiv_cs_ai_processed.csv
│   └── embeddings/                 # document_embeddings.npy
│
├── 🤖 models/                       # Saved BERTopic model
├── 📈 results/                      # Metrics, reports, visualizations
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # This file
```

---

## ⚙️ Pipeline Details

### Notebook 1: Data Collection
- **Source:** arXiv API (`cs.AI` category)
- **Volume:** 20,000 papers (recent 5 months)
- **Bypass API limit:** Two date-range queries, deduplicated
- **Output:** `data/raw/arxiv_cs_ai_raw.csv`

### Notebook 2: Preprocessing
- Unicode normalization (NFKD)
- LaTeX removal (`$...$`, `\begin{equation}`)
- URL and arXiv reference removal
- Boilerplate phrase removal ("In this paper...")
- **Output:** `data/processed/arxiv_cs_ai_processed.csv`

### Notebook 3: Topic Modeling
```
Documents → SBERT → UMAP → HDBSCAN → c-TF-IDF → Topics
   20K      768-dim  5-dim   167 clusters   Keywords
```
- **Output:** `models/bertopic_model/`, `data/embeddings/`

### Notebook 4: Evaluation
- NPMI Coherence (Gensim)
- Topic Diversity
- Silhouette Score
- LDA Baseline (50 topics)
- **Output:** `results/evaluation_results.json`

### Notebook 5: Visualization
- Topic barchart (top 20 topics)
- Topic similarity heatmap
- Hierarchical clustering dendrogram
- 2D document scatter (UMAP)
- Word clouds per topic
- **Output:** `results/visualizations/*.html`

---

## 🔬 Methodology

### BERTopic Pipeline
1. **Embed:** Convert abstracts to 768-dim vectors using Sentence-BERT
2. **Reduce:** UMAP projects to 5 dimensions preserving neighborhood structure
3. **Cluster:** HDBSCAN finds density-based clusters (topics) + outliers
4. **Represent:** c-TF-IDF extracts distinctive keywords per cluster

### Hyperparameters
| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| UMAP | n_neighbors | 15 | Balance local/global structure |
| UMAP | n_components | 5 | Sufficient for clustering |
| UMAP | min_dist | 0.0 | Tighter clusters |
| HDBSCAN | min_cluster_size | 20 | Minimum topic size |
| c-TF-IDF | ngram_range | (1, 2) | Include bigrams |

---

## 📏 Evaluation Metrics

| Metric | Description | Our Score |
|--------|-------------|-----------|
| **NPMI Coherence** | Normalized PMI of top-10 word pairs. Range: -1 to 1. Positive = words co-occur meaningfully. | **0.0798** |
| **Topic Diversity** | % unique words across all topic top-10 lists. Higher = less overlap between topics. | **69.0%** |
| **Silhouette Score** | Cluster separation quality. Range: -1 to 1. >0 = samples closer to own cluster. | **0.0358** |

### Comparison with LDA
| Model | NPMI Coherence | Winner |
|-------|----------------|--------|
| **BERTopic** | +0.0798 | ✅ |
| LDA (50 topics) | -0.0869 | ❌ |
| **Delta** | +0.167 | BERTopic wins |

---

## 📚 References

1. **BERTopic:** Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794

2. **Sentence-BERT:** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP

3. **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv:1802.03426

4. **HDBSCAN:** Campello, R. J., et al. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD

5. **LDA:** Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR

---

## 📝 License

Educational project for CS5660 Final at California State University, Los Angeles.

---

*Last updated: December 2025*
