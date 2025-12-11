# 🤖 Topic Modeling arXiv cs.AI with BERTopic

[![GitHub Repo](https://img.shields.io/badge/GitHub-pavannn16%2FBERTopic--arXiv--Analysis-blue?logo=github)](https://github.com/pavannn16/BERTopic-arXiv-Analysis)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/01_data_collection.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Team:** Pavan Chauhan, Vedanta Nayak  
**Course:** CS5660 – Advanced Topics in AI  
**Institution:** California State University, Los Angeles  
**Date:** December 2025

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Results Summary](#-results-summary)
4. [Quick Start (Local)](#-quick-start-local---recommended)
5. [Google Colab (Alternative)](#-google-colab-alternative)
6. [Project Structure](#-project-structure)
7. [Pipeline Details](#-pipeline-details)
8. [Methodology](#-methodology)
9. [Evaluation Metrics](#-evaluation-metrics)
10. [References](#-references)

---

## 🎯 Project Overview

This project implements a **state-of-the-art unsupervised topic modeling system** for arXiv cs.AI research paper abstracts using **BERTopic**. The pipeline automatically discovers, labels, and visualizes coherent research topics from **20,000 AI research papers**.

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
| **Dim. Reduction** | UMAP (optimized hyperparameters) | Preserve local structure |
| **Clustering** | HDBSCAN (tuned min_cluster_size) | Density-based, handles outliers |
| **Topic Labels** | c-TF-IDF with bigrams | Interpretable keywords |

---

## ✨ Key Features

### 🔬 Industry-Standard Methodology
- **Pre-trained Sentence-BERT** for semantic understanding (no training required)
- **Systematic hyperparameter tuning** with grid search
- **Multiple embedding model comparison** (MPNet vs MiniLM)
- **Outlier reduction** for improved topic assignment

### 📊 Comprehensive Evaluation
- **3 coherence metrics**: NPMI, C_V, UMass
- **Topic diversity analysis**
- **Silhouette score** for cluster quality
- **LDA baseline comparison**

### 🎨 Rich Visualizations
- Interactive 2D topic maps (UMAP projection)
- Topic similarity heatmaps
- Hierarchical dendrograms
- Word clouds per topic
- Topic trends over time

---

## 📊 Results Summary

### Best Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Documents Processed** | 19,998 | arXiv cs.AI papers |
| **Topics Discovered** | 91 | Auto-determined via HDBSCAN |
| **Coherence (NPMI)** | **0.0949** | ✅ Good (positive = coherent) |
| **Topic Diversity** | **58.1%** | ✅ Good variety |
| **Improvement over LDA** | **+279%** | Significantly better |

### Model Comparison

| Model | Coherence (NPMI) | Diversity | Parameters | Winner |
|-------|------------------|-----------|------------|--------|
| **BERTopic (Best Config)** | **0.0949** | **58.1%** | 110M | 🏆 |
| BERTopic (MPNet default) | 0.0774 | 66.9% | 110M | |
| BERTopic (MiniLM) | 0.0574 | 69.4% | 22M | ⚡ Fast |
| LDA Baseline | 0.0250 | N/A | N/A | ❌ |

### Best Hyperparameters
- **Embedding Model**: `all-mpnet-base-v2`
- **min_cluster_size**: 50
- **n_neighbors**: 10
- **n_components**: 10

---

## 🚀 Quick Start (Local) - RECOMMENDED

**All data, trained models, and results are included in the repository!** Clone and run - no downloads required.

### Prerequisites
- Python 3.9+ (tested with 3.11)
- 16GB+ RAM recommended
- ~500MB disk space

### Installation

```bash
# 1. Clone repository (includes all data!)
git clone https://github.com/pavannn16/BERTopic-arXiv-Analysis.git
cd BERTopic-arXiv-Analysis

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook
```

Open notebooks in order and run cells:

| Order | Notebook | What It Does | Runtime |
|-------|----------|--------------|---------|
| 1️⃣ | `01_data_collection.ipynb` | Loads existing arXiv data (20K papers) | ~1 sec |
| 2️⃣ | `02_preprocessing.ipynb` | Loads preprocessed documents | ~1 sec |
| 3️⃣ | `03_topic_modeling.ipynb` | Loads trained BERTopic model | ~10 sec |
| 4️⃣ | `04_evaluation.ipynb` | Runs evaluation metrics | ~2 min |
| 5️⃣ | `05_visualization.ipynb` | Generates visualizations | ~1 min |

**Total runtime in INFER mode: ~5 minutes**

### Two Running Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **🔮 INFER** (default) | Uses pre-existing data & models in repo | Quick evaluation, reproducing results |
| **🏋️ TRAIN** | Fetches fresh data from arXiv, trains new model | New experiments, custom training |

> **Mode is configured in `config.yaml`** - set `mode: "infer"` (default) or `mode: "train"`

---

## ☁️ Google Colab (Alternative)

**Run in browser with free GPU - useful for TRAIN mode or if you don't have local Python.**

### Open Notebooks Directly

Click these links (no setup required!):

| Notebook | Link |
|----------|------|
| 01 Data Collection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/01_data_collection.ipynb) |
| 02 Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/02_preprocessing.ipynb) |
| 03 Topic Modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/03_topic_modeling.ipynb) |
| 04 Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/04_evaluation.ipynb) |
| 05 Visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/05_visualization.ipynb) |

### Enable GPU (Optional)

1. **Runtime → Change runtime type**
2. Select **GPU** (T4 is free)
3. Click **Save**

### Run Notebooks

Notebooks automatically clone the repo and load config. Just run cells in order!

**🔮 INFER Mode (Default):** Uses data from repo - no external downloads needed.

**🏋️ TRAIN Mode:** Change `mode: "train"` in `config.yaml` and mount your Google Drive for persistent storage.

---

## 📁 Project Structure

```
BERTopic-arXiv-Analysis/
│
├── ⚙️ config.yaml                        # ⭐ Configuration file (mode, hyperparameters)
│
├── 📓 notebooks/                         # Run these in order
│   ├── 01_data_collection.ipynb         # Load arXiv data
│   ├── 02_preprocessing.ipynb           # Text cleaning
│   ├── 03_topic_modeling.ipynb          # BERTopic training (GPU)
│   ├── 04_evaluation.ipynb              # Metrics & baseline comparisons
│   └── 05_visualization.ipynb           # Interactive visualizations
│
├── 📦 src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── fetch_arxiv.py                   # arXiv API utilities
│   ├── preprocess.py                    # Text cleaning functions
│   ├── topic_model.py                   # BERTopic wrapper
│   ├── evaluate.py                      # Evaluation metrics
│   └── utils.py                         # Config loader & setup utilities
│
├── 📊 data/                              # ✅ INCLUDED IN REPO (clone and run!)
│   ├── raw/                             # arxiv_cs_ai_raw.csv (20K papers)
│   │   ├── arxiv_cs_ai_raw.csv
│   │   └── arxiv_cs_ai_raw.json
│   ├── processed/                       # Cleaned documents
│   │   ├── arxiv_cs_ai_processed.csv
│   │   └── documents.json
│   └── embeddings/                      # Pre-computed SBERT embeddings
│       ├── embeddings_mpnet.npy         # MPNet (768-dim)
│       ├── embeddings_minilm.npy        # MiniLM (384-dim)
│       └── embeddings_2d*.npy           # UMAP reduced
│
├── 🤖 models/                            # ✅ INCLUDED IN REPO
│   ├── bertopic_model/                  # Default trained model
│   └── bertopic_best_model/             # Best from hyperparameter tuning
│
├── 📈 results/                           # ✅ INCLUDED IN REPO
│   ├── evaluation_metrics.json          # All metrics in JSON
│   ├── evaluation_report.txt            # Comprehensive report
│   ├── model_comparison.csv             # MPNet vs MiniLM vs LDA
│   ├── topic_assignments_best.csv       # Topic per document
│   ├── *.html                           # Interactive visualizations
│   └── *.png                            # Static plots
│
├── 📄 requirements.txt                   # Python dependencies
└── 📄 README.md                          # This file
```

> **📦 Repository size: ~460MB** - All data, models, and results are included for instant inference!

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
   20K      768-dim  5-dim   ~150 clusters   Keywords
```
- **Output:** `models/bertopic_model/`, `data/embeddings/`

### Notebook 3b: Hyperparameter Tuning ⭐ NEW
- **Grid search** over 60 configurations
- **Embedding comparison**: MPNet (110M) vs MiniLM (22M params)
- **HDBSCAN tuning**: min_cluster_size ∈ [10, 15, 20, 30, 50]
- **UMAP tuning**: n_neighbors ∈ [10, 15, 25], n_components ∈ [5, 10]
- **Outlier reduction**: c-TF-IDF reassignment strategy
- **Output:** `models/bertopic_best_model/`, `results/hyperparameter_search_results.csv`

### Notebook 4: Evaluation
- Multiple coherence metrics (NPMI, C_V, UMass)
- Topic Diversity
- Silhouette Score
- **Multi-model comparison**: BERTopic (MPNet) vs BERTopic (MiniLM) vs LDA
- **Output:** `results/evaluation_report.txt`, `results/model_comparison.csv`

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
1. **Embed:** Convert abstracts to 768-dim vectors using Sentence-BERT (pre-trained, no fine-tuning)
2. **Reduce:** UMAP projects to 5-10 dimensions preserving neighborhood structure
3. **Cluster:** HDBSCAN finds density-based clusters (topics) + outliers
4. **Represent:** c-TF-IDF extracts distinctive keywords per cluster
5. **Reduce Outliers:** Reassign outliers using c-TF-IDF similarity

### Hyperparameter Tuning Strategy
We perform systematic grid search to find optimal parameters:

| Component | Parameters Tested | Best Value |
|-----------|------------------|------------|
| **Embedding** | MPNet, MiniLM | MPNet |
| **UMAP n_neighbors** | 10, 15, 25 | 15 |
| **UMAP n_components** | 5, 10 | 5 |
| **HDBSCAN min_cluster_size** | 10, 15, 20, 30, 50 | 15-20 |

### Evaluation Criteria
Models are ranked by **combined score**:
```
Score = 0.5 × Coherence + 0.3 × Diversity + 0.2 × (1 - Outlier%)
```

---

## 📏 Evaluation Metrics

| Metric | Description | Our Score |
|--------|-------------|-----------|
| **NPMI Coherence** | Normalized PMI of top-10 word pairs. Range: -1 to 1. Positive = words co-occur meaningfully. | **+0.08** |
| **Topic Diversity** | % unique words across all topic top-10 lists. Higher = less overlap between topics. | **69%** |
| **Silhouette Score** | Cluster separation quality. Range: -1 to 1. >0 = samples closer to own cluster. | **0.04** |

### Multi-Model Comparison

| Model | Embedding | Params | Coherence | Speed | Use Case |
|-------|-----------|--------|-----------|-------|----------|
| **BERTopic (MPNet)** | all-mpnet-base-v2 | 110M | **+0.08** | Slower | Best quality |
| **BERTopic (MiniLM)** | all-MiniLM-L6-v2 | 22M | +0.07 | **5x faster** | Resource-constrained |
| **LDA** | Bag-of-Words | - | -0.09 | Fast | Baseline only |

### Key Findings
1. ✅ **BERTopic outperforms LDA** by +0.17 NPMI (significant improvement)
2. ✅ **MPNet slightly better than MiniLM** (+0.01 coherence) but 5x slower
3. ✅ **Hyperparameter tuning improves results** vs default configuration
4. ✅ **Outlier reduction assigns ~50% more documents** to meaningful topics

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
