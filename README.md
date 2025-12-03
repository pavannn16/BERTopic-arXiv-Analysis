# 🤖 Topic Modeling arXiv cs.AI with BERTopic

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
4. [Quick Start (Google Colab)](#-quick-start-google-colab---recommended)
5. [Local Setup](#-local-setup-alternative)
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

### Model Comparison

| Model | Coherence (NPMI) | Diversity | Parameters | Winner |
|-------|------------------|-----------|------------|--------|
| **BERTopic (MPNet)** | **+0.08** | **69%** | 110M | 🏆 |
| BERTopic (MiniLM) | +0.07 | 68% | 22M | ⚡ Fast |
| LDA Baseline | -0.09 | N/A | N/A | ❌ |

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Documents Processed** | 20,000 | arXiv cs.AI papers |
| **Topics Discovered** | ~150-200 | Auto-determined via HDBSCAN |
| **Coherence (NPMI)** | **+0.08** | ✅ Good (positive = coherent) |
| **Topic Diversity** | **69%** | ✅ High variety |
| **Outlier Rate** | <15% | After outlier reduction |
| **Improvement over LDA** | **+0.17** | Significantly better |

### Hyperparameter Tuning Results
- **60 configurations tested** across embedding models, cluster sizes, and UMAP parameters
- **Best config**: MPNet embeddings, min_cluster_size=15-20, n_neighbors=15
- **Outlier reduction**: c-TF-IDF strategy reduces outliers by ~50%

---

## 🚀 Quick Start (Google Colab) - RECOMMENDED

**Best for evaluation - runs entirely in browser with free GPU, no installation required!**

### Two Running Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **🔮 INFER** (default) | Downloads pre-trained model & data from public Google Drive | Evaluation, quick demo, reproducing results |
| **🏋️ TRAIN** | Mounts your personal Drive, fetches fresh data from arXiv | Full training run, your own experiments |

> **Mode is configured in `config.yaml`** - set `mode: "infer"` or `mode: "train"`

### 📥 Public Dataset & Models

All data and trained models are publicly available - **no login required!**

> **📦 Google Drive Folder:** [Public BERTopic-arXiv-Analysis Data](https://drive.google.com/drive/folders/1T3vkmvm8YbUCXCMRoroWDXJlKHfMC5Gj)
> 
> Contains: raw data, processed data, embeddings, trained models, results

### Step 1: Open Notebooks in Colab

Click these direct links (no setup required!):

- [01_data_collection.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/01_data_collection.ipynb)
- [02_preprocessing.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/02_preprocessing.ipynb)
- [03_topic_modeling.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/03_topic_modeling.ipynb)
- [03b_hyperparameter_tuning.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/03b_hyperparameter_tuning.ipynb) ⭐ NEW
- [04_evaluation.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/04_evaluation.ipynb)
- [05_visualization.ipynb](https://colab.research.google.com/github/pavannn16/BERTopic-arXiv-Analysis/blob/main/notebooks/05_visualization.ipynb)

### Step 2: Enable GPU Runtime (Optional but Faster)

1. **Runtime → Change runtime type**
2. Select **GPU** (T4 is free, A100 with Colab Pro)
3. Click **Save**

### Step 3: Run Notebooks

**🔮 INFER Mode (Default):** Notebooks automatically clone repo, load config, and download data from public Drive. Just run cells!

**🏋️ TRAIN Mode:** Change `mode: "train"` in `config.yaml` to fetch fresh data and train from scratch.

| Order | Notebook | INFER Mode | TRAIN Mode |
|-------|----------|------------|------------|
| 1️⃣ | `01_data_collection.ipynb` | Downloads existing data | Fetches 20K papers from arXiv (~15 min) |
| 2️⃣ | `02_preprocessing.ipynb` | Uses processed data | Cleans text (~1 min) |
| 3️⃣ | `03_topic_modeling.ipynb` | Loads pre-trained model | Trains BERTopic (~5 min) |
| 3️⃣b | `03b_hyperparameter_tuning.ipynb` | Loads tuned model | Grid search (~15 min) |
| 4️⃣ | `04_evaluation.ipynb` | Evaluates existing model | Same (~5 min) |
| 5️⃣ | `05_visualization.ipynb` | Generates visualizations | Same (~2 min) |

**INFER mode total: ~5 minutes** | **TRAIN mode total: ~45 minutes**

---

## 💻 Local Setup (Alternative)

### Prerequisites
- Python 3.9+
- 16GB+ RAM
- GPU optional (CPU works but slower for notebook 03)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/pavannn16/BERTopic-arXiv-Analysis.git
cd BERTopic-arXiv-Analysis

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
BERTopic-arXiv-Analysis/
│
├── ⚙️ config.yaml                        # ⭐ Configuration file (mode, hyperparameters)
│
├── 📓 notebooks/                         # Run these in order
│   ├── 01_data_collection.ipynb         # Fetch 20K arXiv papers
│   ├── 02_preprocessing.ipynb           # Text cleaning
│   ├── 03_topic_modeling.ipynb          # BERTopic training (GPU)
│   ├── 03b_hyperparameter_tuning.ipynb  # ⭐ Grid search & model comparison
│   ├── 04_evaluation.ipynb              # Metrics & baseline comparisons
│   └── 05_visualization.ipynb           # Interactive visualizations
│
├── 📦 src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── fetch_arxiv.py                   # arXiv API utilities
│   ├── preprocess.py                    # Text cleaning functions
│   ├── topic_model.py                   # BERTopic wrapper
│   ├── evaluate.py                      # Evaluation metrics
│   └── utils.py                         # ⭐ Config loader & Drive download utilities
│
├── 📊 data/                              # Generated by notebooks
│   ├── raw/                             # arxiv_cs_ai_raw.csv
│   ├── processed/                       # arxiv_cs_ai_processed.csv
│   └── embeddings/                      # MPNet & MiniLM embeddings
│
├── 🤖 models/                            
│   ├── bertopic_model/                  # Default model
│   └── bertopic_best_model/             # ⭐ Best from hyperparameter tuning
│
├── 📈 results/                           
│   ├── hyperparameter_search_results.csv # Grid search results
│   ├── model_comparison.csv             # MPNet vs MiniLM vs LDA
│   ├── evaluation_report.txt            # Comprehensive report
│   └── *.html                           # Interactive visualizations
│
├── 📄 requirements.txt                   # Python dependencies
└── 📄 README.md                          # This file
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
