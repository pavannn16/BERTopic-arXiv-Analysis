"""
Project utilities for BERTopic arXiv Analysis
Handles configuration loading, Google Drive access, and path setup
"""

import os
import yaml
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for config.yaml
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Search for config.yaml in common locations
        possible_paths = [
            'config.yaml',
            '../config.yaml',
            '/content/BERTopic-arXiv-Analysis/config.yaml',
            '/content/drive/MyDrive/BERTopic-arXiv-Analysis/config.yaml'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        print("⚠️ Config file not found, using defaults")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from {config_path}")
    return config


def get_default_config():
    """Return default configuration if config.yaml not found."""
    return {
        'mode': 'infer',
        'data': {
            'gdrive_folder_id': '1T3vkmvm8YbUCXCMRoroWDXJlKHfMC5Gj',
            'arxiv': {
                'category': 'cs.AI',
                'max_results': 20000,
                'months_back': 24,
                'batch_size': 100,
                'delay_seconds': 3.0
            }
        },
        'model': {
            'embedding_model': 'all-mpnet-base-v2',
            'umap': {'n_neighbors': 10, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine'},
            'hdbscan': {'min_cluster_size': 50, 'min_samples': 10, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
            'vectorizer': {'ngram_range': [1, 2], 'min_df': 5, 'max_df': 0.95, 'stop_words': 'english'}
        },
        'outlier_reduction': {'enabled': True, 'strategy': 'c-tf-idf', 'threshold': 0.1},
        'evaluation': {'coherence_metric': 'c_npmi', 'top_n_words': 10, 'run_lda_baseline': True},
        'random_seed': 42
    }


def setup_environment(config=None):
    """
    Set up the project environment based on execution context.
    Handles both personal Drive mounting and public folder access.
    
    Returns:
        tuple: (PROJECT_PATH, is_colab, mode)
    """
    if config is None:
        config = load_config()
    
    mode = config.get('mode', 'infer')
    
    # Detect environment
    try:
        from IPython import get_ipython
        is_colab = 'google.colab' in str(get_ipython())
    except:
        is_colab = False
    
    if is_colab:
        PROJECT_PATH = setup_colab_environment(config, mode)
    else:
        # Running locally
        PROJECT_PATH = str(Path(os.getcwd()).parent) if 'notebooks' in os.getcwd() else os.getcwd()
        print("✅ Running locally")
    
    # Create directory structure
    for folder in ['data/raw', 'data/processed', 'data/embeddings', 'models', 'results/visualizations']:
        os.makedirs(f'{PROJECT_PATH}/{folder}', exist_ok=True)
    
    print(f"📁 Project path: {PROJECT_PATH}")
    print(f"🔧 Mode: {mode.upper()}")
    
    return PROJECT_PATH, is_colab, mode


def setup_colab_environment(config, mode):
    """
    Set up Google Colab environment.
    - In TRAIN mode: Mount personal Drive for read/write
    - In INFER mode: Download from public shared folder
    """
    from google.colab import drive
    
    if mode == 'train':
        # Training mode - need personal Drive for saving results
        print("🔧 TRAIN mode: Mounting personal Google Drive...")
        drive.mount('/content/drive')
        PROJECT_PATH = '/content/drive/MyDrive/BERTopic-arXiv-Analysis'
        print("✅ Personal Drive mounted (read/write access)")
    else:
        # Inference mode - use public shared data
        print("🔧 INFER mode: Using public shared data...")
        PROJECT_PATH = '/content/BERTopic-arXiv-Analysis'
        os.makedirs(PROJECT_PATH, exist_ok=True)
        
        # Download data from public Drive folder
        download_from_public_drive(config, PROJECT_PATH)
        print("✅ Public data downloaded (read-only mode)")
    
    return PROJECT_PATH


def download_from_public_drive(config, project_path):
    """
    Download files from public Google Drive folder using gdown.
    """
    import subprocess
    
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        subprocess.run(['pip', 'install', 'gdown', '-q'], check=True)
        import gdown
    
    folder_id = config['data']['gdrive_folder_id']
    
    print(f"📥 Downloading from public Drive folder: {folder_id}")
    
    # Download entire folder
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    try:
        gdown.download_folder(url, output=project_path, quiet=False, use_cookies=False)
        print("✅ Download complete!")
    except Exception as e:
        print(f"⚠️ Folder download failed: {e}")
        print("Trying individual file downloads...")
        download_essential_files(folder_id, project_path)


def download_essential_files(folder_id, project_path):
    """
    Download essential files individually if folder download fails.
    """
    import gdown
    
    # Known file IDs from your public folder (you may need to update these)
    essential_files = {
        'data/processed/arxiv_cs_ai_processed.csv': None,  # Will be fetched from folder listing
        'data/embeddings/embeddings_mpnet.npy': None,
        'models/bertopic_best_model': None,
        'results/best_config.json': None,
    }
    
    print("📥 Downloading essential files...")
    
    # List files in folder and download
    try:
        import requests
        # Use gdown to list folder contents
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=project_path, quiet=True, use_cookies=False)
    except Exception as e:
        print(f"⚠️ Could not download files: {e}")
        print("Please ensure the Drive folder is publicly accessible.")


def get_model_params_from_config(config):
    """
    Extract model parameters from config for BERTopic initialization.
    
    Returns:
        dict: Parameters for UMAP, HDBSCAN, and vectorizer
    """
    model_config = config.get('model', {})
    
    return {
        'embedding_model': model_config.get('embedding_model', 'all-mpnet-base-v2'),
        'umap': model_config.get('umap', {}),
        'hdbscan': model_config.get('hdbscan', {}),
        'vectorizer': model_config.get('vectorizer', {}),
    }


def print_config_summary(config):
    """Print a summary of the current configuration."""
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Mode: {config.get('mode', 'infer').upper()}")
    print(f"Embedding Model: {config['model']['embedding_model']}")
    print(f"min_cluster_size: {config['model']['hdbscan']['min_cluster_size']}")
    print(f"n_neighbors: {config['model']['umap']['n_neighbors']}")
    print(f"n_components: {config['model']['umap']['n_components']}")
    print(f"Outlier Reduction: {config['outlier_reduction']['enabled']}")
    print("="*50 + "\n")
