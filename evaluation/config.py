"""
Configuration settings for RAG evaluation
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
TEST_DATASETS_DIR = PROJECT_ROOT / "test_datasets"
LOAD_INDEX_DIR = PROJECT_ROOT / "load_index"

# Evaluation settings
EVALUATION_CONFIG = {
    # Retrieval evaluation
    'retrieval': {
        'top_k': 5,
        'metrics': ['hit_rate', 'mrr', 'precision']
    },
    
    # Response quality evaluation
    'response_quality': {
        'use_rouge': True,
        'use_llm_judge': True,
        'llm_judge_model': 'gemini-2.5-flash',
        'max_response_length': 1000
    },
    
    # Stress testing
    'stress_test': {
        'concurrent_users': 10,
        'queries_per_user': 5,
        'timeout_seconds': 30
    },
    
    # Test data generation
    'test_generation': {
        'questions_per_document': 3,
        'max_document_length': 2000,
        'question_types': ['factual', 'analytical', 'summary']
    }
}

# Create directories if they don't exist
def ensure_directories():
    EVALUATION_RESULTS_DIR.mkdir(exist_ok=True)
    TEST_DATASETS_DIR.mkdir(exist_ok=True)

ensure_directories()