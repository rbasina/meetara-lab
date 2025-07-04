#!/usr/bin/env python3
"""
MeeTARA Lab - Centralized Path Configuration
Defines the organized folder structure paths for consistency across all components
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Organized Data Paths
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_DATA_PATH = DATA_ROOT / "training"
REAL_DATA_PATH = DATA_ROOT / "real"
SYNTHESIS_DATA_PATH = DATA_ROOT / "synthesis"
DOMAINS_DATA_PATH = DATA_ROOT / "domains"

# Organized Models Paths
MODELS_ROOT = PROJECT_ROOT / "models"
UNIVERSAL_FULL_MODELS = MODELS_ROOT / "A_universal_full"
UNIVERSAL_LITE_MODELS = MODELS_ROOT / "B_universal_lite"
CATEGORY_SPECIFIC_MODELS = MODELS_ROOT / "C_category_specific"
DOMAIN_SPECIFIC_MODELS = MODELS_ROOT / "D_domain_specific"
GGUF_MODELS = MODELS_ROOT / "gguf"

# Legacy paths (for backwards compatibility)
LEGACY_TRAINING_DATA = PROJECT_ROOT / "model-factory" / "real_training_data"
LEGACY_GGUF_MODELS = PROJECT_ROOT / "model-factory" / "trinity_gguf_models"

def get_training_data_path(category: str) -> Path:
    """Get organized training data path for a category"""
    return REAL_DATA_PATH / category

def get_category_models_path(category: str) -> Path:
    """Get organized category-specific models path"""
    return CATEGORY_SPECIFIC_MODELS / category

def get_domain_models_path(category: str, domain: str = None) -> Path:
    """Get organized domain-specific models path"""
    if domain:
        return DOMAIN_SPECIFIC_MODELS / category / domain
    return DOMAIN_SPECIFIC_MODELS / category

def get_gguf_models_path(model_type: str = "production") -> Path:
    """Get organized GGUF models path"""
    return GGUF_MODELS / model_type

if __name__ == "__main__":
    print("MeeTARA Lab - Path Configuration")
    print("=" * 50)
    print(f" Project Root: {PROJECT_ROOT}")
    print(f" Data Root: {DATA_ROOT}")
    print(f" Models Root: {MODELS_ROOT}")
    print("\n Organized Structure:")
    print(f"   Training Data: {TRAINING_DATA_PATH}")
    print(f"   Real Data: {REAL_DATA_PATH}")
    print(f"   Category Models: {CATEGORY_SPECIFIC_MODELS}")
    print(f"   Domain Models: {DOMAIN_SPECIFIC_MODELS}")
    print(f"   GGUF Models: {GGUF_MODELS}")
