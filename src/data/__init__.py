"""
Data Module - Clean imports from model-factory data generation
Wraps existing data functionality without breaking changes.
"""

import sys
from pathlib import Path

# Add model-factory to path for importing
project_root = Path(__file__).parent.parent.parent
model_factory_path = project_root / "model-factory"
sys.path.insert(0, str(model_factory_path))

# Import from existing files
try:
    from meetara_training_data_generator import (
        generate_training_data,
        create_domain_dataset,
        validate_data_quality
    )
    
    from integrated_gpu_pipeline import (
        generate_domain_training_data,
        prepare_training_datasets
    )
    
except ImportError as e:
    print(f"Warning: Could not import from model-factory data generators: {e}")
    # Provide fallback functions
    def generate_training_data(*args, **kwargs):
        raise NotImplementedError("Training data generation not available")
    
    def validate_data_quality(*args, **kwargs):
        raise NotImplementedError("Data validation not available")

# Additional data utilities
def validate_data(data_path: str) -> bool:
    """Validate data quality and format"""
    try:
        return validate_data_quality(data_path)
    except:
        return False

# Clean exports
__all__ = [
    "generate_training_data",
    "create_domain_dataset", 
    "validate_data_quality",
    "validate_data",
    "generate_domain_training_data",
    "prepare_training_datasets"
] 