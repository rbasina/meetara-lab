"""
Models Module - Clean imports from model-factory directory
Wraps existing functionality without breaking changes.
"""

import sys
import os
from pathlib import Path

# Add model-factory to path for importing
project_root = Path(__file__).parent.parent.parent
model_factory_path = project_root / "model-factory"
sys.path.insert(0, str(model_factory_path))

# Import from existing files with clean naming
try:
    from trinity_master_gguf_factory import (
        create_gguf_model,
        optimize_model_size,
        validate_gguf_quality
    )
    
    from integrated_gpu_pipeline import (
        train_gpu_model,
        run_training_pipeline,
        generate_domain_model
    )
    
    from gpu_training_engine import (
        setup_gpu_environment,
        monitor_training_progress
    )
    
except ImportError as e:
    print(f"Warning: Could not import from model-factory: {e}")
    # Provide fallback functions
    def create_gguf_model(*args, **kwargs):
        raise NotImplementedError("GGUF creation not available")
    
    def train_gpu_model(*args, **kwargs):
        raise NotImplementedError("GPU training not available")

# Clean exports
__all__ = [
    "create_gguf_model",
    "train_gpu_model", 
    "run_training_pipeline",
    "optimize_model_size",
    "validate_gguf_quality",
    "generate_domain_model",
    "setup_gpu_environment",
    "monitor_training_progress"
] 
