"""
MeeTARA Lab - Modern Import Structure
Provides clean imports from existing scattered directories without breaking changes.
"""

# Version info
__version__ = "1.0.0"
__title__ = "MeeTARA Lab"
__description__ = "Trinity Architecture AI Training Evolution"

# Import wrappers for backward compatibility
from src.models import train_gpu_model, create_gguf_model
from src.intelligence import route_request, detect_emotion, manage_voice
from src.data import generate_training_data, validate_data

# Main entry points
from src.main import run_training_pipeline, run_inference_pipeline

__all__ = [
    # Training functions
    "train_gpu_model",
    "create_gguf_model", 
    "generate_training_data",
    
    # Intelligence functions
    "route_request",
    "detect_emotion", 
    "manage_voice",
    
    # Main pipelines
    "run_training_pipeline",
    "run_inference_pipeline",
    
    # Validation
    "validate_data",
] 