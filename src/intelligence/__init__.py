"""
Intelligence Module - Clean imports from trinity-core directory
Wraps existing AI functionality without breaking changes.
"""

import sys
from pathlib import Path

# Add trinity-core to path for importing
project_root = Path(__file__).parent.parent.parent
trinity_core_path = project_root / "trinity-core"
intelligence_hub_path = project_root / "intelligence-hub"
sys.path.insert(0, str(trinity_core_path))
sys.path.insert(0, str(intelligence_hub_path))

# Import from existing files with clean naming
try:
    from intelligent_router import (
        route_request,
        intelligent_routing,
        process_query
    )
    
    from emotion_detector import (
        detect_emotion,
        analyze_emotional_state,
        process_audio_emotion
    )
    
    from tts_manager import (
        manage_voice,
        synthesize_speech,
        configure_voice_settings
    )
    
    from domain_experts import (
        get_domain_expert,
        coordinate_experts,
        expert_consultation
    )
    
    from trinity_intelligence import (
        trinity_processing,
        enhanced_reasoning,
        context_analysis
    )
    
except ImportError as e:
    print(f"Warning: Could not import from trinity-core/intelligence-hub: {e}")
    # Provide fallback functions
    def route_request(*args, **kwargs):
        raise NotImplementedError("Request routing not available")
    
    def detect_emotion(*args, **kwargs):
        raise NotImplementedError("Emotion detection not available")
    
    def manage_voice(*args, **kwargs):
        raise NotImplementedError("Voice management not available")

# Clean exports
__all__ = [
    "route_request",
    "detect_emotion",
    "manage_voice",
    "intelligent_routing",
    "analyze_emotional_state", 
    "synthesize_speech",
    "get_domain_expert",
    "coordinate_experts",
    "trinity_processing",
    "enhanced_reasoning",
    "context_analysis"
] 