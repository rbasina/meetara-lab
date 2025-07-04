#!/usr/bin/env python3
"""
TARA Core Intelligence Module
Provides foundational intelligence capabilities for domain-agnostic, empathetic AI

This module contains TARA's core intelligence components:
- Language Foundation: Understanding of letters, numbers, characters, languages
- Domain Detector: Intelligent detection of what domain humans need
- Empathy Engine: Emotionally intelligent response enhancement
- Context Manager: Conversation context and continuity management
"""

# Core intelligence imports
from .language_foundation import (
    TARALanguageFoundation,
    analyze_text_foundation
)

from .domain_detector import (
    TARACoreIntelligence,
    detect_domain_for_human
)

from .empathy_engine import (
    TARAEmpathyEngine,
    enhance_response_with_empathy
)

from .context_manager import (
    TARAContextManager,
    manage_conversation_context
)

# Version information
__version__ = "1.0.0"
__author__ = "MeeTARA Lab"
__description__ = "TARA Core Intelligence - Domain-agnostic, empathetic AI intelligence"

# Main intelligence coordinator class
class TARAIntelligenceCoordinator:
    """
    Coordinates all TARA intelligence modules for seamless operation
    Provides a unified interface to all intelligence capabilities
    """
    
    def __init__(self):
        # Initialize all intelligence modules
        self.language_foundation = TARALanguageFoundation()
        self.domain_detector = TARACoreIntelligence()
        self.empathy_engine = TARAEmpathyEngine()
        self.context_manager = TARAContextManager()
        
        # Intelligence coordination settings
        self.coordination_enabled = True
        self.intelligence_level = "enhanced"  # basic, standard, enhanced
        
    async def process_human_input(self, user_input: str, context: str = "",
                                conversation_history: list = None) -> dict:
        """
        Process human input through all intelligence modules
        
        Args:
            user_input: The human's message
            context: Additional context information
            conversation_history: Previous conversation for context
            
        Returns:
            Dict containing comprehensive intelligence analysis
        """
        
        # Step 1: Language Foundation Analysis
        foundation_analysis = await self.language_foundation.analyze_text_foundation(user_input)
        
        # Step 2: Domain Detection
        domain_analysis = await self.domain_detector.detect_domain_intelligent(
            user_input, context, conversation_history
        )
        
        # Step 3: Get relevant context
        relevant_context = await self.context_manager.get_relevant_context(user_input)
        
        # Combine all intelligence insights
        intelligence_result = {
            "user_input": user_input,
            "foundation_analysis": foundation_analysis,
            "domain_analysis": domain_analysis,
            "relevant_context": relevant_context,
            "processing_timestamp": foundation_analysis["foundation_timestamp"],
            "intelligence_level": self.intelligence_level,
            "success": True
        }
        
        return intelligence_result
    
    async def enhance_response(self, base_response: str, user_input: str,
                             detected_emotion: dict, domain: str,
                             urgency_level: str = "low") -> dict:
        """
        Enhance a response with empathy and context awareness
        
        Args:
            base_response: The base response to enhance
            user_input: Original user input
            detected_emotion: Emotion detection results
            domain: Detected domain
            urgency_level: Urgency level
            
        Returns:
            Dict containing enhanced response and metadata
        """
        
        # Enhance with empathy
        empathy_result = await self.empathy_engine.enhance_response_with_empathy(
            base_response, user_input, detected_emotion, domain, urgency_level
        )
        
        # Update context
        context_result = await self.context_manager.add_conversation_turn(
            user_input, empathy_result["enhanced_response"], 
            detected_emotion, domain, urgency_level
        )
        
        return {
            "enhanced_response": empathy_result["enhanced_response"],
            "empathy_metadata": empathy_result["enhancement_metadata"],
            "context_update": context_result["context_analysis"],
            "intelligence_coordination": True,
            "success": True
        }
    
    def get_intelligence_status(self) -> dict:
        """Get status of all intelligence modules"""
        return {
            "language_foundation": self.language_foundation.get_foundation_statistics(),
            "domain_detector": self.domain_detector.get_detection_statistics(),
            "empathy_engine": self.empathy_engine.get_empathy_statistics(),
            "context_manager": self.context_manager.get_context_statistics(),
            "coordination_enabled": self.coordination_enabled,
            "intelligence_level": self.intelligence_level
        }

# Convenience functions for easy integration
async def analyze_human_input(user_input: str, context: str = "",
                            conversation_history: list = None) -> dict:
    """
    Convenience function to analyze human input with all intelligence modules
    
    Usage:
        result = await analyze_human_input("I'm feeling stressed about my health")
        domain = result["domain_analysis"]["primary_domain"]
        emotion = result["foundation_analysis"]["intent_analysis"]["primary_intent"]
    """
    coordinator = TARAIntelligenceCoordinator()
    return await coordinator.process_human_input(user_input, context, conversation_history)

async def create_empathetic_response(base_response: str, user_input: str,
                                   detected_emotion: dict, domain: str,
                                   urgency_level: str = "low") -> str:
    """
    Convenience function to create an empathetic response
    
    Usage:
        enhanced_response = await create_empathetic_response(
            base_response="Here's how to manage stress...",
            user_input="I'm really stressed about work",
            detected_emotion={"primary_emotion": "stress", "emotion_intensity": 0.8},
            domain="stress_management",
            urgency_level="medium"
        )
    """
    coordinator = TARAIntelligenceCoordinator()
    result = await coordinator.enhance_response(
        base_response, user_input, detected_emotion, domain, urgency_level
    )
    return result["enhanced_response"]

# Export all components for direct access
__all__ = [
    # Core classes
    "TARALanguageFoundation",
    "TARACoreIntelligence", 
    "TARAEmpathyEngine",
    "TARAContextManager",
    "TARAIntelligenceCoordinator",
    
    # Convenience functions
    "analyze_text_foundation",
    "detect_domain_for_human",
    "enhance_response_with_empathy",
    "manage_conversation_context",
    "analyze_human_input",
    "create_empathetic_response",
    
    # Module metadata
    "__version__",
    "__author__",
    "__description__"
] 