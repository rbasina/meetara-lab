#!/usr/bin/env python3
"""
TARA Core Intelligence - Empathy Engine
Enhances responses with appropriate empathy and clarity based on emotional context
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

# Import emotion detector for emotional intelligence
from ..emotion_detector import EnhancedEmotionDetector

class TARAEmpathyEngine:
    """
    TARA's empathy engine for generating emotionally intelligent responses
    Provides appropriate empathy and clarity based on human emotional state and domain context
    """
    
    def __init__(self):
        # Empathy levels and response patterns
        self.empathy_levels = {
            "very_high": {
                "description": "Deep emotional support and validation",
                "response_patterns": [
                    "I can really understand how {emotion} you must be feeling.",
                    "It sounds like this is really {emotion_adjective} for you.",
                    "I want you to know that your feelings are completely valid.",
                    "You're not alone in feeling this way."
                ],
                "domains": ["mental_health", "crisis_intervention", "grief_support", "relationships"]
            },
            
            "high": {
                "description": "Warm, supportive, and understanding",
                "response_patterns": [
                    "I understand this can be {emotion_adjective}.",
                    "It's completely normal to feel {emotion} about this.",
                    "I'm here to help you through this.",
                    "Let's work together to address this."
                ],
                "domains": ["general_health", "parenting", "stress_management", "conflict_resolution"]
            },
            
            "moderate": {
                "description": "Professional yet caring",
                "response_patterns": [
                    "I can see this is important to you.",
                    "Let me help you with this concern.",
                    "I understand you're looking for guidance on this.",
                    "This is definitely something we can work on together."
                ],
                "domains": ["education", "career_guidance", "skill_development", "business"]
            },
            
            "professional": {
                "description": "Respectful, clear, and solution-focused",
                "response_patterns": [
                    "I'll help you find the best approach for this.",
                    "Let's focus on practical solutions.",
                    "Here's how we can address this effectively.",
                    "I'll provide you with clear, actionable guidance."
                ],
                "domains": ["legal", "financial", "technology", "consulting"]
            }
        }
        
        # Emotion-to-empathy mapping
        self.emotion_empathy_mapping = {
            "anxiety": {
                "empathy_level": "high",
                "response_tone": "calming",
                "validation_phrases": [
                    "Anxiety can feel overwhelming",
                    "It's okay to feel anxious about this",
                    "Many people experience anxiety in similar situations"
                ],
                "action_phrases": [
                    "Let's take this step by step",
                    "We'll work through this together",
                    "I'll help you find ways to manage this"
                ]
            },
            
            "frustration": {
                "empathy_level": "moderate",
                "response_tone": "understanding",
                "validation_phrases": [
                    "Frustration is a natural response to this",
                    "I can understand why this would be frustrating",
                    "It's completely reasonable to feel this way"
                ],
                "action_phrases": [
                    "Let's find a solution that works",
                    "I'll help you navigate this challenge",
                    "We can work together to resolve this"
                ]
            },
            
            "sadness": {
                "empathy_level": "very_high",
                "response_tone": "gentle",
                "validation_phrases": [
                    "I'm sorry you're going through this difficult time",
                    "It's important to acknowledge these feelings",
                    "Sadness is a natural part of processing difficult experiences"
                ],
                "action_phrases": [
                    "I'm here to support you",
                    "Let's explore ways to help you feel better",
                    "You don't have to go through this alone"
                ]
            },
            
            "confusion": {
                "empathy_level": "moderate",
                "response_tone": "clarifying",
                "validation_phrases": [
                    "It's okay to feel confused about this",
                    "This can be a complex topic to understand",
                    "Confusion often means you're learning something new"
                ],
                "action_phrases": [
                    "Let me help clarify this for you",
                    "I'll break this down into simpler steps",
                    "We'll work through this together until it makes sense"
                ]
            },
            
            "excitement": {
                "empathy_level": "moderate",
                "response_tone": "encouraging",
                "validation_phrases": [
                    "I can feel your enthusiasm about this",
                    "It's wonderful that you're excited about this opportunity",
                    "Your passion for this really comes through"
                ],
                "action_phrases": [
                    "Let's channel that excitement into action",
                    "I'll help you make the most of this opportunity",
                    "Together we can turn this enthusiasm into results"
                ]
            }
        }
        
        # Domain-specific empathy adjustments
        self.domain_empathy_adjustments = {
            "mental_health": {
                "base_empathy_boost": 0.3,
                "crisis_detection": True,
                "validation_emphasis": "very_high",
                "professional_boundaries": True
            },
            
            "emergency_care": {
                "base_empathy_boost": 0.2,
                "urgency_awareness": True,
                "calm_authority": True,
                "immediate_action_focus": True
            },
            
            "parenting": {
                "base_empathy_boost": 0.2,
                "non_judgmental": True,
                "supportive_guidance": True,
                "practical_solutions": True
            },
            
            "relationships": {
                "base_empathy_boost": 0.25,
                "emotional_validation": True,
                "perspective_taking": True,
                "conflict_sensitivity": True
            },
            
            "business": {
                "base_empathy_boost": 0.1,
                "professional_tone": True,
                "solution_focused": True,
                "efficiency_valued": True
            },
            
            "education": {
                "base_empathy_boost": 0.15,
                "encouraging": True,
                "patience_emphasis": True,
                "growth_mindset": True
            }
        }
        
        # Clarity enhancement patterns
        self.clarity_patterns = {
            "simple_explanation": [
                "Let me explain this in simple terms:",
                "Here's what this means in everyday language:",
                "To put it simply:",
                "The key point is:"
            ],
            
            "step_by_step": [
                "Let's break this down step by step:",
                "Here's how we can approach this:",
                "I'll walk you through this process:",
                "Let's tackle this one step at a time:"
            ],
            
            "reassurance": [
                "This is more common than you might think.",
                "You're asking exactly the right questions.",
                "This is a perfectly normal concern.",
                "Many people find this challenging at first."
            ],
            
            "action_oriented": [
                "Here's what you can do:",
                "Your next steps would be:",
                "I recommend starting with:",
                "The most effective approach is:"
            ]
        }
        
        # Performance tracking
        self.empathy_stats = {
            "responses_enhanced": 0,
            "emotion_detections": 0,
            "empathy_level_distribution": {},
            "domain_empathy_usage": {}
        }
    
    async def enhance_response_with_empathy(self, base_response: str, user_input: str, 
                                          detected_emotion: Dict[str, Any], 
                                          domain: str, urgency_level: str = "low") -> Dict[str, Any]:
        """
        Enhance a base response with appropriate empathy and clarity
        
        Args:
            base_response: The original response to enhance
            user_input: The original user input for context
            detected_emotion: Emotion detection results
            domain: The domain context
            urgency_level: Urgency level (low, medium, high, critical)
            
        Returns:
            Dict containing enhanced response and empathy metadata
        """
        
        # Step 1: Determine appropriate empathy level
        empathy_level = await self._determine_empathy_level(detected_emotion, domain, urgency_level)
        
        # Step 2: Generate empathetic opening
        empathetic_opening = await self._generate_empathetic_opening(
            detected_emotion, empathy_level, domain
        )
        
        # Step 3: Enhance response clarity
        clarity_enhanced = await self._enhance_clarity(base_response, detected_emotion, domain)
        
        # Step 4: Add appropriate emotional validation
        validation = await self._add_emotional_validation(detected_emotion, empathy_level, domain)
        
        # Step 5: Generate supportive closing
        supportive_closing = await self._generate_supportive_closing(
            detected_emotion, empathy_level, domain, urgency_level
        )
        
        # Step 6: Combine all elements
        enhanced_response = await self._combine_response_elements(
            empathetic_opening, clarity_enhanced, validation, supportive_closing
        )
        
        # Step 7: Apply domain-specific adjustments
        final_response = await self._apply_domain_adjustments(enhanced_response, domain, urgency_level)
        
        result = {
            "enhanced_response": final_response,
            "empathy_level": empathy_level,
            "emotional_tone": detected_emotion.get("primary_emotion", "neutral"),
            "domain_context": domain,
            "urgency_awareness": urgency_level,
            "enhancement_metadata": {
                "empathetic_opening": empathetic_opening,
                "validation_added": validation,
                "clarity_enhanced": True,
                "supportive_closing": supportive_closing
            },
            "enhancement_timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Update performance stats
        await self._update_empathy_stats(result, domain, empathy_level)
        
        return result
    
    async def _determine_empathy_level(self, detected_emotion: Dict[str, Any], 
                                     domain: str, urgency_level: str) -> str:
        """Determine the appropriate empathy level for the response"""
        
        primary_emotion = detected_emotion.get("primary_emotion", "neutral")
        emotion_intensity = detected_emotion.get("emotion_intensity", 0.5)
        
        # Start with emotion-based empathy level
        emotion_config = self.emotion_empathy_mapping.get(primary_emotion, {})
        base_empathy = emotion_config.get("empathy_level", "moderate")
        
        # Apply domain adjustments
        domain_config = self.domain_empathy_adjustments.get(domain, {})
        
        # Adjust for urgency
        if urgency_level in ["critical", "high"]:
            if domain in ["mental_health", "emergency_care", "crisis_intervention"]:
                return "very_high"
            else:
                return "high"
        
        # Adjust for emotion intensity
        if emotion_intensity > 0.8 and primary_emotion in ["sadness", "anxiety", "fear"]:
            return "very_high"
        elif emotion_intensity > 0.6 and primary_emotion in ["frustration", "anger", "stress"]:
            return "high"
        
        # Domain-specific defaults
        if domain in ["mental_health", "crisis_intervention", "grief_support"]:
            return "very_high"
        elif domain in ["general_health", "parenting", "relationships"]:
            return "high"
        elif domain in ["business", "legal", "financial"]:
            return "professional"
        else:
            return base_empathy
    
    async def _generate_empathetic_opening(self, detected_emotion: Dict[str, Any], 
                                         empathy_level: str, domain: str) -> str:
        """Generate an empathetic opening for the response"""
        
        primary_emotion = detected_emotion.get("primary_emotion", "neutral")
        emotion_config = self.emotion_empathy_mapping.get(primary_emotion, {})
        empathy_config = self.empathy_levels.get(empathy_level, {})
        
        # Select appropriate validation phrase
        validation_phrases = emotion_config.get("validation_phrases", [])
        if validation_phrases:
            return validation_phrases[0]  # Use the first/primary validation phrase
        
        # Fallback to empathy level patterns
        response_patterns = empathy_config.get("response_patterns", [])
        if response_patterns:
            # Simple template replacement
            pattern = response_patterns[0]
            return pattern.replace("{emotion}", primary_emotion).replace("{emotion_adjective}", f"{primary_emotion}")
        
        return "I understand you're reaching out for help with this."
    
    async def _enhance_clarity(self, base_response: str, detected_emotion: Dict[str, Any], 
                             domain: str) -> str:
        """Enhance response clarity based on emotional state and domain"""
        
        primary_emotion = detected_emotion.get("primary_emotion", "neutral")
        
        # If user is confused, add clarity enhancement
        if primary_emotion == "confusion":
            clarity_intro = self.clarity_patterns["simple_explanation"][0]
            return f"{clarity_intro}\n\n{base_response}"
        
        # If user is anxious, break into steps
        elif primary_emotion in ["anxiety", "overwhelm"]:
            step_intro = self.clarity_patterns["step_by_step"][0]
            return f"{step_intro}\n\n{base_response}"
        
        # If user is frustrated, add reassurance
        elif primary_emotion in ["frustration", "anger"]:
            reassurance = self.clarity_patterns["reassurance"][0]
            return f"{reassurance}\n\n{base_response}"
        
        return base_response
    
    async def _add_emotional_validation(self, detected_emotion: Dict[str, Any], 
                                      empathy_level: str, domain: str) -> str:
        """Add appropriate emotional validation"""
        
        primary_emotion = detected_emotion.get("primary_emotion", "neutral")
        emotion_config = self.emotion_empathy_mapping.get(primary_emotion, {})
        
        validation_phrases = emotion_config.get("validation_phrases", [])
        
        if empathy_level in ["very_high", "high"] and validation_phrases:
            return validation_phrases[0]
        
        return ""
    
    async def _generate_supportive_closing(self, detected_emotion: Dict[str, Any], 
                                         empathy_level: str, domain: str, 
                                         urgency_level: str) -> str:
        """Generate a supportive closing for the response"""
        
        primary_emotion = detected_emotion.get("primary_emotion", "neutral")
        emotion_config = self.emotion_empathy_mapping.get(primary_emotion, {})
        
        action_phrases = emotion_config.get("action_phrases", [])
        
        if urgency_level in ["critical", "high"]:
            return "I'm here to help you through this. Please don't hesitate to reach out if you need immediate support."
        
        if empathy_level == "very_high":
            return "Remember, you're not alone in this. I'm here to support you every step of the way."
        elif empathy_level == "high":
            return "I'm here to help you work through this. Feel free to ask if you need any clarification."
        elif empathy_level == "professional":
            return "I hope this information is helpful. Please let me know if you need any additional guidance."
        
        if action_phrases:
            return action_phrases[0]
        
        return "I'm here to help if you have any other questions."
    
    async def _combine_response_elements(self, empathetic_opening: str, 
                                       clarity_enhanced: str, validation: str, 
                                       supportive_closing: str) -> str:
        """Combine all response elements into a cohesive response"""
        
        elements = []
        
        if empathetic_opening:
            elements.append(empathetic_opening)
        
        if validation and validation != empathetic_opening:
            elements.append(validation)
        
        elements.append(clarity_enhanced)
        
        if supportive_closing:
            elements.append(supportive_closing)
        
        return "\n\n".join(elements)
    
    async def _apply_domain_adjustments(self, response: str, domain: str, 
                                      urgency_level: str) -> str:
        """Apply domain-specific adjustments to the response"""
        
        domain_config = self.domain_empathy_adjustments.get(domain, {})
        
        # Emergency care adjustments
        if domain == "emergency_care" and urgency_level in ["critical", "high"]:
            emergency_prefix = "⚠️ If this is a medical emergency, please call 911 immediately.\n\n"
            return emergency_prefix + response
        
        # Mental health adjustments
        elif domain == "mental_health" and urgency_level in ["critical", "high"]:
            crisis_prefix = "🆘 If you're having thoughts of self-harm, please contact the National Suicide Prevention Lifeline: 988\n\n"
            return crisis_prefix + response
        
        return response
    
    async def _update_empathy_stats(self, result: Dict[str, Any], domain: str, empathy_level: str):
        """Update empathy engine performance statistics"""
        self.empathy_stats["responses_enhanced"] += 1
        
        # Track empathy level distribution
        if empathy_level not in self.empathy_stats["empathy_level_distribution"]:
            self.empathy_stats["empathy_level_distribution"][empathy_level] = 0
        self.empathy_stats["empathy_level_distribution"][empathy_level] += 1
        
        # Track domain usage
        if domain not in self.empathy_stats["domain_empathy_usage"]:
            self.empathy_stats["domain_empathy_usage"][domain] = 0
        self.empathy_stats["domain_empathy_usage"][domain] += 1
    
    def get_empathy_statistics(self) -> Dict[str, Any]:
        """Get empathy engine performance statistics"""
        return {
            "total_responses_enhanced": self.empathy_stats["responses_enhanced"],
            "empathy_level_distribution": self.empathy_stats["empathy_level_distribution"],
            "domain_usage": self.empathy_stats["domain_empathy_usage"],
            "available_empathy_levels": list(self.empathy_levels.keys()),
            "supported_emotions": list(self.emotion_empathy_mapping.keys())
        }

# Convenience function for easy integration
async def enhance_response_with_empathy(base_response: str, user_input: str, 
                                      detected_emotion: Dict[str, Any], 
                                      domain: str, urgency_level: str = "low") -> str:
    """
    Convenience function to enhance a response with empathy and clarity
    
    Usage:
        enhanced = await enhance_response_with_empathy(
            base_response="Here's how to solve your problem...",
            user_input="I'm really stressed about this coding issue",
            detected_emotion={"primary_emotion": "stress", "emotion_intensity": 0.7},
            domain="programming",
            urgency_level="medium"
        )
    """
    empathy_engine = TARAEmpathyEngine()
    result = await empathy_engine.enhance_response_with_empathy(
        base_response, user_input, detected_emotion, domain, urgency_level
    )
    return result["enhanced_response"] 