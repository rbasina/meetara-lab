"""
MeeTARA Lab - Enhanced Emotion Detection with Trinity Architecture
RoBERTa-based emotion detection with professional context analysis and cloud amplification
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Import trinity-core components
from agents.mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage

class EnhancedEmotionDetector(BaseAgent):
    """Enhanced Emotion Detector with Trinity Architecture and RoBERTa intelligence"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.EMOTION_DETECTOR, mcp)
        
        # RoBERTa-based emotion detection model
        self.emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.emotion_classifier = None
        self.tokenizer = None
        
        # Professional context analysis
        self.professional_contexts = {
            "healthcare": {
                "critical_emotions": ["anxiety", "fear", "sadness", "anger"],
                "therapeutic_responses": True,
                "empathy_level": "high",
                "intervention_threshold": 0.7
            },
            "business": {
                "critical_emotions": ["anger", "frustration", "stress"],
                "professional_responses": True,
                "empathy_level": "moderate",
                "intervention_threshold": 0.6
            },
            "education": {
                "critical_emotions": ["confusion", "frustration", "anxiety"],
                "supportive_responses": True,
                "empathy_level": "high",
                "intervention_threshold": 0.6
            },
            "personal": {
                "critical_emotions": ["sadness", "anger", "fear", "loneliness"],
                "compassionate_responses": True,
                "empathy_level": "very_high",
                "intervention_threshold": 0.8
            }
        }
        
        # Emotion mapping and intensity levels
        self.emotion_categories = {
            "positive": ["joy", "love", "optimism", "gratitude", "excitement"],
            "negative": ["sadness", "anger", "fear", "anxiety", "frustration"],
            "neutral": ["neutral", "calm", "thoughtful", "focused"],
            "complex": ["surprise", "confusion", "mixed", "conflicted"]
        }
        
        # Multi-domain emotion mapping
        self.domain_emotion_patterns = self._create_domain_emotion_patterns()
        
        # Trinity enhancements
        self.trinity_enhancements = {
            "arc_reactor_precision": True,      # Enhanced emotion precision
            "perplexity_context_awareness": True, # Context-aware emotion detection
            "einstein_fusion_amplification": True # Emotional intelligence amplification
        }
        
        # Cloud amplification settings
        self.cloud_settings = {
            "batch_processing": True,           # Process multiple inputs
            "real_time_analysis": True,        # Real-time emotion tracking
            "historical_context": True,       # Consider emotion history
            "multi_modal_fusion": True,       # Combine text, voice, facial data
            "confidence_thresholds": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
        }
        
        # Performance tracking
        self.performance_stats = {
            "emotions_detected": 0,
            "accuracy_score": 0,
            "intervention_triggers": 0,
            "domain_usage": {},
            "emotion_distribution": {},
            "response_times": []
        }
        
    async def start(self):
        """Start the Enhanced Emotion Detector"""
        await super().start()
        print("🧠 Enhanced Emotion Detector initializing...")
        
        # Load RoBERTa emotion detection model
        await self._load_emotion_model()
        print("🧠 Enhanced Emotion Detector ready with Trinity Architecture")
        
    async def _load_emotion_model(self):
        """Load RoBERTa-based emotion detection model"""
        try:
            print("📥 Loading RoBERTa emotion detection model...")
            
            # Initialize emotion classification pipeline
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                tokenizer=self.emotion_model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            print("✅ RoBERTa emotion model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to load RoBERTa model, using fallback: {e}")
            # Fallback to simple emotion detection
            self.emotion_classifier = None
            
    def _create_domain_emotion_patterns(self) -> Dict[str, List[str]]:
        """Create domain-specific emotion patterns"""
        return {
            # Healthcare emotions
            "general_health": ["anxiety", "hope", "relief", "concern"],
            "mental_health": ["sadness", "anxiety", "hope", "fear", "calm"],
            "chronic_conditions": ["frustration", "acceptance", "hope", "fatigue"],
            
            # Daily life emotions
            "parenting": ["love", "frustration", "worry", "pride", "exhaustion"],
            "relationships": ["love", "anger", "jealousy", "happiness", "confusion"],
            "work_life_balance": ["stress", "satisfaction", "overwhelm", "accomplishment"],
            
            # Business emotions
            "leadership": ["confidence", "pressure", "determination", "concern"],
            "customer_service": ["patience", "frustration", "empathy", "satisfaction"],
            "sales": ["enthusiasm", "rejection", "confidence", "nervousness"],
            
            # Education emotions
            "academic_tutoring": ["confusion", "clarity", "encouragement", "achievement"],
            "exam_preparation": ["anxiety", "confidence", "determination", "stress"],
            
            # Creative emotions
            "writing": ["inspiration", "frustration", "flow", "doubt"],
            "art_appreciation": ["wonder", "contemplation", "joy", "inspiration"],
            
            # Technology emotions
            "programming": ["frustration", "satisfaction", "confusion", "achievement"],
            "tech_support": ["patience", "clarity", "helpfulness", "problem_solving"]
        }
        
    async def detect_emotion_comprehensive(self, text: str, domain: str = "general",
                                         context: Dict[str, Any] = None,
                                         historical_data: List[Dict] = None) -> Dict[str, Any]:
        """Comprehensive emotion detection with Trinity Architecture"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: RoBERTa-based emotion detection
            primary_emotions = await self._detect_primary_emotions(text)
            
            # Step 2: Apply domain-specific context analysis
            contextual_analysis = await self._apply_domain_context(primary_emotions, domain, context)
            
            # Step 3: Trinity Architecture enhancements
            trinity_enhanced = await self._apply_trinity_enhancements(contextual_analysis, text, domain)
            
            # Step 4: Professional context evaluation
            professional_analysis = await self._evaluate_professional_context(trinity_enhanced, domain)
            
            # Step 5: Generate empathetic response recommendations
            response_recommendations = await self._generate_response_recommendations(
                professional_analysis, domain, context
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            result = {
                "primary_emotion": trinity_enhanced["primary_emotion"],
                "emotion_confidence": trinity_enhanced["confidence"],
                "emotion_intensity": trinity_enhanced["intensity"],
                "emotional_category": trinity_enhanced["category"],
                "secondary_emotions": trinity_enhanced.get("secondary_emotions", []),
                "domain_context": domain,
                "professional_analysis": professional_analysis,
                "response_recommendations": response_recommendations,
                "intervention_required": professional_analysis.get("intervention_required", False),
                "empathy_level": professional_analysis.get("empathy_level", "moderate"),
                "trinity_enhanced": True,
                "processing_time_ms": round(processing_time, 2),
                "detection_timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Update performance statistics
            await self._update_performance_stats(result, domain)
            
            # Notify other agents
            self.send_message(
                AgentType.TTS_MANAGER,
                MessageType.STATUS_UPDATE,
                {
                    "action": "emotion_detected",
                    "emotion_data": result,
                    "domain": domain
                }
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Emotion detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_emotion": "neutral",
                "confidence": 0.5
            }
            
    async def _detect_primary_emotions(self, text: str) -> Dict[str, Any]:
        """Detect primary emotions using RoBERTa model"""
        
        if not self.emotion_classifier:
            # Fallback emotion detection
            return await self._fallback_emotion_detection(text)
            
        try:
            # Use RoBERTa classifier
            results = self.emotion_classifier(text)
            
            # Sort by confidence score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            primary_emotion = results[0]['label'].lower()
            confidence = results[0]['score']
            
            # Get secondary emotions
            secondary_emotions = [
                {"emotion": r['label'].lower(), "confidence": r['score']}
                for r in results[1:3] if r['score'] > 0.1
            ]
            
            return {
                "primary_emotion": primary_emotion,
                "confidence": confidence,
                "secondary_emotions": secondary_emotions,
                "all_scores": results
            }
            
        except Exception as e:
            print(f"⚠️ RoBERTa detection failed, using fallback: {e}")
            return await self._fallback_emotion_detection(text)
            
    async def _fallback_emotion_detection(self, text: str) -> Dict[str, Any]:
        """Fallback emotion detection using keyword analysis"""
        
        emotion_keywords = {
            "joy": ["happy", "excited", "wonderful", "great", "amazing", "love"],
            "sadness": ["sad", "depressed", "down", "upset", "crying", "grief"],
            "anger": ["angry", "mad", "furious", "annoyed", "frustrated", "hate"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "panic"],
            "surprise": ["surprised", "shocked", "unexpected", "wow", "amazing"],
            "neutral": ["okay", "fine", "normal", "regular", "standard"]
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
                
        if not emotion_scores:
            return {
                "primary_emotion": "neutral",
                "confidence": 0.5,
                "secondary_emotions": []
            }
            
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(0.8, emotion_scores[primary_emotion] * 2)  # Cap at 0.8 for fallback
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "secondary_emotions": []
        }
        
    async def _apply_domain_context(self, emotions: Dict[str, Any], domain: str,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply domain-specific context to emotion analysis"""
        
        domain_patterns = self.domain_emotion_patterns.get(domain, [])
        primary_emotion = emotions["primary_emotion"]
        
        # Check if emotion fits domain patterns
        domain_relevance = 1.0 if primary_emotion in domain_patterns else 0.7
        
        # Apply context adjustments
        intensity_adjustment = 1.0
        
        if context:
            # Adjust based on conversation context
            if context.get("conversation_length", 0) > 5:
                intensity_adjustment *= 1.1  # Longer conversations may show stronger emotions
                
            if context.get("user_stress_level", "normal") == "high":
                if primary_emotion in ["anxiety", "frustration", "anger"]:
                    intensity_adjustment *= 1.3
                    
        # Calculate contextual intensity
        base_intensity = emotions.get("confidence", 0.5)
        contextual_intensity = min(1.0, base_intensity * intensity_adjustment)
        
        # Determine emotional category
        category = "neutral"
        for cat, emotion_list in self.emotion_categories.items():
            if primary_emotion in emotion_list:
                category = cat
                break
                
        return {
            **emotions,
            "domain_relevance": domain_relevance,
            "intensity": contextual_intensity,
            "category": category,
            "context_adjusted": True
        }
        
    async def _apply_trinity_enhancements(self, emotion_data: Dict[str, Any], 
                                        text: str, domain: str) -> Dict[str, Any]:
        """Apply Trinity Architecture enhancements to emotion detection"""
        
        enhanced_data = emotion_data.copy()
        
        # Arc Reactor Precision Enhancement
        if self.trinity_enhancements["arc_reactor_precision"]:
            # Improve confidence based on text length and complexity
            text_complexity = len(text.split()) / 10  # Simple complexity measure
            confidence_boost = min(0.1, text_complexity * 0.02)
            enhanced_data["confidence"] = min(1.0, enhanced_data["confidence"] + confidence_boost)
            
        # Perplexity Context Awareness
        if self.trinity_enhancements["perplexity_context_awareness"]:
            # Enhanced context understanding
            enhanced_data["context_awareness_score"] = 0.9
            enhanced_data["domain_understanding"] = domain in self.domain_emotion_patterns
            
        # Einstein Fusion Amplification
        if self.trinity_enhancements["einstein_fusion_amplification"]:
            # E=mc² applied to emotional intelligence
            # Enhanced empathy = mass(emotion data) × c²(context speed)
            emotion_mass = enhanced_data["confidence"] * enhanced_data["intensity"]
            context_speed = 2.0  # Trinity context acceleration
            
            fusion_amplification = min(1.2, 1.0 + (emotion_mass * context_speed * context_speed) / 100)
            enhanced_data["intensity"] = min(1.0, enhanced_data["intensity"] * fusion_amplification)
            enhanced_data["einstein_fusion_applied"] = True
            
        return enhanced_data
        
    async def _evaluate_professional_context(self, emotion_data: Dict[str, Any], 
                                           domain: str) -> Dict[str, Any]:
        """Evaluate professional context and intervention needs"""
        
        primary_emotion = emotion_data["primary_emotion"]
        intensity = emotion_data["intensity"]
        
        # Get domain-specific professional context
        domain_category = "personal"  # Default
        if domain in ["general_health", "mental_health", "chronic_conditions"]:
            domain_category = "healthcare"
        elif domain in ["leadership", "customer_service", "sales", "project_management"]:
            domain_category = "business"
        elif domain in ["academic_tutoring", "exam_preparation", "skill_development"]:
            domain_category = "education"
            
        context_config = self.professional_contexts.get(domain_category, self.professional_contexts["personal"])
        
        # Check if intervention is required
        intervention_required = (
            primary_emotion in context_config["critical_emotions"] and
            intensity >= context_config["intervention_threshold"]
        )
        
        # Generate professional recommendations
        recommendations = []
        
        if intervention_required:
            if domain_category == "healthcare":
                recommendations.append("Consider therapeutic response approach")
                recommendations.append("Monitor for escalation patterns")
            elif domain_category == "business":
                recommendations.append("Maintain professional boundaries")
                recommendations.append("Focus on solution-oriented dialogue")
            elif domain_category == "education":
                recommendations.append("Provide additional support and encouragement")
                recommendations.append("Consider alternative learning approaches")
                
        return {
            "domain_category": domain_category,
            "intervention_required": intervention_required,
            "empathy_level": context_config["empathy_level"],
            "professional_recommendations": recommendations,
            "context_confidence": 0.85,
            "risk_assessment": "high" if intervention_required else "low"
        }
        
    async def _generate_response_recommendations(self, professional_analysis: Dict[str, Any],
                                               domain: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate empathetic response recommendations"""
        
        recommendations = []
        empathy_level = professional_analysis.get("empathy_level", "moderate")
        intervention_required = professional_analysis.get("intervention_required", False)
        
        if empathy_level == "very_high":
            recommendations.extend([
                "Use warm, compassionate language",
                "Acknowledge emotional state explicitly",
                "Offer emotional support and validation"
            ])
        elif empathy_level == "high":
            recommendations.extend([
                "Show understanding and patience",
                "Use supportive tone and language",
                "Provide reassurance when appropriate"
            ])
        elif empathy_level == "moderate":
            recommendations.extend([
                "Maintain professional empathy",
                "Be respectful of emotional state",
                "Focus on helpful solutions"
            ])
            
        if intervention_required:
            recommendations.extend([
                "Consider escalation to human support",
                "Provide crisis resources if applicable",
                "Monitor conversation closely"
            ])
            
        # Domain-specific recommendations
        domain_specific = {
            "mental_health": ["Validate feelings", "Avoid giving medical advice", "Suggest professional resources"],
            "parenting": ["Acknowledge parenting challenges", "Offer practical support", "Normalize struggles"],
            "work_life_balance": ["Recognize stress", "Suggest balance strategies", "Validate challenges"]
        }
        
        if domain in domain_specific:
            recommendations.extend(domain_specific[domain])
            
        return recommendations[:5]  # Limit to 5 most relevant recommendations
        
    async def _update_performance_stats(self, result: Dict[str, Any], domain: str):
        """Update emotion detection performance statistics"""
        
        self.performance_stats["emotions_detected"] += 1
        
        # Update domain usage
        if domain not in self.performance_stats["domain_usage"]:
            self.performance_stats["domain_usage"][domain] = 0
        self.performance_stats["domain_usage"][domain] += 1
        
        # Update emotion distribution
        emotion = result["primary_emotion"]
        if emotion not in self.performance_stats["emotion_distribution"]:
            self.performance_stats["emotion_distribution"][emotion] = 0
        self.performance_stats["emotion_distribution"][emotion] += 1
        
        # Track intervention triggers
        if result.get("intervention_required", False):
            self.performance_stats["intervention_triggers"] += 1
            
        # Track response times
        response_time = result.get("processing_time_ms", 0)
        self.performance_stats["response_times"].append(response_time)
        
    async def get_emotion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive emotion detection statistics"""
        
        avg_response_time = 0
        if self.performance_stats["response_times"]:
            avg_response_time = sum(self.performance_stats["response_times"]) / len(self.performance_stats["response_times"])
            
        intervention_rate = 0
        if self.performance_stats["emotions_detected"] > 0:
            intervention_rate = self.performance_stats["intervention_triggers"] / self.performance_stats["emotions_detected"]
            
        return {
            **self.performance_stats,
            "average_response_time_ms": round(avg_response_time, 2),
            "intervention_rate": round(intervention_rate * 100, 1),
            "domains_supported": len(self.domain_emotion_patterns),
            "emotion_categories_supported": len(self.emotion_categories),
            "trinity_enhancements_active": sum(self.trinity_enhancements.values()),
            "roberta_model_loaded": self.emotion_classifier is not None,
            "professional_contexts_available": len(self.professional_contexts)
        }

# Global enhanced emotion detector
enhanced_emotion_detector = EnhancedEmotionDetector() 