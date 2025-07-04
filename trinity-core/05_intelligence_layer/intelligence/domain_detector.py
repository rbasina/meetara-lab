#!/usr/bin/env python3
"""
TARA Core Intelligence - Domain Detector
Intelligently detects what domain the human needs based on context, intent, and patterns
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime

# Import domain integration for dynamic domain loading
from ..domain_integration import get_all_domains, get_domain_categories, validate_domain

class TARACoreIntelligence:
    """
    TARA's core intelligence for domain detection
    Understands human intent and intelligently routes to appropriate domains
    """
    
    def __init__(self):
        # Load all available domains dynamically
        self.all_domains = get_all_domains()
        self.domain_categories = get_domain_categories()
        
        # Intent patterns for intelligent domain detection
        self.intent_patterns = {
            "health_seeking": {
                "patterns": [
                    r"\b(feel|feeling|pain|hurt|sick|ill|health|medical|doctor|symptom)\b",
                    r"\b(headache|fever|tired|exhausted|stressed|anxious|depressed)\b",
                    r"\b(medicine|medication|treatment|therapy|hospital|clinic)\b"
                ],
                "domains": ["general_health", "mental_health", "emergency_care", "stress_management"],
                "urgency_indicators": ["emergency", "urgent", "severe", "can't", "help"]
            },
            
            "learning_seeking": {
                "patterns": [
                    r"\b(learn|study|understand|explain|teach|education|school|course)\b",
                    r"\b(how to|what is|can you explain|help me understand)\b",
                    r"\b(homework|assignment|exam|test|grade|academic)\b"
                ],
                "domains": ["academic_tutoring", "skill_development", "career_guidance", "exam_preparation"],
                "urgency_indicators": ["deadline", "tomorrow", "urgent", "due"]
            },
            
            "problem_solving": {
                "patterns": [
                    r"\b(problem|issue|trouble|error|bug|fix|solve|help)\b",
                    r"\b(not working|broken|failed|wrong|confused)\b",
                    r"\b(technical|programming|code|software|computer)\b"
                ],
                "domains": ["tech_support", "programming", "cybersecurity", "software_development"],
                "urgency_indicators": ["critical", "emergency", "urgent", "down", "broken"]
            },
            
            "emotional_support": {
                "patterns": [
                    r"\b(feel|feeling|emotion|sad|happy|angry|frustrated|overwhelmed)\b",
                    r"\b(relationship|family|friend|partner|conflict|argument)\b",
                    r"\b(support|advice|talk|listen|understand|empathy)\b"
                ],
                "domains": ["mental_health", "relationships", "conflict_resolution", "parenting"],
                "urgency_indicators": ["crisis", "emergency", "suicidal", "harm"]
            },
            
            "business_guidance": {
                "patterns": [
                    r"\b(business|company|startup|entrepreneur|marketing|sales)\b",
                    r"\b(revenue|profit|customer|client|strategy|planning)\b",
                    r"\b(team|leadership|management|operations|finance)\b"
                ],
                "domains": ["entrepreneurship", "marketing", "sales", "team_leadership", "financial_planning"],
                "urgency_indicators": ["crisis", "urgent", "deadline", "critical"]
            },
            
            "creative_expression": {
                "patterns": [
                    r"\b(write|writing|story|creative|art|design|music)\b",
                    r"\b(inspiration|idea|brainstorm|create|imagine)\b",
                    r"\b(content|blog|social media|photography)\b"
                ],
                "domains": ["writing", "storytelling", "content_creation", "social_media", "design_thinking"],
                "urgency_indicators": ["deadline", "due", "urgent"]
            }
        }
        
        # Context clues for better domain detection
        self.context_clues = {
            "time_indicators": {
                "urgent": ["now", "immediately", "asap", "urgent", "emergency"],
                "scheduled": ["tomorrow", "next week", "deadline", "due"],
                "ongoing": ["always", "usually", "often", "regularly"]
            },
            
            "emotional_indicators": {
                "distress": ["worried", "scared", "anxious", "panicked", "overwhelmed"],
                "curiosity": ["wondering", "curious", "interested", "exploring"],
                "frustration": ["frustrated", "annoyed", "stuck", "confused"],
                "excitement": ["excited", "enthusiastic", "motivated", "inspired"]
            },
            
            "complexity_indicators": {
                "simple": ["basic", "simple", "easy", "quick", "brief"],
                "moderate": ["detailed", "comprehensive", "thorough"],
                "complex": ["advanced", "complex", "sophisticated", "expert"]
            }
        }
        
        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "multi_domain_detections": 0,
            "high_confidence_detections": 0
        }
        
    async def detect_domain_intelligent(self, user_input: str, context: str = "", 
                                      conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Intelligently detect what domain the human needs
        
        Args:
            user_input: The human's message/request
            context: Additional context information
            conversation_history: Previous conversation for context
            
        Returns:
            Dict containing domain detection results with confidence and reasoning
        """
        
        # Combine all available text for analysis
        full_text = f"{context} {user_input}".strip()
        if conversation_history:
            full_text += " " + " ".join(conversation_history[-3:])  # Last 3 messages for context
        
        # Step 1: Intent Analysis
        intent_analysis = await self._analyze_intent(full_text)
        
        # Step 2: Domain Scoring
        domain_scores = await self._calculate_domain_scores(full_text, intent_analysis)
        
        # Step 3: Context Enhancement
        context_enhanced = await self._enhance_with_context(domain_scores, full_text)
        
        # Step 4: Urgency Detection
        urgency_analysis = await self._detect_urgency(full_text, intent_analysis)
        
        # Step 5: Final Domain Selection
        final_selection = await self._select_optimal_domains(context_enhanced, urgency_analysis)
        
        # Step 6: Generate Human-Readable Reasoning
        reasoning = await self._generate_reasoning(user_input, intent_analysis, final_selection)
        
        result = {
            "primary_domain": final_selection["primary_domain"],
            "confidence": final_selection["confidence"],
            "secondary_domains": final_selection.get("secondary_domains", []),
            "intent_category": intent_analysis["category"],
            "urgency_level": urgency_analysis["level"],
            "reasoning": reasoning,
            "context_clues": final_selection.get("context_clues", []),
            "multi_domain": len(final_selection.get("secondary_domains", [])) > 0,
            "detection_timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Update performance stats
        await self._update_detection_stats(result)
        
        return result
    
    async def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze user intent from text patterns"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent_category, config in self.intent_patterns.items():
            score = 0.0
            matched_patterns = []
            
            for pattern in config["patterns"]:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += len(matches) * 0.1
                    matched_patterns.extend(matches)
            
            # Check for urgency indicators
            urgency_boost = 0.0
            for urgency_word in config["urgency_indicators"]:
                if urgency_word in text_lower:
                    urgency_boost += 0.2
            
            intent_scores[intent_category] = {
                "score": min(score + urgency_boost, 1.0),
                "matched_patterns": matched_patterns,
                "urgency_detected": urgency_boost > 0
            }
        
        # Select highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "category": best_intent[0],
            "confidence": best_intent[1]["score"],
            "all_scores": intent_scores,
            "matched_patterns": best_intent[1]["matched_patterns"]
        }
    
    async def _calculate_domain_scores(self, text: str, intent_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate relevance scores for each domain"""
        domain_scores = {}
        text_lower = text.lower()
        
        # Get suggested domains from intent analysis
        intent_category = intent_analysis["category"]
        suggested_domains = self.intent_patterns.get(intent_category, {}).get("domains", [])
        
        # Score all domains, with boost for intent-suggested domains
        for domain in self.all_domains:
            score = 0.0
            
            # Intent-based boost
            if domain in suggested_domains:
                score += 0.4 * intent_analysis["confidence"]
            
            # Keyword-based scoring
            domain_keywords = self._get_domain_keywords(domain)
            for keyword in domain_keywords:
                if keyword in text_lower:
                    score += 0.1
            
            # Category-based scoring
            domain_category = self._get_domain_category(domain)
            if domain_category and domain_category in text_lower:
                score += 0.2
            
            domain_scores[domain] = min(score, 1.0)
        
        return domain_scores
    
    async def _enhance_with_context(self, domain_scores: Dict[str, float], text: str) -> Dict[str, float]:
        """Enhance domain scores with contextual analysis"""
        enhanced_scores = domain_scores.copy()
        text_lower = text.lower()
        
        # Time-based context enhancement
        for time_type, indicators in self.context_clues["time_indicators"].items():
            for indicator in indicators:
                if indicator in text_lower:
                    if time_type == "urgent":
                        # Boost healthcare and emergency domains
                        for domain in ["emergency_care", "mental_health", "crisis_intervention"]:
                            if domain in enhanced_scores:
                                enhanced_scores[domain] += 0.2
                    break
        
        # Emotional context enhancement
        for emotion_type, indicators in self.context_clues["emotional_indicators"].items():
            for indicator in indicators:
                if indicator in text_lower:
                    if emotion_type == "distress":
                        # Boost mental health and support domains
                        for domain in ["mental_health", "stress_management", "conflict_resolution"]:
                            if domain in enhanced_scores:
                                enhanced_scores[domain] += 0.15
                    elif emotion_type == "curiosity":
                        # Boost educational domains
                        for domain in ["academic_tutoring", "skill_development", "research_assistance"]:
                            if domain in enhanced_scores:
                                enhanced_scores[domain] += 0.15
                    break
        
        # Normalize scores
        for domain in enhanced_scores:
            enhanced_scores[domain] = min(enhanced_scores[domain], 1.0)
        
        return enhanced_scores
    
    async def _detect_urgency(self, text: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect urgency level from text and intent"""
        text_lower = text.lower()
        urgency_indicators = {
            "critical": ["emergency", "crisis", "critical", "life threatening", "suicide", "911"],
            "high": ["urgent", "asap", "immediately", "now", "help", "can't"],
            "medium": ["soon", "today", "deadline", "important"],
            "low": ["when possible", "eventually", "sometime", "casual"]
        }
        
        detected_level = "low"
        matched_indicators = []
        
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    detected_level = level
                    matched_indicators.append(indicator)
                    break
            if detected_level == level and level in ["critical", "high"]:
                break  # Stop at first high-priority match
        
        return {
            "level": detected_level,
            "indicators": matched_indicators,
            "requires_immediate_attention": detected_level in ["critical", "high"]
        }
    
    async def _select_optimal_domains(self, domain_scores: Dict[str, float], 
                                    urgency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the optimal primary and secondary domains"""
        
        # Sort domains by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select primary domain (highest score)
        primary_domain = sorted_domains[0][0]
        primary_confidence = sorted_domains[0][1]
        
        # Select secondary domains (score > 0.3 and not primary)
        secondary_domains = [
            domain for domain, score in sorted_domains[1:6]  # Top 5 alternatives
            if score > 0.3
        ]
        
        # Adjust confidence based on urgency
        if urgency_analysis["level"] in ["critical", "high"]:
            # In urgent situations, be more decisive
            primary_confidence = min(primary_confidence + 0.1, 1.0)
        
        return {
            "primary_domain": primary_domain,
            "confidence": primary_confidence,
            "secondary_domains": secondary_domains,
            "all_scores": dict(sorted_domains),
            "context_clues": urgency_analysis.get("indicators", [])
        }
    
    async def _generate_reasoning(self, user_input: str, intent_analysis: Dict[str, Any], 
                                final_selection: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for domain selection"""
        
        intent_category = intent_analysis["category"]
        primary_domain = final_selection["primary_domain"]
        confidence = final_selection["confidence"]
        
        reasoning_templates = {
            "health_seeking": f"Detected health-related concern in your message. Routing to {primary_domain} domain for appropriate care.",
            "learning_seeking": f"Identified learning intent. Directing to {primary_domain} for educational support.",
            "problem_solving": f"Recognized technical problem. Routing to {primary_domain} for solution assistance.",
            "emotional_support": f"Detected emotional context. Connecting to {primary_domain} for supportive guidance.",
            "business_guidance": f"Identified business inquiry. Routing to {primary_domain} for professional advice.",
            "creative_expression": f"Recognized creative intent. Directing to {primary_domain} for inspiration and guidance."
        }
        
        base_reasoning = reasoning_templates.get(
            intent_category, 
            f"Analyzed your request and determined {primary_domain} is the best match."
        )
        
        confidence_note = ""
        if confidence > 0.8:
            confidence_note = " (High confidence match)"
        elif confidence > 0.6:
            confidence_note = " (Good confidence match)"
        elif confidence > 0.4:
            confidence_note = " (Moderate confidence - may ask for clarification)"
        else:
            confidence_note = " (Low confidence - will ask for clarification)"
        
        return base_reasoning + confidence_note
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords associated with a domain for scoring"""
        # This would be expanded with comprehensive keyword mapping
        keyword_mapping = {
            "general_health": ["health", "medical", "doctor", "medicine", "wellness", "body"],
            "mental_health": ["mental", "therapy", "counseling", "anxiety", "depression", "emotional"],
            "programming": ["code", "programming", "software", "development", "bug", "algorithm"],
            "writing": ["write", "writing", "story", "content", "blog", "author"],
            # Add more mappings as needed
        }
        
        return keyword_mapping.get(domain, [domain.replace("_", " ")])
    
    def _get_domain_category(self, domain: str) -> Optional[str]:
        """Get the category for a domain"""
        for category, domains in self.domain_categories.items():
            if domain in domains:
                return category
        return None
    
    async def _update_detection_stats(self, result: Dict[str, Any]):
        """Update performance statistics"""
        self.detection_stats["total_detections"] += 1
        
        if result["confidence"] > 0.5:
            self.detection_stats["successful_detections"] += 1
        
        if result["multi_domain"]:
            self.detection_stats["multi_domain_detections"] += 1
        
        if result["confidence"] > 0.8:
            self.detection_stats["high_confidence_detections"] += 1
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get domain detection performance statistics"""
        total = self.detection_stats["total_detections"]
        
        return {
            "total_detections": total,
            "success_rate": (self.detection_stats["successful_detections"] / total * 100) if total > 0 else 0,
            "multi_domain_rate": (self.detection_stats["multi_domain_detections"] / total * 100) if total > 0 else 0,
            "high_confidence_rate": (self.detection_stats["high_confidence_detections"] / total * 100) if total > 0 else 0,
            "available_domains": len(self.all_domains),
            "domain_categories": len(self.domain_categories)
        }

# Convenience function for easy integration
async def detect_domain_for_human(user_input: str, context: str = "", 
                                conversation_history: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to detect what domain a human needs
    
    Usage:
        result = await detect_domain_for_human("I have a headache and feel stressed")
        primary_domain = result["primary_domain"]  # e.g., "general_health"
        confidence = result["confidence"]  # e.g., 0.85
        reasoning = result["reasoning"]  # Human-readable explanation
    """
    detector = TARACoreIntelligence()
    return await detector.detect_domain_intelligent(user_input, context, conversation_history) 