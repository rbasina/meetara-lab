#!/usr/bin/env python3
"""
Comprehensive Data Generator for MeeTARA Lab
Uses all 65 domains from original TARA Universal Model with rich multi-scenario format
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import random
import string
import uuid
import os
from datetime import datetime
import re
from dataclasses import dataclass

# Trinity Architecture imports
from trinity_core.agents.coordination.lightweight_mcp_v2 import LightweightMCPv2, MCPMessage
from trinity_core.intelligence_layer.intelligence.comprehensive_intelligence import TARAComprehensiveIntelligence
from trinity_core.core_components.config_manager import SmartTrinityConfigManager
from trinity_core.agents.comprehensive_domain_templates import COMPREHENSIVE_DOMAIN_TEMPLATES

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveDataConfig:
    """Enhanced configuration for comprehensive data generation."""
    output_dir: str = "data/training"
    quality_threshold: float = 0.8
    diversity_threshold: float = 0.7
    samples_per_domain: int = 5000
    synthetic_data_path: str = "data/training"
    
    # Trinity Architecture enhancements
    enable_trinity_enhancement: bool = True
    target_accuracy: float = 0.9999  # 99.99% accuracy target
    enable_crisis_intervention: bool = True
    enable_emotional_intelligence: bool = True
    enable_domain_expertise: bool = True
    enable_urgency_analysis: bool = True
    enable_dynamic_ratio: bool = True
    enable_blended_conversations: bool = True
    
    # Advanced modularity features
    urgency_patterns: Dict[str, float] = None
    domain_criticality: Dict[str, float] = None
    user_intent_urgency: Dict[str, float] = None
    dynamic_ratio_config: Dict[str, Any] = None

class ComprehensiveDataGenerator:
    """
    Comprehensive Data Generator Agent - All 65 domains from original TARA Universal Model
    Responsible for generating high-quality, domain-specific training data with rich multi-scenario format.
    """
    def __init__(self, hub: Any):
        self.hub = hub
        self.config_manager = hub.config_manager
        self.config = self.config_manager.get_config_dict()
        self.mcp = hub.mcp
        self.intelligence = hub.intelligence
        self.domain_templates = {}
        self.comprehensive_config = ComprehensiveDataConfig()
        
        # Initialize advanced modularity components
        self._initialize_urgency_patterns()
        self._initialize_domain_criticality()
        self._initialize_user_intent_urgency()
        self._initialize_dynamic_ratio_config()
        self._initialize_comprehensive_templates()

    def _initialize_urgency_patterns(self) -> None:
        """Initialize urgency pattern analysis for Trinity Architecture."""
        self.urgency_patterns = {
            "emergency_keywords": ["emergency", "urgent", "crisis", "immediate", "critical", "help", "now"],
            "medical_emergency": ["heart attack", "stroke", "bleeding", "unconscious", "not breathing"],
            "mental_health_crisis": ["suicide", "self-harm", "hopeless", "can't take it", "end it all"],
            "safety_emergency": ["fire", "accident", "danger", "threat", "violence"],
            "financial_crisis": ["bankruptcy", "eviction", "debt crisis", "financial emergency"],
            "relationship_crisis": ["breakup", "divorce", "abuse", "cheating", "betrayal"],
            "work_crisis": ["fired", "layoff", "workplace emergency", "career crisis"],
            "academic_crisis": ["failing", "expulsion", "academic emergency", "test failure"]
        }

    def _initialize_domain_criticality(self) -> None:
        """Initialize domain criticality levels for Trinity Architecture."""
        self.domain_criticality = {
            "general_health": 0.95,      # Life-critical
            "mental_health": 0.95,   # Life-critical
            "emergency_care": 0.98,  # Maximum criticality
            "crisis_management": 0.98, # Maximum criticality
            "legal": 0.90,           # High criticality
            "financial": 0.85,       # High criticality
            "business": 0.70,        # Medium criticality
            "education": 0.65,       # Medium criticality
            "creative": 0.40,        # Low criticality
            "entertainment": 0.30,   # Low criticality
            "shopping": 0.25,        # Very low criticality
            "general": 0.50          # Default criticality
        }

    def _initialize_user_intent_urgency(self) -> None:
        """Initialize user intent urgency analysis."""
        self.user_intent_urgency = {
            "immediate_help": 0.95,
            "crisis_support": 0.98,
            "emergency_guidance": 0.99,
            "urgent_advice": 0.85,
            "quick_question": 0.30,
            "general_inquiry": 0.20,
            "casual_conversation": 0.10,
            "information_seeking": 0.40,
            "problem_solving": 0.60,
            "decision_support": 0.70,
            "emotional_support": 0.80,
            "professional_guidance": 0.75
        }

    def _initialize_dynamic_ratio_config(self) -> None:
        """Initialize dynamic ratio configuration for Trinity Architecture."""
        self.dynamic_ratio_config = {
            "base_realtime_ratio": 0.30,  # 30% real-time scenarios by default
            "urgency_weight": 0.40,       # 40% weight for urgency analysis
            "criticality_weight": 0.35,   # 35% weight for domain criticality
            "intent_weight": 0.25,        # 25% weight for user intent
            "min_realtime_ratio": 0.10,   # Minimum 10% real-time scenarios
            "max_realtime_ratio": 0.80,   # Maximum 80% real-time scenarios
            "trinity_amplification": 1.5   # Trinity Architecture amplification factor
        }

    def _initialize_comprehensive_templates(self) -> None:
        """Initialize comprehensive domain templates from original TARA Universal Model."""
        
        # Use comprehensive domain templates from original TARA Universal Model
        self.domain_templates = COMPREHENSIVE_DOMAIN_TEMPLATES.copy()
        
        logger.info(f"✅ Loaded comprehensive templates for {len(self.domain_templates)} domains from original TARA Universal Model")
        logger.info("🎯 All domains now have rich multi-scenario format with Trinity Architecture enhancements")
        
        # Log domain categories
        categories = self._categorize_domains()
        for category, domains in categories.items():
            if domains:
                logger.info(f"  {category}: {len(domains)} domains")

    def _categorize_domains(self) -> Dict[str, List[str]]:
        """Categorize domains for logging purposes."""
        categories = {
            "Healthcare": [],
            "Daily Life": [],
            "Business": [],
            "Education": [],
            "Technology": [],
            "Space Technology": [],
            "Creative": [],
            "Specialized": []
        }
        
        for domain in self.domain_templates.keys():
            if any(h in domain for h in ["health", "mental", "nutrition", "fitness", "sleep", "stress", "preventive", "chronic", "medication", "emergency", "women", "senior"]):
                categories["Healthcare"].append(domain)
            elif any(dl in domain for dl in ["parenting", "relationships", "personal", "communication", "home", "shopping", "planning", "transportation", "time", "decision", "conflict", "work_life"]):
                categories["Daily Life"].append(domain)
            elif any(b in domain for b in ["entrepreneurship", "marketing", "sales", "customer", "project", "team", "financial", "operations", "hr", "strategy", "consulting", "legal_business"]):
                categories["Business"].append(domain)
            elif any(e in domain for e in ["academic", "skill", "career", "exam", "language", "research", "study", "educational"]):
                categories["Education"].append(domain)
            elif any(t in domain for t in ["programming", "ai_ml", "cybersecurity", "data", "tech", "software"]):
                categories["Technology"].append(domain)
            elif any(s in domain for s in ["space", "aerospace", "satellite"]):
                categories["Space Technology"].append(domain)
            elif any(c in domain for c in ["creative", "visual", "music", "content", "design", "innovation", "photography", "film"]):
                categories["Creative"].append(domain)
            elif any(sp in domain for sp in ["legal", "financial", "scientific", "engineering"]):
                categories["Specialized"].append(domain)
        
        return categories

    def _analyze_urgency_patterns(self, conversation_starters: List[str]) -> float:
        """Analyze urgency patterns in conversation starters for Trinity Architecture."""
        total_urgency_score = 0.0
        total_starters = len(conversation_starters)
        
        for starter in conversation_starters:
            starter_lower = starter.lower()
            urgency_score = 0.0
            
            # Check for emergency keywords
            for keyword in self.urgency_patterns["emergency_keywords"]:
                if keyword in starter_lower:
                    urgency_score += 0.3
            
            # Check for medical emergencies
            for emergency in self.urgency_patterns["medical_emergency"]:
                if emergency in starter_lower:
                    urgency_score += 0.8
            
            # Check for mental health crises
            for crisis in self.urgency_patterns["mental_health_crisis"]:
                if crisis in starter_lower:
                    urgency_score += 0.9
            
            # Check for safety emergencies
            for safety in self.urgency_patterns["safety_emergency"]:
                if safety in starter_lower:
                    urgency_score += 0.7
            
            # Check for financial crises
            for financial in self.urgency_patterns["financial_crisis"]:
                if financial in starter_lower:
                    urgency_score += 0.6
            
            # Check for relationship crises
            for relationship in self.urgency_patterns["relationship_crisis"]:
                if relationship in starter_lower:
                    urgency_score += 0.5
            
            # Check for work crises
            for work in self.urgency_patterns["work_crisis"]:
                if work in starter_lower:
                    urgency_score += 0.4
            
            # Check for academic crises
            for academic in self.urgency_patterns["academic_crisis"]:
                if academic in starter_lower:
                    urgency_score += 0.3
            
            # Normalize urgency score
            urgency_score = min(urgency_score, 1.0)
            total_urgency_score += urgency_score
        
        return total_urgency_score / total_starters if total_starters > 0 else 0.0

    def _detect_domain_criticality(self, domain: str) -> float:
        """Detect domain criticality level for Trinity Architecture."""
        domain_lower = domain.lower()
        
        # Healthcare domains (highest criticality)
        if any(h in domain_lower for h in ["health", "medical", "emergency", "crisis"]):
            return self.domain_criticality.get("general_health", 0.95)
        
        # Legal and financial domains (high criticality)
        elif any(l in domain_lower for l in ["legal", "financial", "law"]):
            return self.domain_criticality.get("legal", 0.90)
        
        # Business domains (medium criticality)
        elif any(b in domain_lower for b in ["business", "entrepreneurship", "marketing", "sales"]):
            return self.domain_criticality.get("business", 0.70)
        
        # Education domains (medium criticality)
        elif any(e in domain_lower for e in ["education", "academic", "learning", "study"]):
            return self.domain_criticality.get("education", 0.65)
        
        # Creative domains (low criticality)
        elif any(c in domain_lower for c in ["creative", "art", "music", "writing"]):
            return self.domain_criticality.get("creative", 0.40)
        
        # Entertainment domains (lowest criticality)
        elif any(en in domain_lower for en in ["entertainment", "shopping", "leisure"]):
            return self.domain_criticality.get("entertainment", 0.30)
        
        # Default criticality
        else:
            return self.domain_criticality.get("general", 0.50)

    def _analyze_user_intent_urgency(self, user_intents: List[str]) -> float:
        """Analyze user intent urgency for Trinity Architecture."""
        total_urgency_score = 0.0
        total_intents = len(user_intents)
        
        for intent in user_intents:
            intent_lower = intent.lower()
            urgency_score = 0.0
            
            # Check for immediate help intents
            if any(immediate in intent_lower for immediate in ["immediate", "urgent", "crisis", "emergency"]):
                urgency_score = self.user_intent_urgency.get("immediate_help", 0.95)
            elif any(crisis in intent_lower for crisis in ["crisis", "emergency", "critical"]):
                urgency_score = self.user_intent_urgency.get("crisis_support", 0.98)
            elif any(urgent in intent_lower for urgent in ["urgent", "quick", "fast"]):
                urgency_score = self.user_intent_urgency.get("urgent_advice", 0.85)
            elif any(emotional in intent_lower for emotional in ["emotional", "support", "help"]):
                urgency_score = self.user_intent_urgency.get("emotional_support", 0.80)
            elif any(professional in intent_lower for professional in ["professional", "expert", "guidance"]):
                urgency_score = self.user_intent_urgency.get("professional_guidance", 0.75)
            elif any(decision in intent_lower for decision in ["decision", "choice", "option"]):
                urgency_score = self.user_intent_urgency.get("decision_support", 0.70)
            elif any(problem in intent_lower for problem in ["problem", "issue", "trouble"]):
                urgency_score = self.user_intent_urgency.get("problem_solving", 0.60)
            elif any(information in intent_lower for information in ["information", "learn", "understand"]):
                urgency_score = self.user_intent_urgency.get("information_seeking", 0.40)
            elif any(quick in intent_lower for quick in ["quick", "simple", "basic"]):
                urgency_score = self.user_intent_urgency.get("quick_question", 0.30)
            elif any(general in intent_lower for general in ["general", "casual", "chat"]):
                urgency_score = self.user_intent_urgency.get("general_inquiry", 0.20)
            else:
                urgency_score = self.user_intent_urgency.get("casual_conversation", 0.10)
            
            total_urgency_score += urgency_score
        
        return total_urgency_score / total_intents if total_intents > 0 else 0.0

    def _calculate_dynamic_ratio(self, urgency_score: float, domain_criticality: float, user_intent_urgency: float) -> float:
        """Calculate dynamic real-time scenario ratio for Trinity Architecture."""
        config = self.dynamic_ratio_config
        
        # Calculate weighted average
        weighted_score = (
            urgency_score * config["urgency_weight"] +
            domain_criticality * config["criticality_weight"] +
            user_intent_urgency * config["intent_weight"]
        )
        
        # Apply Trinity Architecture amplification
        amplified_score = weighted_score * config["trinity_amplification"]
        
        # Calculate final ratio within bounds
        final_ratio = config["base_realtime_ratio"] + (amplified_score * 0.5)
        final_ratio = max(config["min_realtime_ratio"], min(config["max_realtime_ratio"], final_ratio))
        
        return final_ratio

    def _create_domain_expert_agent(self, domain: str) -> Dict[str, Any]:
        """Create comprehensive domain expert agent with Trinity Architecture."""
        domain_expert = {
            "domain": domain,
            "expertise_level": "comprehensive",
            "trinity_phase": "einstein_fusion",
            "capabilities": [],
            "response_patterns": [],
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True,
            "safety_level": "high",
            "privacy_level": "standard"
        }
        
        # Get domain template
        template = self.domain_templates.get(domain, {})
        
        # Update domain expert with template information
        if template:
            domain_expert.update({
                "capabilities": template.get("scenarios", []),
                "response_patterns": template.get("response_patterns", []),
                "crisis_intervention": template.get("crisis_intervention", False),
                "emotional_intelligence": template.get("emotional_intelligence", True),
                "professional_boundaries": template.get("professional_boundaries", True),
                "trinity_phase": template.get("trinity_phase", "perplexity_intelligence")
            })
        
        return domain_expert

    def generate_domain_data(self, domain: str, num_samples: int = 5000,
                           output_path: str = None, quality_threshold: float = 0.8,
                           templates: Dict = None, split_type: str = "train",
                           realtime_scenarios: bool = False) -> str:
        """Generate comprehensive domain data with rich multi-scenario format."""
        
        logger.info(f"🚀 Generating comprehensive data for domain: {domain}")
        logger.info(f"📊 Target samples: {num_samples}")
        logger.info(f"🎯 Quality threshold: {quality_threshold}")
        
        # Get domain template
        domain_config = self.domain_templates.get(domain, {})
        if not domain_config:
            logger.error(f"❌ Domain template not found for: {domain}")
            return ""
        
        # Create domain expert agent
        domain_expert = self._create_domain_expert_agent(domain)
        
        # Analyze urgency patterns
        conversation_starters = domain_config.get("conversation_starters", [])
        user_intents = domain_config.get("user_intents", [])
        
        urgency_score = self._analyze_urgency_patterns(conversation_starters)
        domain_criticality = self._detect_domain_criticality(domain)
        user_intent_urgency = self._analyze_user_intent_urgency(user_intents)
        
        # Calculate dynamic ratio
        dynamic_ratio = self._calculate_dynamic_ratio(urgency_score, domain_criticality, user_intent_urgency)
        
        logger.info(f"📈 Urgency analysis: {urgency_score:.3f}")
        logger.info(f"🎯 Domain criticality: {domain_criticality:.3f}")
        logger.info(f"💡 User intent urgency: {user_intent_urgency:.3f}")
        logger.info(f"⚡ Dynamic ratio: {dynamic_ratio:.3f}")
        
        # Generate blended conversations
        conversations = []
        for i in range(num_samples):
            conversation = self._generate_blended_conversation(domain, domain_config)
            conversations.append(conversation)
        
        # Save conversations
        output_file = output_path or f"data/training/{domain}_comprehensive.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + '\n')
        
        logger.info(f"✅ Generated {len(conversations)} comprehensive conversations for {domain}")
        logger.info(f"💾 Saved to: {output_file}")
        
        return output_file

    def _generate_blended_conversation(self, domain: str, domain_config: Dict) -> Dict[str, Any]:
        """Generate blended conversation with Trinity Architecture enhancements."""
        
        # Get conversation starters and split by urgency
        all_starters = domain_config.get("conversation_starters", [])
        scenarios = domain_config.get("scenarios", [])
        user_intents = domain_config.get("user_intents", [])
        response_patterns = domain_config.get("response_patterns", [])
        
        # Calculate dynamic ratio
        urgency_score = self._analyze_urgency_patterns(all_starters)
        domain_criticality = self._detect_domain_criticality(domain)
        user_intent_urgency = self._analyze_user_intent_urgency(user_intents)
        dynamic_ratio = self._calculate_dynamic_ratio(urgency_score, domain_criticality, user_intent_urgency)
        
        # Split starters by urgency
        urgent_starters, general_starters = self._split_starters_by_urgency(all_starters, dynamic_ratio)
        
        # Choose conversation type based on dynamic ratio
        if random.random() < dynamic_ratio and urgent_starters:
            # Generate real-time scenario
            conversation = self._generate_realtime_conversation(domain, urgent_starters, domain_config)
        else:
            # Generate general conversation
            conversation = self._generate_general_conversation(domain, general_starters, domain_config)
        
        # Add Trinity Architecture metadata
        conversation["trinity_metadata"] = {
            "domain": domain,
            "urgency_score": urgency_score,
            "domain_criticality": domain_criticality,
            "user_intent_urgency": user_intent_urgency,
            "dynamic_ratio": dynamic_ratio,
            "trinity_phase": domain_config.get("trinity_phase", "perplexity_intelligence"),
            "crisis_intervention": domain_config.get("crisis_intervention", False),
            "emotional_intelligence": domain_config.get("emotional_intelligence", True),
            "professional_boundaries": domain_config.get("professional_boundaries", True)
        }
        
        return conversation

    def _split_starters_by_urgency(self, all_starters: List[str], realtime_ratio: float) -> Tuple[List[str], List[str]]:
        """Split conversation starters by urgency level."""
        urgent_starters = []
        general_starters = []
        
        for starter in all_starters:
            starter_lower = starter.lower()
            
            # Check for urgent indicators
            is_urgent = any(keyword in starter_lower for keyword in [
                "emergency", "urgent", "crisis", "immediate", "critical", "help", "now",
                "can't", "don't know", "terrified", "worried", "scared", "panic"
            ])
            
            if is_urgent:
                urgent_starters.append(starter)
            else:
                general_starters.append(starter)
        
        # Ensure we have enough starters for each category
        if not urgent_starters:
            urgent_starters = general_starters[:len(general_starters)//2]
            general_starters = general_starters[len(general_starters)//2:]
        
        return urgent_starters, general_starters

    def _generate_realtime_conversation(self, domain: str, urgent_starters: List[str], domain_config: Dict) -> Dict[str, Any]:
        """Generate real-time scenario conversation."""
        
        # Select random urgent starter
        user_message = random.choice(urgent_starters) if urgent_starters else "I need immediate help!"
        
        # Detect emotion
        emotion = self._detect_emotion(user_message)
        
        # Generate assistant response
        assistant_response = self._generate_blended_assistant_response(
            user_message, domain, "crisis_intervention", emotion, "realtime"
        )
        
        # Generate follow-up
        followup_user = self._generate_followup_user([{"user": user_message, "assistant": assistant_response}], "crisis", emotion)
        followup_assistant = self._generate_blended_followup_assistant(
            [{"user": user_message, "assistant": assistant_response}], domain, "crisis_intervention", "realtime"
        )
        
        return {
            "conversation": [
                {"user": user_message, "assistant": assistant_response},
                {"user": followup_user, "assistant": followup_assistant}
            ],
            "scenario_type": "realtime",
            "emotion": emotion,
            "domain": domain
        }

    def _generate_general_conversation(self, domain: str, general_starters: List[str], domain_config: Dict) -> Dict[str, Any]:
        """Generate general conversation."""
        
        # Select random general starter
        user_message = random.choice(general_starters) if general_starters else "I need help with this topic."
        
        # Detect emotion
        emotion = self._detect_emotion(user_message)
        
        # Generate assistant response
        assistant_response = self._generate_blended_assistant_response(
            user_message, domain, "general_guidance", emotion, "general"
        )
        
        # Generate follow-up
        followup_user = self._generate_followup_user([{"user": user_message, "assistant": assistant_response}], "general", emotion)
        followup_assistant = self._generate_blended_followup_assistant(
            [{"user": user_message, "assistant": assistant_response}], domain, "general_guidance", "general"
        )
        
        return {
            "conversation": [
                {"user": user_message, "assistant": assistant_response},
                {"user": followup_user, "assistant": followup_assistant}
            ],
            "scenario_type": "general",
            "emotion": emotion,
            "domain": domain
        }

    def _detect_emotion(self, message: str) -> str:
        """Detect emotion in user message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["terrified", "scared", "panic", "fear"]):
            return "fear"
        elif any(word in message_lower for word in ["angry", "furious", "mad", "upset"]):
            return "anger"
        elif any(word in message_lower for word in ["sad", "depressed", "hopeless", "crying"]):
            return "sadness"
        elif any(word in message_lower for word in ["excited", "happy", "joy", "thrilled"]):
            return "joy"
        elif any(word in message_lower for word in ["worried", "anxious", "nervous", "stress"]):
            return "anxiety"
        elif any(word in message_lower for word in ["confused", "unsure", "don't know"]):
            return "confusion"
        else:
            return "neutral"

    def _generate_blended_assistant_response(self, user_message: str, domain: str, 
                                           pattern: str, emotion: str, scenario_type: str) -> str:
        """Generate blended assistant response with Trinity Architecture."""
        
        # Base response based on pattern
        if pattern == "crisis_intervention":
            base_response = f"I understand this is urgent. Let me help you with {domain.replace('_', ' ')} immediately."
        elif pattern == "general_guidance":
            base_response = f"I can help you with {domain.replace('_', ' ')}. Let me provide some guidance."
        else:
            base_response = f"I'm here to help with {domain.replace('_', ' ')}. What specific information do you need?"
        
        # Enhance with emotional intelligence
        if emotion != "neutral":
            base_response = self._enhance_with_emotional_intelligence(base_response, emotion)
        
        # Enhance based on scenario type
        if scenario_type == "realtime":
            base_response = self._enhance_crisis_response(base_response, {"domain": domain})
        else:
            base_response = self._enhance_general_response(base_response, {"domain": domain})
        
        return base_response

    def _enhance_crisis_response(self, base_response: str, domain_expert: Dict) -> str:
        """Enhance crisis response with Trinity Architecture."""
        domain = domain_expert.get("domain", "general")
        
        crisis_enhancements = {
            "general_health": "This is a medical situation that requires immediate attention. ",
            "mental_health": "Your mental health is important and you deserve support. ",
            "emergency_care": "This is an emergency situation. Please seek immediate help. ",
            "legal": "This is a serious legal matter that requires professional attention. ",
            "financial": "This is a critical financial situation that needs immediate attention. "
        }
        
        enhancement = crisis_enhancements.get(domain, "This is an urgent situation that requires immediate attention. ")
        return enhancement + base_response

    def _enhance_general_response(self, base_response: str, domain_expert: Dict) -> str:
        """Enhance general response with Trinity Architecture."""
        domain = domain_expert.get("domain", "general")
        
        general_enhancements = {
            "general_health": "I can provide general health information to help guide you. ",
            "business": "I can offer business insights and strategies to help you succeed. ",
            "education": "I can help you with learning strategies and educational guidance. ",
            "creative": "I can help you explore your creativity and artistic expression. ",
            "technology": "I can provide technical guidance and problem-solving support. "
        }
        
        enhancement = general_enhancements.get(domain, "I can provide helpful information and guidance. ")
        return enhancement + base_response

    def _enhance_with_emotional_intelligence(self, base_response: str, user_emotion: str) -> str:
        """Enhance response with emotional intelligence."""
        emotion_enhancements = {
            "fear": "I understand this is frightening. Let me help you feel safer. ",
            "anger": "I can see you're frustrated. Let me help you find a solution. ",
            "sadness": "I hear how difficult this is for you. You're not alone. ",
            "joy": "I'm glad you're feeling positive! Let me help you build on that. ",
            "anxiety": "I understand this is causing you anxiety. Let me help you feel more at ease. ",
            "confusion": "I can see this is confusing. Let me help clarify things for you. "
        }
        
        enhancement = emotion_enhancements.get(user_emotion, "")
        return enhancement + base_response

    def _generate_followup_user(self, conversation_history: List[Dict], 
                               scenario: str, emotion: str) -> str:
        """Generate follow-up user message."""
        
        followup_templates = {
            "crisis": [
                "What should I do next?",
                "I'm still really worried. Can you help more?",
                "This is really urgent. What's the next step?",
                "I need more specific guidance right now."
            ],
            "general": [
                "That's helpful. Can you tell me more?",
                "I have another question about this.",
                "This is useful information. What else should I know?",
                "Can you provide more details?"
            ]
        }
        
        templates = followup_templates.get(scenario, followup_templates["general"])
        return random.choice(templates)

    def _generate_blended_followup_assistant(self, conversation_history: List[Dict],
                                           domain: str, pattern: str, scenario_type: str) -> str:
        """Generate blended follow-up assistant response."""
        
        if scenario_type == "realtime":
            return f"I'm here to continue helping you with this urgent {domain.replace('_', ' ')} situation. Let me provide more specific guidance."
        else:
            return f"I'm happy to continue helping you with {domain.replace('_', ' ')}. What specific aspect would you like to explore further?"

    def generate_all_domains(self, samples_per_domain: int = 1000) -> Dict[str, str]:
        """Generate comprehensive data for all 65 domains."""
        
        logger.info(f"🚀 Starting comprehensive data generation for all {len(self.domain_templates)} domains")
        logger.info(f"📊 Target samples per domain: {samples_per_domain}")
        
        results = {}
        
        for domain in self.domain_templates.keys():
            try:
                output_file = self.generate_domain_data(domain, samples_per_domain)
                results[domain] = output_file
                logger.info(f"✅ Completed {domain}: {output_file}")
            except Exception as e:
                logger.error(f"❌ Error generating data for {domain}: {e}")
                results[domain] = ""
        
        logger.info(f"🎯 Comprehensive data generation completed for {len(results)} domains")
        return results

def main():
    """Test comprehensive data generator."""
    # Create mock hub for testing
    class MockHub:
        def __init__(self):
            self.config_manager = MockConfigManager()
            self.mcp = None
            self.intelligence = None
    
    class MockConfigManager:
        def get_config_dict(self):
            return {"data_generation": {"samples_per_domain": 1000}}
    
    hub = MockHub()
    generator = ComprehensiveDataGenerator(hub)
    
    # Test with a few domains
    test_domains = ["healthcare", "mental_health", "entrepreneurship", "programming"]
    
    for domain in test_domains:
        if domain in generator.domain_templates:
            print(f"✅ Testing {domain}")
            output_file = generator.generate_domain_data(domain, 10)  # Small test
            print(f"📁 Output: {output_file}")
        else:
            print(f"❌ Domain not found: {domain}")

if __name__ == "__main__":
    main() 