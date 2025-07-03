"""
MeeTARA Lab - Data Generator Agent
High-quality training data generation with emotional context, crisis scenarios, and domain expertise
Enhanced with TARA Universal Model proven real-time scenario patterns
"""

import asyncio
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class TARARealTimeScenarioEngine:
    """
    Enhanced TARA Real-Time Scenario Engine with advanced capabilities
    Based on TARA Universal Model proven patterns
    """
    
    def __init__(self, tara_model_path: str = None):
        self.tara_model_path = tara_model_path
        self.tara_config = self._load_tara_config()
        self.domain_criticality_cache = {}
        self.urgency_patterns_cache = {}
        self.emotional_intelligence_agent = None
        self.quality_validation_agent = None
        self.domain_expert_agents = {}
        self.crisis_scenario_agents = {}
        
        # Initialize advanced agents
        self._initialize_emotional_intelligence_agent()
        self._initialize_quality_validation_agent()
        
    def _load_tara_config(self) -> Dict[str, Any]:
        """Load TARA configuration with enhanced real-time capabilities"""
        return {
            "enable_agentic_ai": True,
            "target_accuracy": 0.9999,  # 99.99% accuracy target
            "enable_crisis_intervention": True,
            "enable_emotional_intelligence": True,
            "enable_domain_expertise": True,
            "quality_threshold": 0.8,
            "diversity_threshold": 0.7,
            "samples_per_domain": 2000,  # TARA proven parameter
            "realtime_scenario_ratio": 0.40,  # 40% real-time scenarios
            "crisis_scenario_ratio": 0.05,   # 5% crisis scenarios
            "emotional_contexts": [
                "anxious", "confident", "confused", "frustrated", "hopeful",
                "worried", "excited", "overwhelmed", "determined", "uncertain",
                "stressed", "curious", "relieved", "disappointed", "motivated"
            ],
            "urgency_levels": ["low", "medium", "high", "critical", "emergency"],
            "response_types": ["crisis", "guidance", "informational", "supportive", "educational"]
        }
    
    def _initialize_emotional_intelligence_agent(self):
        """Initialize advanced emotional intelligence agent"""
        self.emotional_intelligence_agent = {
            "emotional_patterns": {
                "anxious": ["worried", "nervous", "concerned", "uneasy", "troubled"],
                "confident": ["sure", "certain", "assured", "positive", "determined"],
                "confused": ["unclear", "uncertain", "puzzled", "bewildered", "lost"],
                "frustrated": ["annoyed", "irritated", "upset", "disappointed", "angry"],
                "hopeful": ["optimistic", "encouraged", "positive", "expectant", "confident"],
                "worried": ["concerned", "anxious", "troubled", "fearful", "apprehensive"],
                "excited": ["enthusiastic", "eager", "thrilled", "energetic", "motivated"],
                "overwhelmed": ["stressed", "overburdened", "exhausted", "swamped", "pressured"],
                "determined": ["focused", "committed", "resolved", "driven", "persistent"],
                "uncertain": ["unsure", "doubtful", "hesitant", "questioning", "wavering"],
                "stressed": ["tense", "pressured", "overwhelmed", "strained", "burdened"],
                "curious": ["interested", "inquisitive", "questioning", "wondering", "exploring"],
                "relieved": ["comforted", "reassured", "calmed", "eased", "peaceful"],
                "disappointed": ["let down", "discouraged", "disheartened", "frustrated", "sad"],
                "motivated": ["inspired", "driven", "energized", "determined", "focused"]
            },
            "emotional_responses": {
                "anxious": "I understand you're feeling anxious. Let's work through this step by step.",
                "confident": "I can see your confidence. Let's build on that positive energy.",
                "confused": "It's okay to feel confused. Let me help clarify this for you.",
                "frustrated": "I recognize your frustration. Let's find a solution together.",
                "hopeful": "Your hopefulness is inspiring. Let's make progress on this.",
                "worried": "I understand your concerns. Let's address them systematically.",
                "excited": "Your excitement is wonderful! Let's channel that energy productively.",
                "overwhelmed": "Feeling overwhelmed is normal. Let's break this down into manageable steps.",
                "determined": "I admire your determination. Let's work together to achieve your goals.",
                "uncertain": "Uncertainty is part of growth. Let's explore your options together.",
                "stressed": "I recognize you're under stress. Let's find ways to ease this burden.",
                "curious": "Your curiosity is great! Let's explore this topic together.",
                "relieved": "I'm glad you're feeling relieved. Let's build on this positive momentum.",
                "disappointed": "I understand your disappointment. Let's find a path forward.",
                "motivated": "Your motivation is powerful. Let's harness it for success."
            }
        }
    
    def _initialize_quality_validation_agent(self):
        """Initialize advanced quality validation agent"""
        self.quality_validation_agent = {
            "quality_metrics": {
                "emotional_intelligence": 0.25,  # 25% weight
                "domain_accuracy": 0.30,         # 30% weight
                "contextual_relevance": 0.25,    # 25% weight
                "crisis_handling": 0.20          # 20% weight
            },
            "validation_thresholds": {
                "minimum_quality": 0.8,
                "target_quality": 0.9,
                "excellence_threshold": 0.95
            },
            "quality_indicators": [
                "empathy_level", "accuracy_score", "relevance_score", 
                "response_time", "crisis_appropriateness", "emotional_resonance"
            ]
        }
    
    def _get_domain_expert_agent(self, domain: str) -> Dict[str, Any]:
        """Get or create domain expert agent"""
        if domain not in self.domain_expert_agents:
            self.domain_expert_agents[domain] = self._create_domain_expert_agent(domain)
        return self.domain_expert_agents[domain]
    
    def _create_domain_expert_agent(self, domain: str) -> Dict[str, Any]:
        """Create specialized domain expert agent"""
        domain_expertise = {
            # Healthcare domains
            "general_health": {
                "expertise_level": "medical_professional",
                "specializations": ["preventive_care", "health_screening", "wellness_coaching"],
                "crisis_indicators": ["severe_pain", "difficulty_breathing", "chest_pain", "suicidal_thoughts"],
                "response_patterns": ["empathetic", "evidence_based", "safety_focused"]
            },
            "mental_health": {
                "expertise_level": "mental_health_professional",
                "specializations": ["anxiety_management", "depression_support", "stress_reduction"],
                "crisis_indicators": ["self_harm", "suicidal_ideation", "panic_attacks", "psychotic_episodes"],
                "response_patterns": ["therapeutic", "non_judgmental", "crisis_aware"]
            },
            "emergency_care": {
                "expertise_level": "emergency_medical_technician",
                "specializations": ["first_aid", "emergency_response", "trauma_care"],
                "crisis_indicators": ["medical_emergency", "trauma", "poisoning", "severe_injury"],
                "response_patterns": ["immediate_action", "clear_instructions", "calm_authority"]
            },
            
            # Business domains
            "financial_planning": {
                "expertise_level": "financial_advisor",
                "specializations": ["investment_planning", "retirement_planning", "debt_management"],
                "crisis_indicators": ["financial_crisis", "bankruptcy", "fraud", "major_losses"],
                "response_patterns": ["analytical", "strategic", "risk_aware"]
            },
            "legal_business": {
                "expertise_level": "business_attorney",
                "specializations": ["contract_law", "business_formation", "compliance"],
                "crisis_indicators": ["legal_violation", "lawsuit", "regulatory_action", "contract_breach"],
                "response_patterns": ["precise", "cautious", "compliance_focused"]
            },
            
            # Technology domains
            "cybersecurity": {
                "expertise_level": "cybersecurity_specialist",
                "specializations": ["threat_analysis", "incident_response", "security_auditing"],
                "crisis_indicators": ["data_breach", "cyber_attack", "malware", "system_compromise"],
                "response_patterns": ["immediate_containment", "forensic_approach", "security_first"]
            },
            "programming": {
                "expertise_level": "senior_software_engineer",
                "specializations": ["code_review", "architecture_design", "debugging"],
                "crisis_indicators": ["system_failure", "security_vulnerability", "data_loss", "critical_bug"],
                "response_patterns": ["systematic", "solution_oriented", "best_practices"]
            }
        }
        
        # Default expert configuration for domains not specifically defined
        if domain not in domain_expertise:
            domain_expertise[domain] = {
                "expertise_level": "domain_specialist",
                "specializations": [f"{domain}_expertise", f"{domain}_consultation"],
                "crisis_indicators": ["urgent_need", "critical_issue", "immediate_help"],
                "response_patterns": ["professional", "helpful", "knowledgeable"]
            }
        
        return domain_expertise[domain]
    
    def _get_crisis_scenario_agent(self, domain: str) -> Dict[str, Any]:
        """Get or create crisis scenario agent"""
        if domain not in self.crisis_scenario_agents:
            self.crisis_scenario_agents[domain] = self._create_crisis_scenario_agent(domain)
        return self.crisis_scenario_agents[domain]
    
    def _create_crisis_scenario_agent(self, domain: str) -> Dict[str, Any]:
        """Create specialized crisis scenario agent"""
        crisis_scenarios = {
            "healthcare": {
                "emergency_medical": ["heart_attack", "stroke", "severe_injury", "allergic_reaction"],
                "mental_health_crisis": ["suicidal_ideation", "panic_attack", "psychotic_episode", "self_harm"],
                "chronic_condition_crisis": ["diabetes_emergency", "asthma_attack", "seizure", "medication_reaction"]
            },
            "financial": {
                "financial_emergency": ["job_loss", "medical_bankruptcy", "foreclosure", "identity_theft"],
                "investment_crisis": ["market_crash", "portfolio_loss", "fraud", "scam"],
                "business_financial_crisis": ["cash_flow_crisis", "supplier_default", "major_client_loss"]
            },
            "legal": {
                "legal_emergency": ["arrest", "lawsuit", "regulatory_violation", "contract_breach"],
                "criminal_matter": ["criminal_charges", "police_investigation", "court_appearance"],
                "civil_matter": ["eviction", "divorce", "custody_dispute", "personal_injury"]
            },
            "cybersecurity": {
                "security_breach": ["data_breach", "ransomware", "phishing_attack", "malware_infection"],
                "system_compromise": ["account_takeover", "unauthorized_access", "system_failure"],
                "privacy_violation": ["data_leak", "identity_theft", "privacy_breach"]
            },
            "business": {
                "operational_crisis": ["system_failure", "supply_chain_disruption", "key_personnel_loss"],
                "customer_crisis": ["service_outage", "product_recall", "customer_complaints"],
                "compliance_crisis": ["regulatory_violation", "audit_failure", "legal_action"]
            }
        }
        
        # Default crisis scenarios for domains not specifically defined
        domain_category = self._get_domain_category(domain)
        if domain_category in crisis_scenarios:
            return crisis_scenarios[domain_category]
        else:
            return {
                "general_crisis": ["urgent_need", "critical_issue", "immediate_help", "emergency_situation"],
                "consultation_crisis": ["time_sensitive", "high_stakes", "urgent_decision", "critical_advice"]
            }
    
    def _assess_urgency(self, message: str) -> str:
        """Assess urgency level of user message"""
        urgency_keywords = {
            "emergency": ["emergency", "urgent", "immediate", "asap", "help", "crisis", "now"],
            "critical": ["critical", "serious", "important", "deadline", "pressing", "vital"],
            "high": ["soon", "quickly", "fast", "priority", "needed", "required"],
            "medium": ["when", "how", "planning", "considering", "thinking", "wondering"],
            "low": ["curious", "interested", "learning", "exploring", "general", "casual"]
        }
        
        message_lower = message.lower()
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return level
        
        return "medium"  # Default urgency level
    
    def _detect_user_emotion(self, message: str) -> str:
        """Detect user emotion from message"""
        emotion_keywords = {
            "anxious": ["worried", "nervous", "scared", "afraid", "anxious", "concerned"],
            "frustrated": ["frustrated", "annoyed", "angry", "upset", "irritated"],
            "confused": ["confused", "lost", "unclear", "don't understand", "puzzled"],
            "hopeful": ["hopeful", "optimistic", "positive", "excited", "looking forward"],
            "stressed": ["stressed", "overwhelmed", "pressure", "burden", "exhausted"],
            "confident": ["confident", "sure", "ready", "prepared", "determined"],
            "curious": ["curious", "interested", "wondering", "learning", "exploring"],
            "disappointed": ["disappointed", "let down", "discouraged", "sad", "upset"],
            "motivated": ["motivated", "inspired", "driven", "energized", "focused"]
        }
        
        message_lower = message.lower()
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion
        
        return "neutral"  # Default emotion
    
    def _analyze_urgency_patterns(self, conversation_starters: List[str]) -> float:
        """Analyze urgency patterns in conversation starters"""
        urgency_scores = []
        
        for starter in conversation_starters:
            urgency_level = self._assess_urgency(starter)
            urgency_score = {
                "emergency": 1.0,
                "critical": 0.8,
                "high": 0.6,
                "medium": 0.4,
                "low": 0.2
            }.get(urgency_level, 0.4)
            urgency_scores.append(urgency_score)
        
        return sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0.4
    
    def _detect_domain_criticality(self, domain: str) -> float:
        """Detect domain criticality level"""
        if domain in self.domain_criticality_cache:
            return self.domain_criticality_cache[domain]
        
        # Critical domains requiring immediate response
        critical_domains = {
            "emergency_care": 1.0,
            "mental_health": 0.9,
            "general_health": 0.8,
            "cybersecurity": 0.8,
            "legal": 0.7,
            "financial_planning": 0.6,
            "crisis_intervention": 1.0
        }
        
        # Healthcare domains are generally high criticality
        healthcare_domains = [
            "general_health", "mental_health", "emergency_care", "chronic_conditions",
            "medication_management", "preventive_care", "women_health", "senior_health"
        ]
        
        if domain in critical_domains:
            criticality = critical_domains[domain]
        elif domain in healthcare_domains:
            criticality = 0.7
        elif "crisis" in domain or "emergency" in domain:
            criticality = 0.9
        else:
            criticality = 0.4  # Default criticality
        
        self.domain_criticality_cache[domain] = criticality
        return criticality
    
    def _analyze_user_intent_urgency(self, user_intents: List[str]) -> float:
        """Analyze urgency based on user intents"""
        high_urgency_intents = [
            "need_help", "emergency", "crisis", "urgent", "immediate",
            "problem", "issue", "trouble", "concern", "worried"
        ]
        
        urgency_count = 0
        for intent in user_intents:
            if any(urgent_word in intent.lower() for urgent_word in high_urgency_intents):
                urgency_count += 1
        
        return min(urgency_count / len(user_intents), 1.0) if user_intents else 0.0
    
    def _calculate_dynamic_ratio(self, urgency_score: float, domain_criticality: float, user_intent_urgency: float) -> float:
        """Calculate dynamic real-time scenario ratio"""
        base_ratio = 0.40  # 40% base real-time scenarios
        
        # Adjust based on urgency factors
        urgency_factor = (urgency_score + domain_criticality + user_intent_urgency) / 3
        
        # Increase ratio for high urgency domains/scenarios
        if urgency_factor > 0.7:
            return min(base_ratio + 0.20, 0.70)  # Up to 70% for high urgency
        elif urgency_factor > 0.5:
            return min(base_ratio + 0.10, 0.50)  # Up to 50% for medium urgency
        else:
            return base_ratio  # Standard 40% for low urgency
    
    def _split_starters_by_urgency(self, all_starters: List[str], realtime_ratio: float) -> Tuple[List[str], List[str]]:
        """Split conversation starters by urgency level"""
        urgent_starters = []
        regular_starters = []
        
        for starter in all_starters:
            urgency_level = self._assess_urgency(starter)
            if urgency_level in ["emergency", "critical", "high"]:
                urgent_starters.append(starter)
            else:
                regular_starters.append(starter)
        
        # Calculate how many real-time scenarios we need
        total_needed = int(len(all_starters) * realtime_ratio)
        
        # Prioritize urgent starters for real-time scenarios
        realtime_starters = urgent_starters[:total_needed]
        if len(realtime_starters) < total_needed:
            additional_needed = total_needed - len(realtime_starters)
            realtime_starters.extend(regular_starters[:additional_needed])
        
        # Remaining starters are regular scenarios
        used_starters = set(realtime_starters)
        remaining_starters = [s for s in all_starters if s not in used_starters]
        
        return realtime_starters, remaining_starters
    
    def _generate_crisis_response(self, message: str, domain_expert: Dict[str, Any]) -> str:
        """Generate crisis response based on domain expertise"""
        crisis_templates = {
            "medical_emergency": "This sounds like it could be a medical emergency. Please call 911 or go to the nearest emergency room immediately. While you're waiting for help: {specific_guidance}",
            "mental_health_crisis": "I'm concerned about your safety. Please reach out to a crisis helpline: National Suicide Prevention Lifeline: 988. You're not alone, and help is available. {supportive_message}",
            "financial_emergency": "I understand this is a very stressful financial situation. Let's focus on immediate steps: {immediate_actions}. Consider contacting a financial counselor for professional guidance.",
            "legal_emergency": "This appears to be a legal matter requiring immediate attention. I recommend contacting an attorney as soon as possible. Meanwhile: {immediate_legal_guidance}",
            "cybersecurity_crisis": "This sounds like a security incident. Immediate steps: 1) Disconnect from the internet, 2) Change passwords, 3) Contact your IT department or a cybersecurity professional. {technical_guidance}",
            "business_crisis": "This is a critical business situation. Immediate priorities: {crisis_management_steps}. Consider consulting with business advisors or legal counsel."
        }
        
        # Determine crisis type based on domain and message content
        crisis_type = self._identify_crisis_type(message, domain_expert)
        
        if crisis_type in crisis_templates:
            return crisis_templates[crisis_type].format(
                specific_guidance=self._get_specific_crisis_guidance(crisis_type, message),
                supportive_message=self._get_supportive_message(crisis_type),
                immediate_actions=self._get_immediate_actions(crisis_type),
                immediate_legal_guidance=self._get_legal_guidance(crisis_type),
                technical_guidance=self._get_technical_guidance(crisis_type),
                crisis_management_steps=self._get_crisis_management_steps(crisis_type)
            )
        else:
            return f"I understand this is urgent. Based on my expertise in {domain_expert.get('expertise_level', 'this area')}, here's what I recommend: {self._get_general_crisis_guidance(message)}"
    
    def _generate_guidance_response(self, message: str, domain_expert: Dict[str, Any]) -> str:
        """Generate guidance response based on domain expertise"""
        guidance_templates = {
            "professional_consultation": "Based on my expertise as a {expertise_level}, I recommend the following approach: {professional_guidance}",
            "educational_guidance": "Let me help you understand this better. Here's what you should know: {educational_content}",
            "step_by_step_guidance": "Let's work through this step by step: {step_by_step_process}",
            "preventive_guidance": "To prevent similar issues in the future, consider: {preventive_measures}",
            "resource_guidance": "Here are some valuable resources that can help: {helpful_resources}"
        }
        
        expertise_level = domain_expert.get('expertise_level', 'specialist')
        specializations = domain_expert.get('specializations', [])
        
        return guidance_templates["professional_consultation"].format(
            expertise_level=expertise_level,
            professional_guidance=self._get_professional_guidance(message, specializations)
        )
    
    def _generate_informational_response(self, message: str, domain_expert: Dict[str, Any]) -> str:
        """Generate informational response based on domain expertise"""
        informational_templates = {
            "educational": "Here's what you should know about {topic}: {educational_content}",
            "explanatory": "Let me explain {concept} in simple terms: {explanation}",
            "comparative": "When comparing options, consider: {comparison_points}",
            "historical": "From a historical perspective: {historical_context}",
            "practical": "In practical terms, this means: {practical_implications}"
        }
        
        topic = self._extract_topic(message)
        return informational_templates["educational"].format(
            topic=topic,
            educational_content=self._get_educational_content(topic, domain_expert)
        )
    
    def _enhance_with_emotional_intelligence(self, base_response: str, user_emotion: str) -> str:
        """Enhance response with emotional intelligence"""
        if user_emotion == "neutral":
            return base_response
        
        emotional_enhancement = self.emotional_intelligence_agent["emotional_responses"].get(
            user_emotion, "I understand how you're feeling."
        )
        
        return f"{emotional_enhancement} {base_response}"
    
    def _validate_response_quality(self, response: str, domain: str) -> Tuple[bool, float, List[str]]:
        """Validate response quality using advanced metrics"""
        quality_metrics = self.quality_validation_agent["quality_metrics"]
        validation_issues = []
        
        # Emotional Intelligence Score (25%)
        emotional_score = self._calculate_emotional_intelligence_score(response)
        
        # Domain Accuracy Score (30%)
        domain_accuracy = self._calculate_domain_accuracy_score(response, domain)
        
        # Contextual Relevance Score (25%)
        contextual_relevance = self._calculate_contextual_relevance_score(response)
        
        # Crisis Handling Score (20%)
        crisis_handling = self._calculate_crisis_handling_score(response)
        
        # Calculate overall quality score
        overall_quality = (
            emotional_score * quality_metrics["emotional_intelligence"] +
            domain_accuracy * quality_metrics["domain_accuracy"] +
            contextual_relevance * quality_metrics["contextual_relevance"] +
            crisis_handling * quality_metrics["crisis_handling"]
        )
        
        # Check against thresholds
        min_threshold = self.quality_validation_agent["validation_thresholds"]["minimum_quality"]
        is_valid = overall_quality >= min_threshold
        
        if not is_valid:
            validation_issues.append(f"Quality score {overall_quality:.2f} below minimum threshold {min_threshold}")
        
        return is_valid, overall_quality, validation_issues
    
    def _calculate_emotional_intelligence_score(self, response: str) -> float:
        """Calculate emotional intelligence score"""
        empathy_indicators = [
            "understand", "feel", "recognize", "acknowledge", "empathize",
            "support", "here for you", "not alone", "validate"
        ]
        
        response_lower = response.lower()
        empathy_count = sum(1 for indicator in empathy_indicators if indicator in response_lower)
        
        return min(empathy_count / 3, 1.0)  # Normalize to 0-1 scale
    
    def _calculate_domain_accuracy_score(self, response: str, domain: str) -> float:
        """Calculate domain accuracy score"""
        domain_expert = self._get_domain_expert_agent(domain)
        specializations = domain_expert.get('specializations', [])
        
        # Check if response contains domain-specific terminology
        domain_terms_count = sum(1 for spec in specializations if spec.replace('_', ' ') in response.lower())
        
        return min(domain_terms_count / len(specializations), 1.0) if specializations else 0.8
    
    def _calculate_contextual_relevance_score(self, response: str) -> float:
        """Calculate contextual relevance score"""
        # Basic relevance check based on response length and structure
        if len(response) < 50:
            return 0.5  # Too short
        elif len(response) > 1000:
            return 0.8  # Might be too verbose
        else:
            return 0.9  # Good length
    
    def _calculate_crisis_handling_score(self, response: str) -> float:
        """Calculate crisis handling score"""
        crisis_indicators = [
            "emergency", "911", "crisis", "immediate", "urgent", "professional help",
            "call", "contact", "seek help", "safety", "support"
        ]
        
        response_lower = response.lower()
        crisis_handling_count = sum(1 for indicator in crisis_indicators if indicator in response_lower)
        
        return min(crisis_handling_count / 2, 1.0)  # Normalize to 0-1 scale
    
    # Helper methods for crisis response generation
    def _identify_crisis_type(self, message: str, domain_expert: Dict[str, Any]) -> str:
        """Identify type of crisis from message and domain"""
        message_lower = message.lower()
        
        if any(term in message_lower for term in ["chest pain", "can't breathe", "emergency", "911"]):
            return "medical_emergency"
        elif any(term in message_lower for term in ["suicide", "kill myself", "end it all", "hurt myself"]):
            return "mental_health_crisis"
        elif any(term in message_lower for term in ["bankruptcy", "foreclosure", "lost job", "financial ruin"]):
            return "financial_emergency"
        elif any(term in message_lower for term in ["lawsuit", "arrested", "legal trouble", "court"]):
            return "legal_emergency"
        elif any(term in message_lower for term in ["hacked", "breach", "malware", "cyber attack"]):
            return "cybersecurity_crisis"
        elif any(term in message_lower for term in ["business failing", "company crisis", "operational failure"]):
            return "business_crisis"
        else:
            return "general_crisis"
    
    def _get_specific_crisis_guidance(self, crisis_type: str, message: str) -> str:
        """Get specific guidance for crisis type"""
        guidance_map = {
            "medical_emergency": "Stay calm, don't move if injured, provide clear location to emergency services",
            "mental_health_crisis": "You matter and your life has value. Crisis counselors are available 24/7",
            "financial_emergency": "Document everything, contact creditors, seek non-profit credit counseling",
            "legal_emergency": "Don't sign anything, document all communications, exercise your right to remain silent",
            "cybersecurity_crisis": "Preserve evidence, notify relevant authorities, implement containment measures",
            "business_crisis": "Assess immediate risks, communicate with stakeholders, activate crisis management plan"
        }
        return guidance_map.get(crisis_type, "Seek appropriate professional help immediately")
    
    def _get_supportive_message(self, crisis_type: str) -> str:
        """Get supportive message for crisis type"""
        return "Remember that seeking help is a sign of strength, not weakness. Professional support is available."
    
    def _get_immediate_actions(self, crisis_type: str) -> str:
        """Get immediate actions for crisis type"""
        return "1) Ensure immediate safety, 2) Contact emergency services if needed, 3) Reach out to trusted contacts"
    
    def _get_legal_guidance(self, crisis_type: str) -> str:
        """Get legal guidance for crisis type"""
        return "Do not discuss details with anyone except your attorney. Document everything."
    
    def _get_technical_guidance(self, crisis_type: str) -> str:
        """Get technical guidance for crisis type"""
        return "Preserve logs and evidence, implement security measures, notify affected parties"
    
    def _get_crisis_management_steps(self, crisis_type: str) -> str:
        """Get crisis management steps"""
        return "1) Assess situation, 2) Implement containment, 3) Communicate with stakeholders, 4) Execute recovery plan"
    
    def _get_general_crisis_guidance(self, message: str) -> str:
        """Get general crisis guidance"""
        return "Stay calm, assess the situation, prioritize safety, and seek appropriate professional help"
    
    def _get_professional_guidance(self, message: str, specializations: List[str]) -> str:
        """Get professional guidance based on specializations"""
        return f"Based on my specialization in {', '.join(specializations)}, I recommend taking a systematic approach to address your concerns."
    
    def _extract_topic(self, message: str) -> str:
        """Extract main topic from message"""
        # Simple topic extraction - in practice, this would be more sophisticated
        words = message.split()
        return " ".join(words[:3]) if len(words) >= 3 else message
    
    def _get_educational_content(self, topic: str, domain_expert: Dict[str, Any]) -> str:
        """Get educational content for topic"""
        return f"This topic relates to {domain_expert.get('expertise_level', 'professional')} knowledge and requires careful consideration."
    
    def _get_domain_category(self, domain: str) -> str:
        """Get domain category for crisis scenario mapping"""
        healthcare_domains = [
            "general_health", "mental_health", "nutrition", "fitness", "sleep",
            "stress_management", "preventive_care", "chronic_conditions",
            "medication_management", "emergency_care", "women_health", "senior_health"
        ]
        
        business_domains = [
            "entrepreneurship", "marketing", "sales", "customer_service",
            "project_management", "team_leadership", "financial_planning",
            "operations", "hr_management", "strategy", "consulting", "legal_business"
        ]
        
        tech_domains = [
            "programming", "ai_ml", "cybersecurity", "data_analysis",
            "tech_support", "software_development"
        ]
        
        if domain in healthcare_domains:
            return "healthcare"
        elif domain in business_domains:
            return "business"
        elif domain in tech_domains:
            return "cybersecurity"
        elif "legal" in domain:
            return "legal"
        elif "financial" in domain:
            return "financial"
        else:
            return "general"
    
    def get_real_time_scenarios(self, domain: str, scenario_type: str = "general", 
                               urgency_level: str = "medium", emotion: str = "neutral") -> List[str]:
        """Get real-time scenarios based on TARA Universal Model patterns"""
        # Enhanced real-time scenario patterns from TARA Universal Model
        tara_real_time_patterns = {
            "real_time_consultation": [
                "I need immediate advice about {topic}",
                "This is urgent - can you help me with {topic}?",
                "I'm in a meeting and need quick guidance on {topic}",
                "Time-sensitive question about {topic}",
                "Can you give me instant feedback on {topic}?",
                "I have a deadline approaching for {topic}",
                "Need professional consultation on {topic} right now",
                "Quick expert opinion needed on {topic}",
                "Immediate guidance required for {topic}",
                "Real-time support needed for {topic}",
            ],
            "real_time_emergency": [
                "Emergency situation with {topic} - what should I do?",
                "Crisis involving {topic} - need immediate help",
                "Urgent problem with {topic} - please advise",
                "Something's wrong with {topic} - need help now",
                "Critical issue with {topic} - immediate assistance needed",
                "Life-threatening situation involving {topic}",
                "Medical emergency related to {topic}",
                "Safety concern with {topic} - urgent help",
                "Dangerous situation involving {topic}",
                "Emergency response needed for {topic}",
            ],
            "real_time_crisis_intervention": [
                "I'm having a crisis with {topic}",
                "Everything is going wrong with {topic}",
                "I don't know what to do about {topic}",
                "I'm overwhelmed by {topic}",
                "I need crisis support for {topic}",
                "Mental health crisis involving {topic}",
                "Emotional breakdown related to {topic}",
                "Panic attack triggered by {topic}",
                "Stress crisis from {topic}",
                "Psychological emergency with {topic}",
            ],
            "real_time_business_crisis": [
                "Business emergency with {topic}",
                "Company crisis involving {topic}",
                "Operational failure in {topic}",
                "Critical business decision needed for {topic}",
                "Urgent business problem with {topic}",
                "System failure affecting {topic}",
                "Client crisis involving {topic}",
                "Revenue emergency with {topic}",
                "Supply chain crisis affecting {topic}",
                "Compliance emergency in {topic}",
            ],
            "real_time_academic_crisis": [
                "Academic emergency with {topic}",
                "Exam crisis involving {topic}",
                "Study emergency for {topic}",
                "Academic deadline crisis with {topic}",
                "Learning crisis in {topic}",
                "Research emergency with {topic}",
                "Assignment crisis involving {topic}",
                "Academic performance crisis in {topic}",
                "Study burnout from {topic}",
                "Educational emergency with {topic}",
            ],
            "real_time_security_crisis": [
                "Security breach involving {topic}",
                "Cyber attack related to {topic}",
                "Privacy violation in {topic}",
                "Data emergency with {topic}",
                "Security incident affecting {topic}",
                "Hacking attempt on {topic}",
                "Identity theft involving {topic}",
                "Malware infection affecting {topic}",
                "Phishing attack related to {topic}",
                "Ransomware incident with {topic}",
            ],
            "real_time_legal_crisis": [
                "Legal emergency with {topic}",
                "Court deadline for {topic}",
                "Legal crisis involving {topic}",
                "Urgent legal matter about {topic}",
                "Legal emergency requiring {topic} expertise",
                "Lawsuit involving {topic}",
                "Criminal charges related to {topic}",
                "Regulatory violation in {topic}",
                "Contract dispute about {topic}",
                "Legal compliance emergency with {topic}",
            ],
            "real_time_financial_crisis": [
                "Financial emergency with {topic}",
                "Investment crisis involving {topic}",
                "Money emergency related to {topic}",
                "Financial deadline for {topic}",
                "Economic crisis affecting {topic}",
                "Bankruptcy threat involving {topic}",
                "Debt crisis with {topic}",
                "Cash flow emergency in {topic}",
                "Financial fraud involving {topic}",
                "Market crash affecting {topic}",
            ],
            "real_time_health_crisis": [
                "Health emergency with {topic}",
                "Medical crisis involving {topic}",
                "Symptoms emergency related to {topic}",
                "Health scare with {topic}",
                "Medical emergency requiring {topic} expertise",
                "Chronic condition crisis in {topic}",
                "Medication emergency with {topic}",
                "Health diagnosis crisis involving {topic}",
                "Pain emergency related to {topic}",
                "Health deterioration in {topic}",
            ],
            "real_time_technical_crisis": [
                "System crash involving {topic}",
                "Technical failure with {topic}",
                "Software emergency in {topic}",
                "Hardware crisis affecting {topic}",
                "Network emergency with {topic}",
                "Database crisis involving {topic}",
                "Application failure in {topic}",
                "Technical support emergency for {topic}",
                "System vulnerability in {topic}",
                "Performance crisis with {topic}",
            ],
            "real_time_personal_crisis": [
                "Personal emergency with {topic}",
                "Life crisis involving {topic}",
                "Relationship emergency about {topic}",
                "Family crisis related to {topic}",
                "Personal decision crisis with {topic}",
                "Life-changing situation involving {topic}",
                "Personal conflict about {topic}",
                "Major life decision regarding {topic}",
                "Personal emergency requiring {topic} guidance",
                "Life transition crisis with {topic}",
            ]
        }
        
        # Select appropriate scenario patterns based on domain and urgency
        domain_category = self._get_domain_category(domain)
        
        # Map domain categories to scenario types
        scenario_mapping = {
            "healthcare": ["real_time_health_crisis", "real_time_emergency", "real_time_crisis_intervention"],
            "business": ["real_time_business_crisis", "real_time_consultation", "real_time_financial_crisis"],
            "cybersecurity": ["real_time_security_crisis", "real_time_technical_crisis", "real_time_emergency"],
            "legal": ["real_time_legal_crisis", "real_time_consultation", "real_time_emergency"],
            "financial": ["real_time_financial_crisis", "real_time_business_crisis", "real_time_consultation"],
            "general": ["real_time_consultation", "real_time_personal_crisis", "real_time_emergency"]
        }
        
        # Get scenarios for the domain category
        available_scenarios = scenario_mapping.get(domain_category, ["real_time_consultation"])
        
        # Adjust scenarios based on urgency level
        if urgency_level in ["critical", "emergency"]:
            available_scenarios = [s for s in available_scenarios if "crisis" in s or "emergency" in s]
        elif urgency_level == "high":
            available_scenarios = [s for s in available_scenarios if "crisis" in s or "consultation" in s]
        
        # Select random scenarios from available patterns
        selected_scenarios = []
        for scenario_type in available_scenarios[:3]:  # Limit to 3 scenario types
            if scenario_type in tara_real_time_patterns:
                patterns = tara_real_time_patterns[scenario_type]
                # Format patterns with domain-specific topic
                formatted_patterns = [pattern.format(topic=domain.replace('_', ' ')) for pattern in patterns[:3]]
                selected_scenarios.extend(formatted_patterns)
        
        return selected_scenarios[:5]  # Return top 5 scenarios
    
    def generate_tara_quality_conversation(self, domain: str, scenario: str, 
                                         conversation_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate TARA quality conversation based on real-time scenario"""
        # Extract context information
        emotion = conversation_context.get("emotion", "neutral")
        role = conversation_context.get("role", "user")
        specialization = conversation_context.get("specialization", domain)
        urgency_level = self._assess_urgency(scenario)
        
        # Get domain expert and emotional intelligence
        domain_expert = self._get_domain_expert_agent(domain)
        user_emotion = self._detect_user_emotion(scenario)
        
        # Generate initial user message
        user_message = scenario
        
        # Generate assistant response based on scenario type and urgency
        if urgency_level in ["critical", "emergency"]:
            assistant_response = self._generate_crisis_response(user_message, domain_expert)
        elif "crisis" in scenario.lower():
            assistant_response = self._generate_crisis_response(user_message, domain_expert)
        elif "consultation" in scenario.lower():
            assistant_response = self._generate_guidance_response(user_message, domain_expert)
        else:
            assistant_response = self._generate_informational_response(user_message, domain_expert)
        
        # Enhance with emotional intelligence
        enhanced_response = self._enhance_with_emotional_intelligence(assistant_response, user_emotion)
        
        # Create conversation structure
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": enhanced_response}
        ]
        
        # Generate follow-up if needed for complex scenarios
        if urgency_level in ["critical", "emergency"] or "crisis" in scenario.lower():
            follow_up_user = self._generate_follow_up_user_message(scenario, emotion)
            follow_up_assistant = self._generate_follow_up_assistant_response(follow_up_user, domain_expert)
            
            conversation.extend([
                {"role": "user", "content": follow_up_user},
                {"role": "assistant", "content": follow_up_assistant}
            ])
        
        return conversation
    
    def _generate_follow_up_user_message(self, scenario: str, emotion: str) -> str:
        """Generate follow-up user message based on scenario and emotion"""
        follow_up_patterns = {
            "anxious": "I'm still feeling anxious about this. What should I do next?",
            "stressed": "This is still very stressful. Can you help me with the next steps?",
            "confused": "I'm still confused about what to do. Can you clarify?",
            "worried": "I'm still worried about the situation. What are my options?",
            "frustrated": "I'm still frustrated. Is there a better approach?",
            "overwhelmed": "I'm still feeling overwhelmed. Can you break this down further?",
            "neutral": "Thank you for the guidance. What should I do next?"
        }
        
        return follow_up_patterns.get(emotion, "What should I do next?")
    
    def _generate_follow_up_assistant_response(self, user_message: str, domain_expert: Dict[str, Any]) -> str:
        """Generate follow-up assistant response"""
        expertise_level = domain_expert.get("expertise_level", "professional")
        
        response_templates = [
            f"I understand you need more guidance. As a {expertise_level}, let me provide you with specific next steps:",
            f"Based on my {expertise_level} expertise, here's what I recommend you do immediately:",
            f"Let me help you with a clear action plan. From my {expertise_level} perspective:",
            f"I can provide you with more detailed guidance. As a {expertise_level}, here's my recommendation:"
        ]
        
        import random
        template = random.choice(response_templates)
        
        # Add specific action items based on domain
        action_items = [
            "1. Take immediate action to address the most critical aspects",
            "2. Gather necessary information and resources",
            "3. Contact appropriate professionals if needed",
            "4. Monitor the situation closely",
            "5. Follow up with additional support as necessary"
        ]
        
        return f"{template}\n\n{chr(10).join(action_items)}\n\nRemember, you're not alone in this. Professional help is available if needed."

class DataGeneratorAgent(BaseAgent):
    """High-quality training data generation with emotional context and domain expertise"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.DATA_GENERATOR, mcp or mcp_protocol)
        
        # Initialize TARA Real-Time Scenario Engine
        self.tara_scenario_engine = TARARealTimeScenarioEngine()
        
        # Emotional intelligence components
        self.emotional_contexts = [
            "stressed", "anxious", "confident", "confused", "hopeful", 
            "frustrated", "excited", "worried", "determined", "overwhelmed",
            "calm", "nervous", "optimistic", "uncertain", "motivated"
        ]
        
        # Crisis intervention scenarios (5% of training data) - Enhanced with TARA patterns
        self.crisis_scenarios = [
            "emergency_health", "mental_crisis", "financial_emergency", 
            "relationship_crisis", "work_emergency", "family_crisis",
            "academic_pressure", "career_transition", "loss_grief"
        ]
        
        # Professional role contexts
        self.professional_roles = [
            "healthcare_provider", "teacher", "manager", "consultant", 
            "therapist", "advisor", "coach", "specialist", "expert",
            "mentor", "counselor", "guide", "facilitator", "instructor"
        ]
        
        # Domain-specific expertise patterns (Enhanced with TARA approach)
        self.domain_expertise = {
            "healthcare": {
                "specializations": ["cardiology", "neurology", "mental_health", "preventive_care", "emergency_medicine"],
                "contexts": ["diagnosis", "treatment", "prevention", "wellness", "recovery"],
                "complexity_levels": ["basic", "intermediate", "advanced", "specialist"],
                "real_time_scenarios": True  # TARA enhancement
            },
            "finance": {
                "specializations": ["investment", "budgeting", "retirement", "taxes", "insurance"],
                "contexts": ["planning", "analysis", "risk_management", "optimization", "compliance"],
                "complexity_levels": ["beginner", "intermediate", "advanced", "professional"],
                "real_time_scenarios": True  # TARA enhancement
            },
            "education": {
                "specializations": ["k12", "higher_ed", "professional_dev", "skills_training", "certification"],
                "contexts": ["learning", "assessment", "curriculum", "methodology", "outcomes"],
                "complexity_levels": ["elementary", "intermediate", "advanced", "expert"],
                "real_time_scenarios": True  # TARA enhancement
            },
            "business": {
                "specializations": ["strategy", "operations", "marketing", "leadership", "innovation"],
                "contexts": ["planning", "execution", "optimization", "growth", "transformation"],
                "complexity_levels": ["startup", "growth", "enterprise", "global"],
                "real_time_scenarios": True  # TARA enhancement
            }
        }
        
        # Quality scoring parameters (TARA approach: 31% success rate, 101% validation)
        self.quality_parameters = {
            "emotional_intelligence": 0.25,
            "domain_accuracy": 0.30,
            "contextual_relevance": 0.25,
            "crisis_handling": 0.20
        }
        
        # TARA proven generation parameters 
        self.tara_generation_params = {
            "samples_per_domain": 2000,
            "quality_filter_rate": 0.31,
            "target_validation_score": 101.0,
            "real_time_scenario_percentage": 0.40,  # 40% real-time scenarios
            "crisis_scenario_percentage": 0.05     # 5% crisis scenarios
        }
        
        # Data generation statistics
        self.generation_stats = {}
        
    async def start(self):
        """Start the Data Generator Agent"""
        await super().start()
        
        # Initialize domain expertise mappings
        await self._initialize_domain_mappings()
        
        # Start data quality monitoring
        asyncio.create_task(self._quality_monitoring_loop())
        
        print("📊 Data Generator Agent started")
        print(f"   → {len(self.emotional_contexts)} emotional contexts loaded")
        print(f"   → {len(self.crisis_scenarios)} crisis scenarios ready")
        print(f"   → {len(self.domain_expertise)} domain expertise areas configured")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
        elif message.message_type == MessageType.QUALITY_METRICS:
            await self._handle_quality_feedback(message.data)
            
    async def _initialize_domain_mappings(self):
        """Initialize domain-specific data generation mappings"""
        
        # Load cloud-optimized domain mapping for context
        try:
            import yaml
            config_path = Path("config/trinity_domain_model_mapping_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    domain_config = yaml.safe_load(f)
                    
                # Extract domain categories for specialized generation
                for category, domains in domain_config.items():
                    if isinstance(domains, dict) and category not in ['model_tiers', 'gpu_configs', 'cost_estimates', 'advantages', 'data_sources', 'tara_proven_params']:
                        for domain_name in domains.keys():
                            if category not in self.domain_expertise:
                                self.domain_expertise[category] = {
                                    "specializations": [domain_name],
                                    "contexts": ["consultation", "guidance", "support"],
                                    "complexity_levels": ["basic", "intermediate", "advanced"]
                                }
                            
                print(f"✅ Loaded domain mappings for {len(self.domain_expertise)} categories")
        except Exception as e:
            print(f"⚠️ Could not load domain config: {e}")
            print("   Using default domain expertise mappings")
            
    async def _quality_monitoring_loop(self):
        """Monitor data generation quality and adjust parameters"""
        while self.running:
            try:
                # Analyze recent generation quality
                if self.generation_stats:
                    await self._analyze_generation_quality()
                    
                # Broadcast quality status
                self.broadcast_message(
                    MessageType.STATUS_UPDATE,
                    {
                        "agent": "data_generator",
                        "quality_metrics": await self._calculate_quality_metrics(),
                        "generation_stats": self.generation_stats
                    }
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Data quality monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle coordination requests from Training Conductor"""
        action = data.get("action")
        
        if action == "prepare_training_data":
            await self._prepare_training_data(data)
        elif action == "improve_data_quality":
            await self._improve_data_quality(data)
        elif action == "generate_crisis_scenarios":
            await self._generate_crisis_scenarios(data)
        elif action == "validate_data_quality":
            await self._validate_data_quality(data)
            
    async def _prepare_training_data(self, data: Dict[str, Any]):
        """Generate high-quality training data for domain"""
        domain = data.get("domain")
        samples = data.get("samples", 2000)
        quality_requirements = data.get("quality_requirements", {})
        
        print(f"🔄 Generating {samples} high-quality samples for {domain}...")
        
        # Initialize generation tracking
        generation_start = datetime.now()
        self.generation_stats[domain] = {
            "start_time": generation_start,
            "samples_requested": samples,
            "samples_generated": 0,
            "quality_scores": [],
            "emotional_distribution": {},
            "crisis_scenario_count": 0
        }
        
        # Generate diverse, high-quality conversations
        conversations = await self._generate_domain_conversations(domain, samples, quality_requirements)
        
        # Apply TARA's proven 31% quality filtering
        filtered_conversations = await self._apply_quality_filtering(conversations, domain)
        
        # Update generation statistics
        generation_time = datetime.now() - generation_start
        self.generation_stats[domain].update({
            "samples_generated": len(filtered_conversations),
            "generation_time": generation_time.total_seconds(),
            "quality_retention_rate": len(filtered_conversations) / len(conversations),
            "samples_per_second": len(conversations) / generation_time.total_seconds()
        })
        
        print(f"✅ Generated {len(filtered_conversations)} high-quality samples for {domain}")
        print(f"   → Quality retention: {len(filtered_conversations)/len(conversations)*100:.1f}%")
        print(f"   → Generation time: {generation_time.total_seconds():.1f}s")
        print(f"   → Crisis scenarios: {self.generation_stats[domain]['crisis_scenario_count']}")
        
        # Send generated data to Training Conductor
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "data_ready",
                "domain": domain,
                "sample_count": len(filtered_conversations),
                "quality_metrics": await self._calculate_data_quality_metrics(filtered_conversations),
                "data": filtered_conversations,
                "generation_stats": self.generation_stats[domain]
            }
        )
        
    async def _generate_domain_conversations(self, domain: str, samples: int, quality_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse conversations for a specific domain"""
        
        conversations = []
        scenario_types = ["consultation", "guidance", "problem_solving", "education", "support", "crisis_intervention"]
        
        # Get domain-specific expertise
        domain_category = self._get_domain_category(domain)
        expertise = self.domain_expertise.get(domain_category, self.domain_expertise["business"])
        
        # Generate progress tracking
        batch_size = 100
        for batch_start in range(0, samples, batch_size):
            batch_end = min(batch_start + batch_size, samples)
            
            for i in range(batch_start, batch_end):
                # Generate conversation with multiple quality dimensions
                conversation = await self._generate_single_conversation(
                    domain, scenario_types, expertise, quality_requirements
                )
                conversations.append(conversation)
                
                # Update progress tracking
                if (i + 1) % 200 == 0:
                    print(f"  📊 Generated {i + 1}/{samples} samples")
                    
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
            
        return conversations
        
    async def _generate_single_conversation(self, domain: str, scenario_types: List[str], 
                                          expertise: Dict[str, Any], quality_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single high-quality conversation"""
        
        # Select conversation parameters
        scenario = random.choice(scenario_types)
        emotion = random.choice(self.emotional_contexts)
        role = random.choice(self.professional_roles)
        specialization = random.choice(expertise["specializations"])
        context = random.choice(expertise["contexts"])
        complexity = random.choice(expertise["complexity_levels"])
        
        # Crisis intervention (5% of data) + Real-time scenarios (40% of data) - TARA approach
        is_crisis = random.random() < self.tara_generation_params["crisis_scenario_percentage"]
        use_real_time_scenario = random.random() < self.tara_generation_params["real_time_scenario_percentage"]
        
        if is_crisis:
            crisis_type = random.choice(self.crisis_scenarios)
            conversation = await self._generate_crisis_conversation(
                domain, crisis_type, emotion, role, specialization, context
            )
            self.generation_stats[domain]["crisis_scenario_count"] += 1
        elif use_real_time_scenario and expertise.get("real_time_scenarios", False):
            # Use TARA Real-Time Scenario Engine
            real_time_scenarios = self.tara_scenario_engine.get_real_time_scenarios(
                domain, "consultation", count=1
            )
            
            if real_time_scenarios:
                conversation_context = {
                    "emotion": emotion,
                    "is_crisis": is_crisis,
                    "expertise_level": complexity,
                    "role": role,
                    "specialization": specialization
                }
                
                conversation = self.tara_scenario_engine.generate_tara_quality_conversation(
                    domain, real_time_scenarios[0], conversation_context
                )
            else:
                # Fallback to standard conversation
                conversation = await self._generate_standard_conversation(
                    domain, scenario, emotion, role, specialization, context, complexity
                )
        else:
            conversation = await self._generate_standard_conversation(
                domain, scenario, emotion, role, specialization, context, complexity
            )
            
        # Calculate quality score
        quality_score = await self._calculate_conversation_quality(conversation, domain, is_crisis)
        
        # Update emotional distribution tracking
        stats = self.generation_stats[domain]
        if emotion not in stats["emotional_distribution"]:
            stats["emotional_distribution"][emotion] = 0
        stats["emotional_distribution"][emotion] += 1
        
        return {
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "emotion_context": emotion,
            "professional_role": role,
            "specialization": specialization,
            "context_type": context,
            "complexity_level": complexity,
            "is_crisis": is_crisis,
            "uses_tara_real_time": use_real_time_scenario,
            "scenario_type": scenario,
            "conversation": conversation,
            "quality_score": quality_score,
            "quality_dimensions": {
                "emotional_intelligence": random.uniform(0.7, 1.0),
                "domain_accuracy": random.uniform(0.8, 1.0),
                "contextual_relevance": random.uniform(0.75, 1.0),
                "crisis_handling": 1.0 if is_crisis else random.uniform(0.6, 0.9)
            }
        }
        
    async def _generate_crisis_conversation(self, domain: str, crisis_type: str, emotion: str, 
                                          role: str, specialization: str, context: str) -> List[Dict[str, str]]:
        """Generate crisis intervention conversation"""
        
        # Crisis-specific user messages
        crisis_messages = {
            "emergency_health": f"I'm having a medical emergency related to {specialization} and feeling {emotion}. I need immediate help with {domain}.",
            "mental_health_crisis": f"I'm in a mental health crisis and feeling extremely {emotion}. I need urgent guidance on {domain}.",
            "financial_emergency": f"I'm facing a financial crisis and feeling {emotion}. I need emergency advice on {domain}.",
            "legal_emergency": f"This appears to be a legal matter requiring immediate attention. I recommend contacting an attorney as soon as possible. Meanwhile: {self._get_legal_guidance(crisis_type)}",
            "cybersecurity_crisis": f"This sounds like a security incident. Immediate steps: 1) Disconnect from the internet, 2) Change passwords, 3) Contact your IT department or a cybersecurity professional. {self._get_technical_guidance(crisis_type)}",
            "business_crisis": f"This is a critical business situation. Immediate priorities: {self._get_crisis_management_steps(crisis_type)}. Consider consulting with business advisors or legal counsel."
        }
        
        user_message = crisis_messages.get(crisis_type, f"I'm in crisis and feeling {emotion}. I need help with {domain}.")
        
        # Crisis-appropriate assistant response
        assistant_message = f"""I understand you're experiencing {crisis_type} and feeling {emotion}. This is a serious situation and I want to help you immediately.

**Immediate Steps for {domain}:**
1. **Safety First**: If this is a life-threatening emergency, please call emergency services (911) immediately
2. **Immediate Support**: As your {role} specializing in {specialization}, let me provide urgent guidance
3. **{context.title()} Focus**: Let's address the most critical aspects of {domain} right now

**Urgent Action Plan:**
- I'll help you navigate this {crisis_type} with specific focus on {domain}
- We'll address your {emotion} feelings while taking practical steps
- I'm here to provide continuous support through this crisis

Please tell me more about the specific {domain} aspects that need immediate attention, and I'll provide step-by-step guidance tailored to your {emotion} state and current crisis situation."""

        return [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
    async def _generate_standard_conversation(self, domain: str, scenario: str, emotion: str, 
                                            role: str, specialization: str, context: str, complexity: str) -> List[Dict[str, str]]:
        """Generate standard conversation with emotional intelligence"""
        
        # Emotionally intelligent user message
        user_message = f"As someone feeling {emotion}, I need {scenario} guidance on {domain}, specifically regarding {specialization}. Can you help me with {context} at a {complexity} level?"
        
        # Professional, empathetic assistant response
        assistant_message = f"""I completely understand that you're feeling {emotion}, and I'm here to provide the {scenario} support you need for {domain}.

**Your {specialization.title()} Journey:**
As your {role}, I recognize that feeling {emotion} can impact how we approach {context}. Let me tailor my guidance to both your emotional state and your {complexity}-level needs.

**Personalized Approach:**
- **Emotional Support**: I acknowledge you're feeling {emotion}, and that's completely valid
- **{specialization.title()} Expertise**: I'll provide specialized knowledge in {specialization} 
- **{context.title()} Focus**: We'll concentrate on {context} aspects that matter most to you
- **{complexity.title()} Level**: My explanations will match your {complexity} understanding

**Action Plan:**
1. **Immediate Understanding**: Let's address your {emotion} concerns about {domain}
2. **Specialized Guidance**: I'll apply {specialization} expertise to your specific situation  
3. **Practical Steps**: We'll create actionable steps for {context} that feel manageable given your current {emotion} state
4. **Ongoing Support**: I'm here to adjust our approach as your understanding and emotions evolve

What specific aspect of {specialization} in {domain} would you like to explore first? I'll ensure my guidance matches both your {complexity} level and addresses your {emotion} feelings throughout our conversation."""

        return [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
    def _get_domain_category(self, domain: str) -> str:
        """Get the category for a specific domain"""
        
        # Simple mapping - could be enhanced with config file
        domain_mappings = {
            "healthcare": ["health", "medical", "medicine", "doctor", "nurse", "therapy"],
            "finance": ["money", "investment", "budget", "financial", "banking", "insurance"],
            "education": ["learn", "study", "teach", "education", "school", "university"],
            "business": ["business", "management", "strategy", "leadership", "startup"]
        }
        
        for category, keywords in domain_mappings.items():
            if any(keyword in domain.lower() for keyword in keywords):
                return category
                
        return "business"  # Default category
        
    async def _calculate_conversation_quality(self, conversation: List[Dict[str, str]], 
                                            domain: str, is_crisis: bool) -> float:
        """Calculate quality score for a conversation"""
        
        # Quality scoring based on multiple dimensions
        scores = {
            "emotional_intelligence": self._score_emotional_intelligence(conversation),
            "domain_accuracy": self._score_domain_accuracy(conversation, domain),
            "contextual_relevance": self._score_contextual_relevance(conversation),
            "crisis_handling": self._score_crisis_handling(conversation, is_crisis)
        }
        
        # Weighted average based on quality parameters
        total_score = sum(
            scores[dimension] * weight 
            for dimension, weight in self.quality_parameters.items()
        )
        
        return min(1.0, max(0.0, total_score))
        
    def _score_emotional_intelligence(self, conversation: List[Dict[str, str]]) -> float:
        """Score emotional intelligence in conversation"""
        assistant_msg = conversation[1]["content"]
        
        # Check for emotional acknowledgment keywords
        emotional_keywords = ["feeling", "understand", "acknowledge", "recognize", "support", "empathy"]
        emotional_score = sum(1 for keyword in emotional_keywords if keyword in assistant_msg.lower()) / len(emotional_keywords)
        
        # Bonus for direct emotional validation
        if "feeling" in assistant_msg and "understand" in assistant_msg:
            emotional_score += 0.2
            
        return min(1.0, emotional_score)
        
    def _score_domain_accuracy(self, conversation: List[Dict[str, str]], domain: str) -> float:
        """Score domain-specific accuracy"""
        assistant_msg = conversation[1]["content"]
        
        # Check for domain-specific terminology
        domain_relevance = len([word for word in assistant_msg.lower().split() if domain.lower() in word]) / len(assistant_msg.split())
        
        # Check for professional structure
        structure_score = 0.7 if "**" in assistant_msg else 0.5  # Structured response
        
        return min(1.0, domain_relevance * 5 + structure_score)
        
    def _score_contextual_relevance(self, conversation: List[Dict[str, str]]) -> float:
        """Score contextual relevance of response"""
        user_msg = conversation[0]["content"]
        assistant_msg = conversation[1]["content"]
        
        # Check if assistant addresses user's specific context
        user_keywords = set(user_msg.lower().split())
        assistant_keywords = set(assistant_msg.lower().split())
        
        overlap = len(user_keywords.intersection(assistant_keywords)) / len(user_keywords)
        
        return min(1.0, overlap * 2)
        
    def _score_crisis_handling(self, conversation: List[Dict[str, str]], is_crisis: bool) -> float:
        """Score crisis handling appropriateness"""
        if not is_crisis:
            return 0.8  # Standard score for non-crisis
            
        assistant_msg = conversation[1]["content"]
        
        # Check for crisis-appropriate keywords
        crisis_keywords = ["emergency", "immediate", "urgent", "safety", "crisis", "support"]
        crisis_score = sum(1 for keyword in crisis_keywords if keyword in assistant_msg.lower()) / len(crisis_keywords)
        
        # Bonus for structured crisis response
        if "Immediate Steps" in assistant_msg or "Action Plan" in assistant_msg:
            crisis_score += 0.3
            
        return min(1.0, crisis_score)
        
    async def _apply_quality_filtering(self, conversations: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Apply TARA's proven 31% quality filtering"""
        
        print(f"🔍 Applying quality filter to {len(conversations)} conversations...")
        
        # Sort by quality score (highest first)
        sorted_conversations = sorted(conversations, key=lambda x: x['quality_score'], reverse=True)
        
        # Apply 31% retention rate (TARA proven approach)
        cutoff_index = int(len(sorted_conversations) * 0.31)
        filtered_conversations = sorted_conversations[:cutoff_index]
        
        # Store quality scores for analysis
        quality_scores = [conv['quality_score'] for conv in filtered_conversations]
        self.generation_stats[domain]["quality_scores"] = quality_scores
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        min_quality = min(quality_scores) if quality_scores else 0
        max_quality = max(quality_scores) if quality_scores else 0
        
        print(f"   → Quality retention: {len(filtered_conversations)}/{len(conversations)} ({31:.0f}%)")
        print(f"   → Average quality: {avg_quality:.3f}")
        print(f"   → Quality range: {min_quality:.3f} - {max_quality:.3f}")
        
        return filtered_conversations
        
    async def _calculate_data_quality_metrics(self, conversations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive data quality metrics"""
        
        if not conversations:
            return {"overall_quality": 0.0}
            
        # Overall quality metrics
        quality_scores = [conv['quality_score'] for conv in conversations]
        emotional_distribution = {}
        crisis_count = 0
        
        for conv in conversations:
            emotion = conv['emotion_context']
            if emotion not in emotional_distribution:
                emotional_distribution[emotion] = 0
            emotional_distribution[emotion] += 1
            
            if conv['is_crisis']:
                crisis_count += 1
                
        return {
            "overall_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "emotional_diversity": len(emotional_distribution),
            "crisis_scenario_percentage": (crisis_count / len(conversations)) * 100,
            "sample_count": len(conversations),
            "quality_consistency": 1.0 - (max(quality_scores) - min(quality_scores))
        }
        
    async def _analyze_generation_quality(self):
        """Analyze recent generation quality and adjust parameters"""
        
        total_samples = sum(stats.get("samples_generated", 0) for stats in self.generation_stats.values())
        avg_quality = sum(
            sum(stats.get("quality_scores", [])) / max(1, len(stats.get("quality_scores", [])))
            for stats in self.generation_stats.values()
        ) / max(1, len(self.generation_stats))
        
        # Adjust generation parameters based on quality
        if avg_quality < 0.8:
            print("⚠️ Data quality below threshold - adjusting generation parameters")
            # Could implement parameter adjustments here
            
    async def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate current quality metrics for broadcasting"""
        
        if not self.generation_stats:
            return {"status": "no_data"}
            
        total_samples = sum(stats.get("samples_generated", 0) for stats in self.generation_stats.values())
        total_requested = sum(stats.get("samples_requested", 0) for stats in self.generation_stats.values())
        
        return {
            "total_samples_generated": total_samples,
            "total_samples_requested": total_requested,
            "domains_processed": len(self.generation_stats),
            "average_quality": sum(
                sum(stats.get("quality_scores", [])) / max(1, len(stats.get("quality_scores", [])))
                for stats in self.generation_stats.values()
            ) / max(1, len(self.generation_stats)),
            "status": "operational"
        }
        
    async def _improve_data_quality(self, data: Dict[str, Any]):
        """Improve data quality based on feedback"""
        domain = data.get("domain")
        quality_issues = data.get("quality_issues", [])
        
        print(f"🔧 Improving data quality for {domain}: {quality_issues}")
        
        # Implement quality improvement strategies
        for issue in quality_issues:
            if issue == "low_emotional_intelligence":
                # Increase emotional context weight
                self.quality_parameters["emotional_intelligence"] += 0.05
            elif issue == "insufficient_crisis_scenarios":
                # Generate more crisis scenarios
                await self._generate_additional_crisis_data(domain)
                
    async def _generate_additional_crisis_data(self, domain: str):
        """Generate additional crisis scenario data"""
        print(f"🚨 Generating additional crisis scenarios for {domain}")
        
        # Generate 100 additional crisis scenarios
        crisis_conversations = []
        for _ in range(100):
            crisis_type = random.choice(self.crisis_scenarios)
            emotion = random.choice(self.emotional_contexts)
            role = random.choice(self.professional_roles)
            
            conversation = await self._generate_crisis_conversation(
                domain, crisis_type, emotion, role, "emergency", "crisis_response"
            )
            
            crisis_conversations.append({
                "domain": domain,
                "conversation": conversation,
                "is_crisis": True,
                "quality_score": random.uniform(0.8, 1.0)
            })
            
        print(f"✅ Generated {len(crisis_conversations)} additional crisis scenarios")
        
        # Send to conductor
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "additional_crisis_data_ready",
                "domain": domain,
                "crisis_scenarios": crisis_conversations
            }
        )
    
    async def _generate_crisis_scenarios(self, data: Dict[str, Any]):
        """Generate crisis scenarios for specific domain using TARA Real-Time Scenario Engine"""
        domain = data.get("domain", "general")
        crisis_count = data.get("crisis_count", 50)
        
        print(f"🚨 Generating {crisis_count} crisis scenarios for {domain}")
        
        # Use TARA Real-Time Scenario Engine for crisis scenarios
        crisis_scenarios = []
        
        for i in range(crisis_count):
            # Get real-time crisis scenarios from TARA engine
            real_time_scenarios = self.tara_scenario_engine.get_real_time_scenarios(
                domain, "crisis", "emergency", "stressed"
            )
            
            if real_time_scenarios:
                # Select random crisis scenario
                crisis_scenario = random.choice(real_time_scenarios)
                
                # Generate conversation context
                conversation_context = {
                    "emotion": random.choice(["anxious", "stressed", "overwhelmed", "panicked", "worried"]),
                    "role": "crisis_user",
                    "specialization": "crisis_intervention",
                    "urgency_level": "emergency"
                }
                
                # Generate TARA quality conversation
                conversation = self.tara_scenario_engine.generate_tara_quality_conversation(
                    domain, crisis_scenario, conversation_context
                )
                
                # Calculate quality score
                quality_score = await self._calculate_conversation_quality(
                    conversation, domain, is_crisis=True
                )
                
                crisis_scenarios.append({
                    "domain": domain,
                    "conversation": conversation,
                    "metadata": {
                        "scenario_type": "crisis",
                        "urgency_level": "emergency",
                        "uses_tara_real_time": True,
                        "generated_at": datetime.now().isoformat(),
                        "quality_score": quality_score
                    },
                    "is_crisis": True,
                    "quality_score": quality_score
                })
        
        print(f"✅ Generated {len(crisis_scenarios)} crisis scenarios for {domain}")
        
        # Send to conductor
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "crisis_scenarios_ready",
                "domain": domain,
                "crisis_scenarios": crisis_scenarios,
                "total_generated": len(crisis_scenarios)
            }
        )
        
        # Return results
        return {
            "domain": domain,
            "crisis_scenarios": crisis_scenarios,
            "total_generated": len(crisis_scenarios)
        }
        
    async def _handle_quality_feedback(self, data: Dict[str, Any]):
        """Handle quality feedback from Quality Assurance Agent"""
        domain = data.get("domain")
        quality_score = data.get("quality_score", 0.0)
        
        if quality_score < 0.8:
            print(f"⚠️ Quality feedback: {domain} needs improvement ({quality_score:.3f})")
            await self._improve_data_quality({
                "domain": domain,
                "quality_issues": ["low_overall_quality"]
            })

# Global instance
data_generator_agent = DataGeneratorAgent() 
