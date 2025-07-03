"""
MeeTARA Lab - Knowledge Transfer Agent
Shares learning patterns between domains for optimal cross-domain intelligence
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import yaml
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class KnowledgeTransferAgent(BaseAgent):
    """Cross-domain knowledge transfer and pattern sharing for enhanced intelligence"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.KNOWLEDGE_TRANSFER, mcp or mcp_protocol)
        
        # Knowledge representation structures
        self.domain_knowledge_maps = {}  # Domain-specific knowledge patterns
        self.cross_domain_patterns = {}  # Shared patterns across domains
        self.transfer_opportunities = {}  # Identified transfer opportunities
        
        # 62-Domain Configuration
        self.domain_mapping = {}
        self.domain_categories = {}
        self.domain_keywords = {}
        self.domain_compatibility_matrix = {}
        
        # Load domain configuration
        self._load_domain_configuration()
        
        # Transfer learning configuration
        self.transfer_config = {
            "similarity_threshold": 0.7,    # Minimum similarity for transfer
            "transfer_weight": 0.3,         # Weight of transferred knowledge
            "adaptation_learning_rate": 0.1, # Learning rate for adaptation
            "quality_improvement_threshold": 0.05,  # Minimum improvement for transfer
            "max_transfer_domains": 5       # Maximum domains to transfer from
        }
        
        # Knowledge categories for transfer
        self.knowledge_categories = {
            "emotional_intelligence": {
                "patterns": ["empathy_responses", "crisis_handling", "emotional_validation"],
                "transferable": True,
                "weight": 0.4
            },
            "communication_style": {
                "patterns": ["professional_tone", "explanation_clarity", "user_engagement"],
                "transferable": True, 
                "weight": 0.3
            },
            "problem_solving": {
                "patterns": ["systematic_approach", "solution_validation", "alternative_options"],
                "transferable": True,
                "weight": 0.25
            },
            "domain_specific": {
                "patterns": ["terminology", "procedures", "regulations"],
                "transferable": False,
                "weight": 0.05
            }
        }
        
        # Transfer success tracking
        self.transfer_history = []
        self.transfer_success_rates = {}
        
        # Cross-domain intelligence metrics
        self.intelligence_metrics = {
            "knowledge_diversity": 0.0,
            "transfer_efficiency": 0.0,
            "cross_domain_accuracy": 0.0,
            "pattern_recognition": 0.0
        }
        
    def _load_domain_configuration(self):
        """Load 62-domain configuration using centralized domain integration"""
        try:
            # Import centralized domain integration
            import sys
            from pathlib import Path
            import os
            
            # Add project root to path for imports - multiple approaches
            current_dir = Path.cwd()
            project_root = current_dir if (current_dir / "trinity-core").exists() else current_dir.parent
            sys.path.append(str(project_root))
            
            # Try centralized domain integration first
            try:
                from trinity_core.domain_integration import (
                    get_domain_categories, 
                    get_all_domains, 
                    get_domain_stats
                )
                print("✅ Knowledge Transfer: Successfully imported centralized domain integration")
            except ImportError:
                # Fallback import for different environments
                sys.path.append(str(project_root / "trinity-core"))
                from domain_integration import (
                    get_domain_categories, 
                    get_all_domains, 
                    get_domain_stats
                )
                print("✅ Knowledge Transfer: Successfully imported domain integration (fallback)")
            
            # Load domain configuration using centralized approach
            domain_categories = get_domain_categories()
            domain_stats = get_domain_stats()
            
            # Store domain mapping
            self.domain_mapping = domain_categories
            
            # Map each domain to its category
            for category, domains in domain_categories.items():
                for domain in domains:
                    self.domain_categories[domain] = category
                    
            # Initialize domain-specific configurations
            self._initialize_domain_keywords()
            self._initialize_domain_compatibility()
            
            print(f"✅ Knowledge Transfer: Using centralized domain mapping")
            print(f"   → Total domains: {domain_stats['total_domains']}")
            print(f"   → Categories: {domain_stats['total_categories']}")
            print(f"   → Config path: {domain_stats.get('config_path', 'Dynamic')}")
                
        except Exception as e:
            print(f"⚠️ Error loading domain configuration: {e}")
            self._initialize_default_configuration()
            
    def _initialize_default_configuration(self):
        """Initialize default domain configuration if YAML not available"""
        print("❌ CRITICAL: Could not load centralized domain mapping!")
        print("   This is a config-driven system - no hardcoded fallbacks!")
        print("   Please ensure config/trinity_domain_model_mapping_config.yaml exists and is accessible.")
        
        # Instead of hardcoded fallback, raise an exception to force proper config
        raise Exception(
            "Knowledge Transfer Agent requires centralized domain integration. "
            "No hardcoded fallbacks available. Please fix the config file: "
            "config/trinity_domain_model_mapping_config.yaml"
        )
        
    def _initialize_domain_keywords(self):
        """Initialize domain-specific keywords for pattern recognition"""
        # Define keywords per category
        category_keywords = {
            "healthcare": ["health", "medical", "treatment", "diagnosis", "symptom", "patient", "therapy", "clinical"],
            "business": ["business", "strategy", "management", "growth", "profit", "market", "revenue", "client"],
            "education": ["learn", "study", "education", "knowledge", "skill", "teaching", "student", "course"],
            "technology": ["technology", "software", "code", "system", "data", "digital", "programming", "tech"],
            "creative": ["creative", "design", "art", "writing", "content", "story", "visual", "aesthetic"],
            "daily_life": ["daily", "personal", "family", "home", "relationship", "lifestyle", "routine", "social"],
            "specialized": ["legal", "financial", "scientific", "research", "analysis", "professional", "expert", "regulation"]
        }
        
        # Map keywords to all domains in each category
        for category, domains in self.domain_mapping.items():
            keywords = category_keywords.get(category, [category])
            for domain in domains:
                self.domain_keywords[domain] = keywords + [domain.replace('_', ' ')]
                
    def _initialize_domain_compatibility(self):
        """Initialize domain compatibility matrix for knowledge transfer"""
        # Category-based compatibility scores
        category_compatibility = {
            ("healthcare", "healthcare"): 0.95,
            ("healthcare", "specialized"): 0.7,
            ("healthcare", "education"): 0.6,
            ("healthcare", "daily_life"): 0.5,
            ("business", "business"): 0.95,
            ("business", "specialized"): 0.8,
            ("business", "education"): 0.7,
            ("business", "technology"): 0.6,
            ("education", "education"): 0.95,
            ("education", "daily_life"): 0.7,
            ("education", "creative"): 0.6,
            ("technology", "technology"): 0.95,
            ("technology", "business"): 0.6,
            ("technology", "education"): 0.6,
            ("creative", "creative"): 0.95,
            ("creative", "education"): 0.6,
            ("creative", "daily_life"): 0.5,
            ("daily_life", "daily_life"): 0.95,
            ("daily_life", "healthcare"): 0.5,
            ("specialized", "specialized"): 0.95,
            ("specialized", "business"): 0.8,
            ("specialized", "healthcare"): 0.7
        }
        
        # Generate compatibility matrix for all domain pairs
        for domain1 in self.domain_categories:
            for domain2 in self.domain_categories:
                cat1 = self.domain_categories[domain1]
                cat2 = self.domain_categories[domain2]
                
                # Same category gets high compatibility
                if cat1 == cat2:
                    self.domain_compatibility_matrix[(domain1, domain2)] = 0.9
                else:
                    # Use category compatibility or default
                    key1 = (cat1, cat2)
                    key2 = (cat2, cat1)
                    compatibility = category_compatibility.get(key1, category_compatibility.get(key2, 0.3))
                    self.domain_compatibility_matrix[(domain1, domain2)] = compatibility
        
    async def start(self):
        """Start the Knowledge Transfer Agent"""
        await super().start()
        
        # Initialize knowledge transfer systems
        await self._initialize_knowledge_systems()
        
        # Load existing knowledge patterns
        await self._load_existing_patterns()
        
        # Start knowledge monitoring loop
        asyncio.create_task(self._knowledge_monitoring_loop())
        
        # Start transfer opportunity detection
        asyncio.create_task(self._transfer_opportunity_detection_loop())
        
        print("🧠 Knowledge Transfer Agent started")
        print(f"   → Knowledge categories: {len(self.knowledge_categories)}")
        print(f"   → Transfer threshold: {self.transfer_config['similarity_threshold']}")
        print(f"   → Max transfer domains: {self.transfer_config['max_transfer_domains']}")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
        elif message.message_type == MessageType.TRAINING_PROGRESS:
            await self._process_training_progress(message.data)
        elif message.message_type == MessageType.QUALITY_METRICS:
            await self._process_quality_metrics(message.data)
            
    async def _initialize_knowledge_systems(self):
        """Initialize knowledge transfer systems"""
        
        # Create knowledge storage directories
        knowledge_dir = Path("trinity-core/knowledge_bank")
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pattern storage
        self.pattern_storage_path = knowledge_dir / "cross_domain_patterns.json"
        self.transfer_log_path = knowledge_dir / "transfer_history.json"
        
        # Load transfer configuration if available
        try:
            config_path = Path("config/knowledge_transfer.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    transfer_config = json.load(f)
                    self.transfer_config.update(transfer_config.get("transfer", {}))
                    self.knowledge_categories.update(transfer_config.get("categories", {}))
                    print("✅ Knowledge transfer configuration loaded")
        except Exception as e:
            print(f"⚠️ Using default transfer configuration: {e}")
            
    async def _load_existing_patterns(self):
        """Load existing cross-domain patterns"""
        
        try:
            if self.pattern_storage_path.exists():
                with open(self.pattern_storage_path, 'r', encoding='utf-8') as f:
                    stored_patterns = json.load(f)
                    self.cross_domain_patterns = stored_patterns.get("patterns", {})
                    self.intelligence_metrics = stored_patterns.get("metrics", self.intelligence_metrics)
                    print(f"✅ Loaded {len(self.cross_domain_patterns)} cross-domain patterns")
                    
            if self.transfer_log_path.exists():
                with open(self.transfer_log_path, 'r', encoding='utf-8') as f:
                    self.transfer_history = json.load(f)
                    print(f"✅ Loaded {len(self.transfer_history)} transfer records")
                    
        except Exception as e:
            print(f"⚠️ Could not load existing patterns: {e}")
            
    async def _knowledge_monitoring_loop(self):
        """Monitor knowledge patterns and transfer opportunities"""
        while self.running:
            try:
                # Analyze current knowledge state
                await self._analyze_knowledge_patterns()
                
                # Update intelligence metrics
                await self._update_intelligence_metrics()
                
                # Broadcast knowledge status
                self.broadcast_message(
                    MessageType.STATUS_UPDATE,
                    {
                        "agent": "knowledge_transfer",
                        "patterns_tracked": len(self.cross_domain_patterns),
                        "domains_monitored": len(self.domain_knowledge_maps),
                        "transfer_opportunities": len(self.transfer_opportunities),
                        "intelligence_metrics": self.intelligence_metrics
                    }
                )
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                print(f"❌ Knowledge monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _transfer_opportunity_detection_loop(self):
        """Detect opportunities for knowledge transfer"""
        while self.running:
            try:
                # Detect transfer opportunities between domains
                await self._detect_transfer_opportunities()
                
                # Execute high-value transfers
                await self._execute_beneficial_transfers()
                
                # Save patterns and progress
                await self._save_knowledge_patterns()
                
                await asyncio.sleep(120)  # Deep analysis every 2 minutes
                
            except Exception as e:
                print(f"❌ Transfer opportunity detection error: {e}")
                await asyncio.sleep(180)
                
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle coordination requests"""
        action = data.get("action")
        
        if action == "analyze_domain_knowledge":
            await self._analyze_domain_knowledge(data)
        elif action == "identify_transfer_opportunities":
            await self._identify_transfer_opportunities(data)
        elif action == "execute_knowledge_transfer":
            await self._execute_knowledge_transfer(data)
        elif action == "optimize_cross_domain_intelligence":
            await self._optimize_cross_domain_intelligence(data)
            
    async def _analyze_domain_knowledge(self, data: Dict[str, Any]):
        """Analyze knowledge patterns for a specific domain"""
        domain = data.get("domain")
        training_data = data.get("training_data", [])
        quality_metrics = data.get("quality_metrics", {})
        
        print(f"🧠 Analyzing knowledge patterns for {domain}")
        
        # Extract knowledge patterns from training data
        patterns = await self._extract_knowledge_patterns(domain, training_data)
        
        # Store domain knowledge
        self.domain_knowledge_maps[domain] = {
            "patterns": patterns,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(training_data),
            "knowledge_categories": await self._categorize_knowledge(patterns)
        }
        
        # Identify transferable patterns
        transferable_patterns = await self._identify_transferable_patterns(domain, patterns)
        
        print(f"✅ Knowledge analysis completed for {domain}")
        print(f"   → Total patterns: {len(patterns)}")
        print(f"   → Transferable patterns: {len(transferable_patterns)}")
        
        # Send analysis results
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "knowledge_analysis_complete",
                "domain": domain,
                "patterns_extracted": len(patterns),
                "transferable_patterns": len(transferable_patterns),
                "knowledge_categories": self.domain_knowledge_maps[domain]["knowledge_categories"]
            }
        )
        
    async def _extract_knowledge_patterns(self, domain: str, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract knowledge patterns from training data"""
        
        patterns = {
            "emotional_intelligence": {},
            "communication_style": {},
            "problem_solving": {},
            "domain_specific": {}
        }
        
        for sample in training_data:
            conversation = sample.get("conversation", [])
            emotion_context = sample.get("emotion_context", "neutral")
            is_crisis = sample.get("is_crisis", False)
            
            # Extract emotional intelligence patterns
            if emotion_context != "neutral":
                ei_pattern = await self._extract_emotional_pattern(conversation, emotion_context, is_crisis)
                patterns["emotional_intelligence"][emotion_context] = ei_pattern
                
            # Extract communication style patterns
            comm_pattern = await self._extract_communication_pattern(conversation)
            patterns["communication_style"][f"style_{len(patterns['communication_style'])}"] = comm_pattern
            
            # Extract problem-solving patterns
            if len(conversation) >= 2:
                ps_pattern = await self._extract_problem_solving_pattern(conversation)
                patterns["problem_solving"][f"approach_{len(patterns['problem_solving'])}"] = ps_pattern
                
            # Extract domain-specific patterns
            domain_pattern = await self._extract_domain_pattern(conversation, domain)
            patterns["domain_specific"][f"domain_{len(patterns['domain_specific'])}"] = domain_pattern
            
        return patterns
        
    async def _extract_emotional_pattern(self, conversation: List[Dict[str, str]], 
                                       emotion: str, is_crisis: bool) -> Dict[str, Any]:
        """Extract emotional intelligence pattern from conversation"""
        
        if len(conversation) < 2:
            return {}
            
        user_msg = conversation[0]["content"]
        assistant_msg = conversation[1]["content"]
        
        # Analyze emotional response patterns
        emotional_keywords = ["understand", "feeling", "support", "acknowledge", "recognize"]
        emotional_response_count = sum(1 for keyword in emotional_keywords if keyword in assistant_msg.lower())
        
        # Analyze crisis handling if applicable
        crisis_handling = {}
        if is_crisis:
            crisis_keywords = ["immediate", "urgent", "safety", "emergency", "help"]
            crisis_response_count = sum(1 for keyword in crisis_keywords if keyword in assistant_msg.lower())
            crisis_handling = {
                "crisis_keywords_used": crisis_response_count,
                "structured_response": "**" in assistant_msg,  # Check for structured format
                "immediate_action": "immediate" in assistant_msg.lower() or "urgent" in assistant_msg.lower()
            }
            
        return {
            "emotion": emotion,
            "emotional_keywords_used": emotional_response_count,
            "empathy_indicators": emotional_response_count / len(emotional_keywords),
            "is_crisis": is_crisis,
            "crisis_handling": crisis_handling,
            "response_length": len(assistant_msg),
            "emotional_validation": "feeling" in assistant_msg.lower() and emotion in assistant_msg.lower()
        }
        
    async def _extract_communication_pattern(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract communication style pattern"""
        
        if len(conversation) < 2:
            return {}
            
        assistant_msg = conversation[1]["content"]
        
        # Analyze communication characteristics
        structure_indicators = assistant_msg.count("**")  # Bold formatting
        list_indicators = assistant_msg.count("•") + assistant_msg.count("-") + assistant_msg.count("1.")
        question_indicators = assistant_msg.count("?")
        
        return {
            "response_length": len(assistant_msg),
            "structured_format": structure_indicators > 0,
            "uses_lists": list_indicators > 0,
            "asks_questions": question_indicators > 0,
            "professional_tone": "I'll" in assistant_msg or "Let's" in assistant_msg,
            "engagement_level": question_indicators + list_indicators + structure_indicators
        }
        
    async def _extract_problem_solving_pattern(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract problem-solving approach pattern"""
        
        if len(conversation) < 2:
            return {}
            
        user_msg = conversation[0]["content"]
        assistant_msg = conversation[1]["content"]
        
        # Analyze problem-solving approach
        systematic_keywords = ["step", "plan", "approach", "strategy", "process"]
        systematic_indicators = sum(1 for keyword in systematic_keywords if keyword in assistant_msg.lower())
        
        validation_keywords = ["understand", "clarify", "confirm", "verify"]
        validation_indicators = sum(1 for keyword in validation_keywords if keyword in assistant_msg.lower())
        
        return {
            "systematic_approach": systematic_indicators > 0,
            "validation_focus": validation_indicators > 0,
            "provides_alternatives": "option" in assistant_msg.lower() or "alternative" in assistant_msg.lower(),
            "structured_solution": "Action Plan" in assistant_msg or "Steps" in assistant_msg,
            "problem_complexity": len(user_msg.split())
        }
        
    async def _extract_domain_pattern(self, conversation: List[Dict[str, str]], domain: str) -> Dict[str, Any]:
        """Extract domain-specific pattern"""
        
        if len(conversation) < 2:
            return {}
            
        assistant_msg = conversation[1]["content"]
        
        # Use dynamic domain keywords if available, fallback to hardcoded
        if hasattr(self, 'domain_keywords') and domain.lower() in self.domain_keywords:
            relevant_keywords = self.domain_keywords[domain.lower()]
        else:
            # Fallback domain-specific analysis (expanded for all 62 domains)
            domain_keywords = {
                # Healthcare domains
                "general_health": ["health", "medical", "wellness", "symptoms", "diagnosis", "treatment"],
                "mental_health": ["mental", "psychological", "emotional", "therapy", "counseling", "mood"],
                "nutrition": ["nutrition", "diet", "food", "vitamins", "minerals", "eating", "healthy"],
                "fitness": ["fitness", "exercise", "workout", "training", "physical", "strength"],
                "sleep": ["sleep", "rest", "insomnia", "dreams", "bedtime", "circadian"],
                "stress_management": ["stress", "anxiety", "relaxation", "coping", "mindfulness", "calm"],
                "preventive_care": ["prevention", "screening", "checkup", "vaccination", "immunization"],
                "chronic_conditions": ["chronic", "diabetes", "hypertension", "arthritis", "management"],
                "medication_management": ["medication", "drugs", "prescription", "dosage", "side effects"],
                "emergency_care": ["emergency", "urgent", "trauma", "first aid", "critical"],
                "women_health": ["women", "pregnancy", "reproductive", "gynecology", "maternal"],
                "senior_health": ["senior", "elderly", "aging", "geriatric", "older adults"],
                
                # Daily Life domains
                "parenting": ["parenting", "children", "kids", "family", "discipline", "development"],
                "relationships": ["relationship", "marriage", "dating", "communication", "love"],
                "personal_assistant": ["schedule", "organize", "planning", "reminders", "tasks"],
                "communication": ["communication", "conversation", "speaking", "listening", "social"],
                "home_management": ["home", "household", "cleaning", "maintenance", "organization"],
                "shopping": ["shopping", "buying", "purchases", "deals", "products", "budget"],
                "planning": ["planning", "goals", "objectives", "strategy", "timeline", "schedule"],
                "transportation": ["transportation", "travel", "commute", "driving", "public transport"],
                "time_management": ["time", "productivity", "efficiency", "scheduling", "priorities"],
                "decision_making": ["decision", "choice", "options", "analysis", "judgment"],
                "conflict_resolution": ["conflict", "dispute", "resolution", "mediation", "negotiation"],
                "work_life_balance": ["work", "life", "balance", "career", "personal", "harmony"],
                
                # Business domains
                "entrepreneurship": ["entrepreneur", "startup", "business", "innovation", "venture"],
                "marketing": ["marketing", "advertising", "promotion", "brand", "customers"],
                "sales": ["sales", "selling", "revenue", "customers", "deals", "negotiation"],
                "customer_service": ["customer", "service", "support", "satisfaction", "complaints"],
                "project_management": ["project", "management", "timeline", "resources", "deliverables"],
                "team_leadership": ["leadership", "team", "management", "motivation", "delegation"],
                "financial_planning": ["financial", "money", "investment", "budget", "planning"],
                "operations": ["operations", "processes", "efficiency", "workflow", "systems"],
                "hr_management": ["human resources", "hiring", "employees", "performance", "training"],
                "strategy": ["strategy", "planning", "competitive", "analysis", "growth"],
                "consulting": ["consulting", "advice", "expertise", "recommendations", "solutions"],
                "legal_business": ["legal", "compliance", "contracts", "regulations", "business law"],
                
                # Education domains
                "academic_tutoring": ["tutoring", "academic", "subjects", "learning", "teaching"],
                "skill_development": ["skills", "development", "training", "competency", "improvement"],
                "career_guidance": ["career", "job", "profession", "guidance", "counseling"],
                "exam_preparation": ["exam", "test", "preparation", "study", "assessment"],
                "language_learning": ["language", "learning", "vocabulary", "grammar", "fluency"],
                "research_assistance": ["research", "analysis", "sources", "methodology", "data"],
                "study_techniques": ["study", "learning", "memory", "techniques", "methods"],
                "educational_technology": ["education", "technology", "digital", "online", "tools"],
                
                # Creative domains
                "writing": ["writing", "content", "creative", "storytelling", "narrative"],
                "storytelling": ["story", "narrative", "plot", "characters", "creative"],
                "content_creation": ["content", "creation", "digital", "media", "publishing"],
                "social_media": ["social", "media", "platforms", "engagement", "content"],
                "design_thinking": ["design", "thinking", "innovation", "creativity", "problem solving"],
                "photography": ["photography", "visual", "images", "composition", "lighting"],
                "music": ["music", "sound", "composition", "rhythm", "melody", "harmony"],
                "art_appreciation": ["art", "appreciation", "aesthetics", "culture", "creativity"],
                
                # Technology domains
                "programming": ["programming", "code", "software", "development", "algorithms"],
                "ai_ml": ["artificial intelligence", "machine learning", "AI", "ML", "algorithms"],
                "cybersecurity": ["security", "cyber", "protection", "threats", "privacy"],
                "data_analysis": ["data", "analysis", "statistics", "insights", "visualization"],
                "tech_support": ["technical", "support", "troubleshooting", "help", "assistance"],
                "software_development": ["software", "development", "programming", "applications"],
                
                # Specialized domains
                "legal": ["legal", "law", "attorney", "court", "regulations", "compliance"],
                "financial": ["financial", "money", "investment", "banking", "economics"],
                "scientific_research": ["science", "research", "methodology", "experiments", "data"],
                "engineering": ["engineering", "technical", "design", "systems", "solutions"]
            }
            relevant_keywords = domain_keywords.get(domain.lower(), [domain.lower().replace('_', ' ')])
        
        domain_relevance = sum(1 for keyword in relevant_keywords if keyword in assistant_msg.lower())
        
        return {
            "domain": domain,
            "domain_relevance": domain_relevance,
            "uses_domain_terminology": domain_relevance > 0,
            "professional_context": "professional" in assistant_msg.lower() or "expert" in assistant_msg.lower()
        }
        
    async def _categorize_knowledge(self, patterns: Dict[str, Any]) -> Dict[str, int]:
        """Categorize extracted knowledge patterns"""
        
        categories = {}
        
        for category, pattern_dict in patterns.items():
            categories[category] = len(pattern_dict)
            
        return categories
        
    async def _identify_transferable_patterns(self, domain: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Identify which patterns are transferable to other domains"""
        
        transferable = {}
        
        for category, config in self.knowledge_categories.items():
            if config["transferable"] and category in patterns:
                transferable[category] = patterns[category]
                
        return transferable
        
    async def _detect_transfer_opportunities(self):
        """Detect opportunities for knowledge transfer between domains"""
        
        if len(self.domain_knowledge_maps) < 2:
            return  # Need at least 2 domains for transfer
            
        domains = list(self.domain_knowledge_maps.keys())
        
        for i, source_domain in enumerate(domains):
            for target_domain in domains[i+1:]:
                # Calculate similarity between domains
                similarity = await self._calculate_domain_similarity(source_domain, target_domain)
                
                if similarity >= self.transfer_config["similarity_threshold"]:
                    opportunity_id = f"{source_domain}_to_{target_domain}"
                    
                    self.transfer_opportunities[opportunity_id] = {
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "similarity_score": similarity,
                        "potential_benefit": await self._estimate_transfer_benefit(source_domain, target_domain),
                        "transfer_patterns": await self._identify_patterns_to_transfer(source_domain, target_domain),
                        "detected_at": datetime.now().isoformat()
                    }
                    
        if self.transfer_opportunities:
            print(f"🔍 Detected {len(self.transfer_opportunities)} transfer opportunities")
            
    async def _calculate_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """Calculate similarity between two domains"""
        
        source_patterns = self.domain_knowledge_maps[source_domain]["patterns"]
        target_patterns = self.domain_knowledge_maps[target_domain]["patterns"]
        
        similarities = []
        
        # Compare each knowledge category
        for category in self.knowledge_categories:
            if category in source_patterns and category in target_patterns:
                category_similarity = await self._calculate_pattern_similarity(
                    source_patterns[category], 
                    target_patterns[category]
                )
                weight = self.knowledge_categories[category]["weight"]
                similarities.append(category_similarity * weight)
                
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    async def _calculate_pattern_similarity(self, source_patterns: Dict, target_patterns: Dict) -> float:
        """Calculate similarity between pattern dictionaries"""
        
        if not source_patterns or not target_patterns:
            return 0.0
            
        # Simple similarity calculation based on common patterns
        source_keys = set(source_patterns.keys())
        target_keys = set(target_patterns.keys())
        
        common_keys = source_keys.intersection(target_keys)
        all_keys = source_keys.union(target_keys)
        
        if not all_keys:
            return 0.0
            
        jaccard_similarity = len(common_keys) / len(all_keys)
        
        # Additional similarity based on pattern values (simplified)
        value_similarities = []
        for key in common_keys:
            if isinstance(source_patterns[key], dict) and isinstance(target_patterns[key], dict):
                source_values = list(source_patterns[key].values())
                target_values = list(target_patterns[key].values())
                
                # Simple numeric similarity for numeric values
                numeric_source = [v for v in source_values if isinstance(v, (int, float))]
                numeric_target = [v for v in target_values if isinstance(v, (int, float))]
                
                if numeric_source and numeric_target:
                    avg_source = sum(numeric_source) / len(numeric_source)
                    avg_target = sum(numeric_target) / len(numeric_target)
                    
                    if avg_source + avg_target > 0:
                        similarity = 1 - abs(avg_source - avg_target) / (avg_source + avg_target)
                        value_similarities.append(similarity)
                        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.5
        
        # Combine Jaccard and value similarities
        return (jaccard_similarity + value_similarity) / 2
        
    async def _estimate_transfer_benefit(self, source_domain: str, target_domain: str) -> float:
        """Estimate potential benefit of knowledge transfer"""
        
        source_quality = self.domain_knowledge_maps[source_domain]["quality_metrics"].get("overall_quality", 0.0)
        target_quality = self.domain_knowledge_maps[target_domain]["quality_metrics"].get("overall_quality", 0.0)
        
        # Higher benefit if source is significantly better than target
        quality_difference = source_quality - target_quality
        
        # Consider domain compatibility
        domain_compatibility = await self._calculate_domain_compatibility(source_domain, target_domain)
        
        # Estimate benefit as combination of quality gap and compatibility
        estimated_benefit = (quality_difference * 0.7) + (domain_compatibility * 0.3)
        
        return max(0.0, min(1.0, estimated_benefit))
        
    async def _calculate_domain_compatibility(self, source_domain: str, target_domain: str) -> float:
        """Calculate compatibility between domains for knowledge transfer"""
        
        # Use dynamic domain compatibility matrix if available
        if hasattr(self, 'domain_compatibility_matrix'):
            key1 = (source_domain.lower(), target_domain.lower())
            key2 = (target_domain.lower(), source_domain.lower())
            
            return self.domain_compatibility_matrix.get(key1, self.domain_compatibility_matrix.get(key2, 0.5))
        
        # Fallback domain compatibility matrix (expanded for all 62 domains)
        compatibility_matrix = {
            # Healthcare cross-domain compatibility
            ("healthcare", "mental_health"): 0.9, ("healthcare", "nutrition"): 0.8,
            ("healthcare", "fitness"): 0.7, ("healthcare", "sleep"): 0.8,
            ("healthcare", "stress_management"): 0.85, ("healthcare", "chronic_conditions"): 0.95,
            ("healthcare", "emergency_care"): 0.9, ("healthcare", "women_health"): 0.9,
            ("healthcare", "senior_health"): 0.9, ("healthcare", "preventive_care"): 0.9,
            
            # Business cross-domain compatibility
            ("business", "entrepreneurship"): 0.95, ("business", "marketing"): 0.9,
            ("business", "sales"): 0.9, ("business", "customer_service"): 0.8,
            ("business", "project_management"): 0.85, ("business", "team_leadership"): 0.8,
            ("business", "financial_planning"): 0.9, ("business", "operations"): 0.9,
            ("business", "hr_management"): 0.8, ("business", "strategy"): 0.95,
            
            # Education cross-domain compatibility
            ("education", "academic_tutoring"): 0.95, ("education", "skill_development"): 0.9,
            ("education", "career_guidance"): 0.8, ("education", "exam_preparation"): 0.9,
            ("education", "language_learning"): 0.8, ("education", "research_assistance"): 0.9,
            ("education", "study_techniques"): 0.9, ("education", "educational_technology"): 0.85,
            
            # Technology cross-domain compatibility
            ("technology", "programming"): 0.95, ("technology", "ai_ml"): 0.9,
            ("technology", "cybersecurity"): 0.85, ("technology", "data_analysis"): 0.9,
            ("technology", "tech_support"): 0.8, ("technology", "software_development"): 0.95,
            
            # Inter-category compatibility
            ("healthcare", "education"): 0.6, ("business", "education"): 0.7,
            ("finance", "healthcare"): 0.4, ("technology", "business"): 0.6,
            ("creative", "education"): 0.6, ("daily_life", "healthcare"): 0.5,
            ("specialized", "business"): 0.8, ("specialized", "healthcare"): 0.7
        }
        
        # Check both directions
        key1 = (source_domain.lower(), target_domain.lower())
        key2 = (target_domain.lower(), source_domain.lower())
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.5))
        
    async def _identify_patterns_to_transfer(self, source_domain: str, target_domain: str) -> List[str]:
        """Identify specific patterns that should be transferred"""
        
        source_patterns = self.domain_knowledge_maps[source_domain]["patterns"]
        
        patterns_to_transfer = []
        
        # Focus on transferable categories
        for category, config in self.knowledge_categories.items():
            if config["transferable"] and category in source_patterns:
                # Identify high-quality patterns in this category
                category_patterns = source_patterns[category]
                
                for pattern_id, pattern_data in category_patterns.items():
                    # Simple quality check (could be more sophisticated)
                    if isinstance(pattern_data, dict):
                        quality_indicators = pattern_data.get("empathy_indicators", 0) + pattern_data.get("engagement_level", 0)
                        if quality_indicators > 1:  # Arbitrary threshold
                            patterns_to_transfer.append(f"{category}.{pattern_id}")
                            
        return patterns_to_transfer
        
    async def _execute_beneficial_transfers(self):
        """Execute knowledge transfers that show high potential benefit"""
        
        # Sort opportunities by potential benefit
        sorted_opportunities = sorted(
            self.transfer_opportunities.items(),
            key=lambda x: x[1]["potential_benefit"],
            reverse=True
        )
        
        # Execute top opportunities
        for opportunity_id, opportunity in sorted_opportunities[:3]:  # Top 3 opportunities
            if opportunity["potential_benefit"] > self.transfer_config["quality_improvement_threshold"]:
                await self._execute_knowledge_transfer_internal(opportunity_id, opportunity)
                
    async def _execute_knowledge_transfer_internal(self, opportunity_id: str, opportunity: Dict[str, Any]):
        """Execute a specific knowledge transfer"""
        
        source_domain = opportunity["source_domain"]
        target_domain = opportunity["target_domain"]
        patterns_to_transfer = opportunity["transfer_patterns"]
        
        print(f"🔄 Executing knowledge transfer: {source_domain} → {target_domain}")
        print(f"   → Patterns to transfer: {len(patterns_to_transfer)}")
        
        # Create transfer record
        transfer_record = {
            "transfer_id": opportunity_id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "patterns_transferred": patterns_to_transfer,
            "similarity_score": opportunity["similarity_score"],
            "expected_benefit": opportunity["potential_benefit"],
            "transfer_time": datetime.now().isoformat(),
            "status": "executed"
        }
        
        # Add to transfer history
        self.transfer_history.append(transfer_record)
        
        # Update cross-domain patterns
        await self._update_cross_domain_patterns(source_domain, target_domain, patterns_to_transfer)
        
        # Notify Training Conductor about transfer
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "knowledge_transfer_executed",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "patterns_count": len(patterns_to_transfer),
                "expected_improvement": opportunity["potential_benefit"]
            }
        )
        
        print(f"✅ Knowledge transfer completed: {source_domain} → {target_domain}")
        
    async def _update_cross_domain_patterns(self, source_domain: str, target_domain: str, patterns: List[str]):
        """Update cross-domain pattern knowledge base"""
        
        pattern_key = f"{source_domain}_to_{target_domain}"
        
        self.cross_domain_patterns[pattern_key] = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "transferred_patterns": patterns,
            "transfer_timestamp": datetime.now().isoformat(),
            "usage_count": 0
        }
        
    async def _analyze_knowledge_patterns(self):
        """Analyze current knowledge patterns for insights"""
        
        if not self.domain_knowledge_maps:
            return
            
        # Analyze pattern diversity
        all_patterns = []
        for domain_map in self.domain_knowledge_maps.values():
            for category_patterns in domain_map["patterns"].values():
                all_patterns.extend(category_patterns.keys())
                
        unique_patterns = set(all_patterns)
        pattern_diversity = len(unique_patterns) / max(1, len(all_patterns))
        
        # Update intelligence metrics
        self.intelligence_metrics["knowledge_diversity"] = pattern_diversity
        
    async def _update_intelligence_metrics(self):
        """Update cross-domain intelligence metrics"""
        
        # Transfer efficiency
        successful_transfers = len([t for t in self.transfer_history if t["status"] == "executed"])
        total_opportunities = len(self.transfer_opportunities)
        transfer_efficiency = successful_transfers / max(1, total_opportunities)
        
        # Pattern recognition (based on identified patterns)
        total_patterns = sum(
            sum(len(category) for category in domain_map["patterns"].values())
            for domain_map in self.domain_knowledge_maps.values()
        )
        pattern_recognition = min(1.0, total_patterns / 100)  # Normalize to max 1.0
        
        self.intelligence_metrics.update({
            "transfer_efficiency": transfer_efficiency,
            "pattern_recognition": pattern_recognition,
            "domains_connected": len(self.domain_knowledge_maps)
        })
        
    async def _save_knowledge_patterns(self):
        """Save knowledge patterns and metrics to storage"""
        
        try:
            # Save cross-domain patterns
            pattern_data = {
                "patterns": self.cross_domain_patterns,
                "metrics": self.intelligence_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.pattern_storage_path, 'w', encoding='utf-8') as f:
                json.dump(pattern_data, f, indent=2)
                
            # Save transfer history
            with open(self.transfer_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.transfer_history, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save knowledge patterns: {e}")

# Global instance
knowledge_transfer_agent = KnowledgeTransferAgent() 
