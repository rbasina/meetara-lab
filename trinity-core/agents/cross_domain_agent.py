"""
MeeTARA Lab - Cross-Domain Intelligence Agent
Handles multi-domain queries and intelligent routing decisions
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class CrossDomainAgent(BaseAgent):
    """Cross-domain intelligence for multi-domain queries and smart routing"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.CROSS_DOMAIN, mcp or mcp_protocol)
        
        # Domain intelligence mapping
        self.domain_intelligence = {}  # Domain-specific intelligence models
        self.query_analysis_cache = {}  # Cache for query analysis
        self.routing_decisions = {}    # Historical routing decisions
        
        # Multi-domain query processing
        self.multi_domain_config = {
            "max_domains_per_query": 4,     # Maximum domains to activate
            "confidence_threshold": 0.7,    # Minimum confidence for domain selection
            "fusion_weight_threshold": 0.3, # Minimum weight for domain fusion
            "context_window_size": 1000,    # Context window for analysis
            "response_synthesis_method": "weighted_fusion"  # Method for combining responses
        }
        
        # Domain detection patterns
        self.domain_patterns = {
            "healthcare": {
                "keywords": ["health", "medical", "doctor", "hospital", "medicine", "treatment", "diagnosis", "symptom", "patient"],
                "phrases": ["medical condition", "health issue", "see a doctor", "medical advice", "health concern"],
                "context_indicators": ["pain", "illness", "recovery", "medication", "therapy"]
            },
            "finance": {
                "keywords": ["money", "finance", "investment", "budget", "savings", "loan", "credit", "debt", "income"],
                "phrases": ["financial planning", "investment strategy", "save money", "financial advice", "budget planning"],
                "context_indicators": ["dollars", "cost", "price", "afford", "expense"]
            },
            "education": {
                "keywords": ["learn", "study", "education", "school", "university", "course", "knowledge", "skill", "training"],
                "phrases": ["learning process", "study guide", "educational resource", "skill development", "knowledge base"],
                "context_indicators": ["teach", "understand", "practice", "homework", "exam"]
            },
            "business": {
                "keywords": ["business", "company", "management", "strategy", "marketing", "sales", "profit", "growth", "leadership"],
                "phrases": ["business strategy", "market analysis", "company growth", "business plan", "team management"],
                "context_indicators": ["employees", "customers", "revenue", "competition", "market"]
            },
            "legal": {
                "keywords": ["legal", "law", "lawyer", "court", "contract", "rights", "lawsuit", "attorney", "litigation"],
                "phrases": ["legal advice", "legal issue", "court case", "legal rights", "contract review"],
                "context_indicators": ["sue", "legal", "attorney", "court", "settlement"]
            },
            "mental_health": {
                "keywords": ["stress", "anxiety", "depression", "mental", "therapy", "counseling", "psychology", "emotional"],
                "phrases": ["mental health", "emotional support", "feeling anxious", "mental wellness", "psychological help"],
                "context_indicators": ["worried", "stressed", "anxious", "depressed", "overwhelmed"]
            }
        }
        
        # Response fusion strategies
        self.fusion_strategies = {
            "weighted_fusion": self._weighted_response_fusion,
            "hierarchical_fusion": self._hierarchical_response_fusion,
            "context_aware_fusion": self._context_aware_response_fusion,
            "quality_based_fusion": self._quality_based_response_fusion
        }
        
        # Cross-domain intelligence metrics
        self.intelligence_metrics = {
            "domain_detection_accuracy": 0.0,
            "routing_efficiency": 0.0,
            "multi_domain_success_rate": 0.0,
            "response_quality_scores": [],
            "average_response_time": 0.0
        }
        
    async def start(self):
        """Start the Cross-Domain Intelligence Agent"""
        await super().start()
        
        # Initialize cross-domain systems
        await self._initialize_cross_domain_systems()
        
        # Load domain intelligence models
        await self._load_domain_intelligence()
        
        # Start intelligence monitoring
        asyncio.create_task(self._intelligence_monitoring_loop())
        
        # Start query analysis optimization
        asyncio.create_task(self._query_analysis_optimization_loop())
        
        print("🌐 Cross-Domain Intelligence Agent started")
        print(f"   → Domains configured: {len(self.domain_patterns)}")
        print(f"   → Max domains per query: {self.multi_domain_config['max_domains_per_query']}")
        print(f"   → Fusion strategies: {len(self.fusion_strategies)}")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
        elif message.message_type == MessageType.QUERY_ROUTING:  # Custom message type for routing
            await self._handle_query_routing(message.data)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._process_agent_status(message.data)
            
    async def _initialize_cross_domain_systems(self):
        """Initialize cross-domain intelligence systems"""
        
        # Create intelligence storage
        intelligence_dir = Path("trinity-core/cross_domain_intelligence")
        intelligence_dir.mkdir(parents=True, exist_ok=True)
        
        self.intelligence_storage_path = intelligence_dir / "domain_intelligence.json"
        self.routing_history_path = intelligence_dir / "routing_history.json"
        self.query_cache_path = intelligence_dir / "query_analysis_cache.json"
        
        # Load configuration if available
        try:
            config_path = Path("config/cross_domain_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cross_domain_config = json.load(f)
                    self.multi_domain_config.update(cross_domain_config.get("config", {}))
                    
                    # Update domain patterns if provided
                    if "domain_patterns" in cross_domain_config:
                        self.domain_patterns.update(cross_domain_config["domain_patterns"])
                        
                    print("✅ Cross-domain configuration loaded")
        except Exception as e:
            print(f"⚠️ Using default cross-domain configuration: {e}")
            
    async def _load_domain_intelligence(self):
        """Load existing domain intelligence models"""
        
        try:
            if self.intelligence_storage_path.exists():
                with open(self.intelligence_storage_path, 'r') as f:
                    stored_intelligence = json.load(f)
                    self.domain_intelligence = stored_intelligence.get("intelligence", {})
                    self.intelligence_metrics = stored_intelligence.get("metrics", self.intelligence_metrics)
                    print(f"✅ Loaded intelligence for {len(self.domain_intelligence)} domains")
                    
            if self.routing_history_path.exists():
                with open(self.routing_history_path, 'r') as f:
                    self.routing_decisions = json.load(f)
                    print(f"✅ Loaded {len(self.routing_decisions)} routing decisions")
                    
            if self.query_cache_path.exists():
                with open(self.query_cache_path, 'r') as f:
                    self.query_analysis_cache = json.load(f)
                    print(f"✅ Loaded {len(self.query_analysis_cache)} cached analyses")
                    
        except Exception as e:
            print(f"⚠️ Could not load existing intelligence: {e}")
            
    async def _intelligence_monitoring_loop(self):
        """Monitor cross-domain intelligence performance"""
        while self.running:
            try:
                # Analyze recent routing decisions
                await self._analyze_routing_performance()
                
                # Update intelligence metrics
                await self._update_intelligence_metrics()
                
                # Optimize domain patterns based on performance
                await self._optimize_domain_patterns()
                
                # Broadcast intelligence status
                self.broadcast_message(
                    MessageType.STATUS_UPDATE,
                    {
                        "agent": "cross_domain",
                        "domains_active": len(self.domain_intelligence),
                        "routing_decisions": len(self.routing_decisions),
                        "intelligence_metrics": self.intelligence_metrics,
                        "cache_size": len(self.query_analysis_cache)
                    }
                )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                print(f"❌ Intelligence monitoring error: {e}")
                await asyncio.sleep(90)
                
    async def _query_analysis_optimization_loop(self):
        """Optimize query analysis based on historical performance"""
        while self.running:
            try:
                # Clean old cache entries
                await self._clean_query_cache()
                
                # Optimize domain detection patterns
                await self._optimize_detection_patterns()
                
                # Save intelligence data
                await self._save_intelligence_data()
                
                await asyncio.sleep(300)  # Deep optimization every 5 minutes
                
            except Exception as e:
                print(f"❌ Query analysis optimization error: {e}")
                await asyncio.sleep(600)
                
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle coordination requests"""
        action = data.get("action")
        
        if action == "analyze_query_domains":
            await self._analyze_query_domains(data)
        elif action == "route_multi_domain_query":
            await self._route_multi_domain_query(data)
        elif action == "synthesize_domain_responses":
            await self._synthesize_domain_responses(data)
        elif action == "optimize_cross_domain_intelligence":
            await self._optimize_cross_domain_intelligence(data)
            
    async def _analyze_query_domains(self, data: Dict[str, Any]):
        """Analyze query to identify relevant domains"""
        query = data.get("query", "")
        context = data.get("context", "")
        user_profile = data.get("user_profile", {})
        
        print(f"🔍 Analyzing query domains: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.query_analysis_cache:
            cached_result = self.query_analysis_cache[cache_key]
            cached_result["cache_hit"] = True
            
            print(f"✅ Cache hit - Domains: {cached_result['domains']}")
            
            # Send cached analysis
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.STATUS_UPDATE,
                {
                    "action": "query_analysis_complete",
                    "analysis": cached_result
                }
            )
            return
            
        # Perform fresh analysis
        analysis = await self._perform_query_analysis(query, context, user_profile)
        
        # Cache the result
        self.query_analysis_cache[cache_key] = analysis
        
        print(f"✅ Query analysis completed")
        print(f"   → Primary domains: {analysis['primary_domains']}")
        print(f"   → Secondary domains: {analysis['secondary_domains']}")
        print(f"   → Confidence scores: {analysis['confidence_scores']}")
        
        # Send analysis results
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "query_analysis_complete",
                "analysis": analysis
            }
        )
        
    async def _perform_query_analysis(self, query: str, context: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive query analysis"""
        
        full_text = f"{context} {query}".strip()
        
        # Domain detection
        domain_scores = await self._calculate_domain_scores(full_text)
        
        # Query complexity analysis
        complexity = await self._analyze_query_complexity(query)
        
        # Multi-domain detection
        is_multi_domain = len([score for score in domain_scores.values() if score > self.multi_domain_config['confidence_threshold']]) > 1
        
        # Primary and secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domains = [domain for domain, score in sorted_domains if score > self.multi_domain_config['confidence_threshold']][:self.multi_domain_config['max_domains_per_query']]
        secondary_domains = [domain for domain, score in sorted_domains if 0.3 <= score <= self.multi_domain_config['confidence_threshold']][:2]
        
        # Intent analysis
        intent = await self._analyze_query_intent(query)
        
        # Emotional context
        emotional_context = await self._analyze_emotional_context(query)
        
        # Routing strategy
        routing_strategy = await self._determine_routing_strategy(primary_domains, complexity, intent)
        
        analysis = {
            "query": query,
            "context": context,
            "primary_domains": primary_domains,
            "secondary_domains": secondary_domains,
            "confidence_scores": domain_scores,
            "is_multi_domain": is_multi_domain,
            "complexity": complexity,
            "intent": intent,
            "emotional_context": emotional_context,
            "routing_strategy": routing_strategy,
            "analysis_timestamp": datetime.now().isoformat(),
            "cache_hit": False
        }
        
        return analysis
        
    async def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate domain relevance scores for given text"""
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in text_lower)
            keyword_score = min(1.0, keyword_matches / len(patterns["keywords"]))
            
            # Phrase matching (higher weight)
            phrase_matches = sum(1 for phrase in patterns["phrases"] if phrase in text_lower)
            phrase_score = min(1.0, phrase_matches / len(patterns["phrases"]) * 2)  # Double weight
            
            # Context indicator matching
            context_matches = sum(1 for indicator in patterns["context_indicators"] if indicator in text_lower)
            context_score = min(1.0, context_matches / len(patterns["context_indicators"]))
            
            # Weighted combination
            total_score = (keyword_score * 0.4) + (phrase_score * 0.4) + (context_score * 0.2)
            
            # Apply domain-specific intelligence if available
            if domain in self.domain_intelligence:
                intelligence_boost = self.domain_intelligence[domain].get("detection_accuracy", 0.0) * 0.1
                total_score += intelligence_boost
                
            domain_scores[domain] = min(1.0, total_score)
            
        return domain_scores
        
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of the query"""
        
        word_count = len(query.split())
        sentence_count = len(re.split(r'[.!?]+', query))
        question_count = query.count('?')
        
        # Complexity indicators
        complex_indicators = ['how', 'why', 'what if', 'compare', 'analyze', 'explain', 'detailed']
        complexity_score = sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        # Determine complexity level
        if word_count < 10 and complexity_score == 0:
            complexity_level = "simple"
        elif word_count < 25 and complexity_score <= 1:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
            
        return {
            "level": complexity_level,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "question_count": question_count,
            "complexity_score": complexity_score
        }
        
    async def _analyze_query_intent(self, query: str) -> str:
        """Analyze the intent behind the query"""
        
        query_lower = query.lower()
        
        # Intent patterns
        intent_patterns = {
            "information_seeking": ["what", "how", "when", "where", "who", "tell me", "explain"],
            "problem_solving": ["help", "solve", "fix", "resolve", "issue", "problem"],
            "comparison": ["compare", "vs", "versus", "difference", "better", "best"],
            "planning": ["plan", "strategy", "approach", "steps", "guide"],
            "advice_seeking": ["should", "recommend", "suggest", "advice", "opinion"],
            "emergency": ["urgent", "emergency", "crisis", "immediate", "asap"]
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
            
        # Return the intent with highest score
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return "information_seeking"  # Default intent
            
    async def _analyze_emotional_context(self, query: str) -> Dict[str, Any]:
        """Analyze emotional context of the query"""
        
        query_lower = query.lower()
        
        # Emotional indicators
        emotional_patterns = {
            "stressed": ["stressed", "overwhelmed", "pressure", "urgent", "worried"],
            "confused": ["confused", "don't understand", "unclear", "lost", "help"],
            "confident": ["confident", "sure", "certain", "ready", "prepared"],
            "anxious": ["anxious", "nervous", "scared", "afraid", "worried"],
            "frustrated": ["frustrated", "annoyed", "stuck", "can't", "unable"],
            "hopeful": ["hope", "optimistic", "positive", "excited", "looking forward"]
        }
        
        emotion_scores = {}
        for emotion, patterns in emotional_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            emotion_scores[emotion] = score
            
        # Determine primary emotion
        primary_emotion = "neutral"
        if emotion_scores:
            max_score = max(emotion_scores.values())
            if max_score > 0:
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                
        # Crisis detection
        crisis_indicators = ["emergency", "crisis", "urgent", "immediate help", "desperate"]
        is_crisis = any(indicator in query_lower for indicator in crisis_indicators)
        
        return {
            "primary_emotion": primary_emotion,
            "emotion_scores": emotion_scores,
            "is_crisis": is_crisis,
            "emotional_intensity": max(emotion_scores.values()) if emotion_scores else 0
        }
        
    async def _determine_routing_strategy(self, domains: List[str], complexity: Dict[str, Any], intent: str) -> str:
        """Determine the best routing strategy for the query"""
        
        if len(domains) == 0:
            return "general_intelligence"
        elif len(domains) == 1:
            return "single_domain"
        elif len(domains) <= 2:
            return "dual_domain_fusion"
        elif complexity["level"] == "complex":
            return "hierarchical_routing"
        else:
            return "parallel_multi_domain"
            
    def _generate_cache_key(self, query: str, context: str) -> str:
        """Generate cache key for query analysis"""
        
        combined_text = f"{context} {query}".strip()
        
        # Simple hash-like key (in production, use proper hashing)
        import hashlib
        return hashlib.md5(combined_text.encode()).hexdigest()[:16]
        
    async def _route_multi_domain_query(self, data: Dict[str, Any]):
        """Route multi-domain query to appropriate agents"""
        
        analysis = data.get("analysis", {})
        query = analysis.get("query", "")
        routing_strategy = analysis.get("routing_strategy", "single_domain")
        
        print(f"🌐 Routing multi-domain query using {routing_strategy} strategy")
        
        # Create routing decision record
        routing_id = f"route_{int(datetime.now().timestamp())}"
        routing_decision = {
            "routing_id": routing_id,
            "query": query,
            "analysis": analysis,
            "routing_strategy": routing_strategy,
            "timestamp": datetime.now().isoformat(),
            "status": "routing"
        }
        
        self.routing_decisions[routing_id] = routing_decision
        
        # Execute routing based on strategy
        if routing_strategy == "single_domain":
            await self._execute_single_domain_routing(routing_id, analysis)
        elif routing_strategy == "dual_domain_fusion":
            await self._execute_dual_domain_routing(routing_id, analysis)
        elif routing_strategy == "parallel_multi_domain":
            await self._execute_parallel_multi_domain_routing(routing_id, analysis)
        elif routing_strategy == "hierarchical_routing":
            await self._execute_hierarchical_routing(routing_id, analysis)
        else:
            await self._execute_general_routing(routing_id, analysis)
            
    async def _execute_single_domain_routing(self, routing_id: str, analysis: Dict[str, Any]):
        """Execute single domain routing"""
        
        primary_domain = analysis["primary_domains"][0] if analysis["primary_domains"] else "general"
        
        print(f"📍 Single domain routing to: {primary_domain}")
        
        # Send query to primary domain agent
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "process_domain_query",
                "routing_id": routing_id,
                "target_domain": primary_domain,
                "query": analysis["query"],
                "analysis": analysis,
                "routing_strategy": "single_domain"
            }
        )
        
        # Update routing decision
        self.routing_decisions[routing_id]["routed_domains"] = [primary_domain]
        self.routing_decisions[routing_id]["status"] = "routed_single"
        
    async def _execute_dual_domain_routing(self, routing_id: str, analysis: Dict[str, Any]):
        """Execute dual domain routing with fusion"""
        
        primary_domains = analysis["primary_domains"][:2]
        
        print(f"🔄 Dual domain routing to: {primary_domains}")
        
        # Send query to both domains
        for domain in primary_domains:
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.COORDINATION_REQUEST,
                {
                    "action": "process_domain_query",
                    "routing_id": routing_id,
                    "target_domain": domain,
                    "query": analysis["query"],
                    "analysis": analysis,
                    "routing_strategy": "dual_domain_fusion",
                    "fusion_weight": 0.5  # Equal weight for dual domains
                }
            )
            
        # Update routing decision
        self.routing_decisions[routing_id]["routed_domains"] = primary_domains
        self.routing_decisions[routing_id]["status"] = "routed_dual"
        
    async def _execute_parallel_multi_domain_routing(self, routing_id: str, analysis: Dict[str, Any]):
        """Execute parallel multi-domain routing"""
        
        primary_domains = analysis["primary_domains"]
        domain_weights = await self._calculate_domain_weights(analysis)
        
        print(f"⚡ Parallel multi-domain routing to: {primary_domains}")
        
        # Send query to all primary domains
        for domain in primary_domains:
            weight = domain_weights.get(domain, 1.0 / len(primary_domains))
            
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.COORDINATION_REQUEST,
                {
                    "action": "process_domain_query",
                    "routing_id": routing_id,
                    "target_domain": domain,
                    "query": analysis["query"],
                    "analysis": analysis,
                    "routing_strategy": "parallel_multi_domain",
                    "fusion_weight": weight
                }
            )
            
        # Update routing decision
        self.routing_decisions[routing_id]["routed_domains"] = primary_domains
        self.routing_decisions[routing_id]["domain_weights"] = domain_weights
        self.routing_decisions[routing_id]["status"] = "routed_parallel"
        
    async def _calculate_domain_weights(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for domain fusion based on confidence scores"""
        
        primary_domains = analysis["primary_domains"]
        confidence_scores = analysis["confidence_scores"]
        
        # Calculate weights based on confidence scores
        total_confidence = sum(confidence_scores.get(domain, 0) for domain in primary_domains)
        
        if total_confidence == 0:
            # Equal weights if no confidence information
            return {domain: 1.0 / len(primary_domains) for domain in primary_domains}
        
        weights = {}
        for domain in primary_domains:
            weights[domain] = confidence_scores.get(domain, 0) / total_confidence
            
        return weights
        
    async def _synthesize_domain_responses(self, data: Dict[str, Any]):
        """Synthesize responses from multiple domains"""
        
        routing_id = data.get("routing_id")
        domain_responses = data.get("domain_responses", {})
        analysis = data.get("analysis", {})
        
        print(f"🔮 Synthesizing responses from {len(domain_responses)} domains")
        
        # Get fusion strategy
        fusion_method = self.multi_domain_config["response_synthesis_method"]
        fusion_function = self.fusion_strategies.get(fusion_method, self._weighted_response_fusion)
        
        # Synthesize responses
        synthesized_response = await fusion_function(domain_responses, analysis)
        
        # Update routing decision with synthesis results
        if routing_id in self.routing_decisions:
            self.routing_decisions[routing_id].update({
                "synthesis_complete": True,
                "domain_responses_count": len(domain_responses),
                "synthesized_response": synthesized_response,
                "synthesis_method": fusion_method,
                "completion_timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
        print(f"✅ Response synthesis completed using {fusion_method}")
        
        # Send synthesized response
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "cross_domain_response_ready",
                "routing_id": routing_id,
                "synthesized_response": synthesized_response,
                "domain_count": len(domain_responses),
                "synthesis_method": fusion_method
            }
        )
        
    async def _weighted_response_fusion(self, domain_responses: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Fuse responses using weighted combination"""
        
        confidence_scores = analysis.get("confidence_scores", {})
        
        # Sort domains by confidence
        sorted_domains = sorted(domain_responses.keys(), 
                              key=lambda d: confidence_scores.get(d, 0), 
                              reverse=True)
        
        # Create weighted fusion
        fused_response = "Based on comprehensive analysis across multiple domains:\n\n"
        
        for i, domain in enumerate(sorted_domains):
            response = domain_responses[domain].get("response", "")
            confidence = confidence_scores.get(domain, 0)
            
            if confidence > self.multi_domain_config['fusion_weight_threshold']:
                weight_indicator = "🔥" if confidence > 0.8 else "✨" if confidence > 0.6 else "💡"
                
                fused_response += f"{weight_indicator} **{domain.title()} Perspective:**\n"
                fused_response += f"{response}\n\n"
                
        fused_response += "This response combines insights from multiple specialized domains to provide comprehensive guidance."
        
        return fused_response
        
    async def _analyze_routing_performance(self):
        """Analyze performance of routing decisions"""
        
        if not self.routing_decisions:
            return
            
        # Analyze recent routing decisions (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_decisions = [
            decision for decision in self.routing_decisions.values()
            if datetime.fromisoformat(decision["timestamp"]) > recent_cutoff
        ]
        
        if not recent_decisions:
            return
            
        # Calculate performance metrics
        completed_decisions = [d for d in recent_decisions if d["status"] == "completed"]
        success_rate = len(completed_decisions) / len(recent_decisions) if recent_decisions else 0
        
        # Update metrics
        self.intelligence_metrics["routing_efficiency"] = success_rate
        
        # Analyze multi-domain success
        multi_domain_decisions = [d for d in completed_decisions if len(d.get("routed_domains", [])) > 1]
        multi_domain_success_rate = len(multi_domain_decisions) / max(1, len([d for d in recent_decisions if len(d.get("routed_domains", [])) > 1]))
        
        self.intelligence_metrics["multi_domain_success_rate"] = multi_domain_success_rate
        
    async def _save_intelligence_data(self):
        """Save intelligence data to storage"""
        
        try:
            # Save domain intelligence
            intelligence_data = {
                "intelligence": self.domain_intelligence,
                "metrics": self.intelligence_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.intelligence_storage_path, 'w') as f:
                json.dump(intelligence_data, f, indent=2)
                
            # Save routing history (keep last 1000 decisions)
            recent_decisions = dict(list(self.routing_decisions.items())[-1000:])
            with open(self.routing_history_path, 'w') as f:
                json.dump(recent_decisions, f, indent=2)
                
            # Save query cache (keep last 500 entries)
            recent_cache = dict(list(self.query_analysis_cache.items())[-500:])
            with open(self.query_cache_path, 'w') as f:
                json.dump(recent_cache, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save intelligence data: {e}")

# Global instance
cross_domain_agent = CrossDomainAgent() 
