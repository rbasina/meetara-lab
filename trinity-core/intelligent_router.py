"""
MeeTARA Lab - Intelligent Router
Enhanced routing with RoBERTa-powered multi-domain analysis
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Use centralized domain mapping
from .domain_integration import domain_integration, get_domain_categories, get_all_domains, validate_domain, get_model_for_domain

class IntelligentRouter:
    """Enhanced intelligent routing with RoBERTa-powered multi-domain analysis"""
    
    def __init__(self):
        # Load domain categories dynamically from config
        self.domain_categories = get_domain_categories()
        
        # Routing intelligence
        self.routing_history = []
        self.performance_metrics = {
            "total_routes": 0,
            "successful_routes": 0,
            "multi_domain_routes": 0,
            "average_confidence": 0.0
        }
        
        # RoBERTa-powered routing enhancement
        self.roberta_routing = {
            "model_name": "roberta-base",
            "confidence_threshold": 0.7,
            "multi_domain_threshold": 0.5,
            "context_window": 512
        }
        
        # MeeTARA integration (ports 2025/8765/8766)
        self.meetara_integration = {
            "tts_port": 2025,
            "chat_port": 8765, 
            "api_port": 8766,
            "connection_timeout": 30
        }
        
        # Universal model coordination
        self.universal_coordination = {
            "model_switching_enabled": True,
            "seamless_handoff": True,
            "context_preservation": True,
            "quality_assurance": True
        }
        
        print(f"✅ Intelligent Router initialized")
        print(f"   → Total domains: {len(get_all_domains())}")
        print(f"   → Categories: {len(self.domain_categories)}")
        print(f"   → Config-driven: Dynamic domain loading enabled")
    
    def analyze_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """Analyze query and determine optimal routing"""
        
        # Domain detection using config-driven approach
        domain_scores = self._calculate_domain_scores(query + " " + context)
        
        # Select best domains
        primary_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[primary_domain]
        
        # Multi-domain detection
        multi_domains = [
            domain for domain, score in domain_scores.items() 
            if score > self.roberta_routing["multi_domain_threshold"]
        ]
        
        # Routing decision
        routing_strategy = self._determine_routing_strategy(
            primary_domain, multi_domains, confidence
        )
        
        # Get model recommendation from config
        recommended_model = get_model_for_domain(primary_domain)
        
        analysis = {
            "primary_domain": primary_domain,
            "confidence": confidence,
            "multi_domains": multi_domains,
            "routing_strategy": routing_strategy,
            "recommended_model": recommended_model,
            "domain_scores": domain_scores,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update performance metrics
        self._update_performance_metrics(analysis)
        
        return analysis
    
    def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate domain relevance scores using dynamic domain mapping"""
        scores = {}
        text_lower = text.lower()
        
        # Get all domains dynamically
        all_domains = get_all_domains()
        
        for domain in all_domains:
            score = 0.0
            
            # Domain-specific keyword matching (simplified approach)
            # In production, this would use RoBERTa embeddings
            domain_keywords = self._get_domain_keywords(domain)
            
            for keyword in domain_keywords:
                if keyword in text_lower:
                    score += 0.1
                    
            # Normalize score
            scores[domain] = min(score, 1.0)
        
        return scores
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for domain detection (simplified approach)"""
        # This is a simplified version - in production would use embeddings
        keyword_mapping = {
            # Healthcare keywords
            "general_health": ["health", "medical", "doctor", "medicine"],
            "mental_health": ["mental", "therapy", "counseling", "anxiety"],
            "nutrition": ["nutrition", "diet", "food", "eating"],
            "fitness": ["fitness", "exercise", "workout", "gym"],
            
            # Business keywords  
            "entrepreneurship": ["business", "startup", "entrepreneur", "venture"],
            "marketing": ["marketing", "advertising", "promotion", "brand"],
            "sales": ["sales", "selling", "customer", "revenue"],
            
            # Education keywords
            "academic_tutoring": ["study", "learn", "education", "academic"],
            "skill_development": ["skill", "development", "training", "improvement"],
            
            # Creative keywords
            "writing": ["writing", "write", "author", "content"],
            "storytelling": ["story", "narrative", "plot", "character"],
            
            # Technology keywords
            "programming": ["programming", "code", "software", "development"],
            "ai_ml": ["ai", "machine learning", "artificial intelligence", "ml"],
            
            # Default fallback
            "default": ["help", "question", "ask", "need"]
        }
        
        return keyword_mapping.get(domain, keyword_mapping["default"])
    
    def _determine_routing_strategy(self, primary_domain: str, multi_domains: List[str], confidence: float) -> str:
        """Determine optimal routing strategy"""
        
        if confidence < 0.3:
            return "clarification_needed"
        elif len(multi_domains) <= 1:
            return "single_domain"
        elif len(multi_domains) <= 3:
            return "multi_domain_parallel"
        else:
            return "multi_domain_sequential"
    
    def _update_performance_metrics(self, analysis: Dict[str, Any]):
        """Update routing performance metrics"""
        self.performance_metrics["total_routes"] += 1
        
        if analysis["confidence"] > 0.5:
            self.performance_metrics["successful_routes"] += 1
            
        if len(analysis["multi_domains"]) > 1:
            self.performance_metrics["multi_domain_routes"] += 1
            
        # Update average confidence
        total = self.performance_metrics["total_routes"]
        current_avg = self.performance_metrics["average_confidence"]
        new_confidence = analysis["confidence"]
        
        self.performance_metrics["average_confidence"] = (
            (current_avg * (total - 1) + new_confidence) / total
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        return {
            "performance_metrics": self.performance_metrics,
            "domain_stats": {
                "total_domains": len(get_all_domains()),
                "total_categories": len(self.domain_categories),
                "domains_per_category": {
                    category: len(domains) 
                    for category, domains in self.domain_categories.items()
                }
            },
            "config_integration": {
                "config_driven": True,
                "dynamic_loading": True,
                "centralized_mapping": True
            },
            "enhanced_features": {
                "roberta_routing": self.roberta_routing,
                "meetara_integration": self.meetara_integration,
                "universal_coordination": self.universal_coordination
            }
        }
    
    def refresh_domain_mapping(self):
        """Refresh domain mapping from config"""
        domain_integration.refresh_config()
        self.domain_categories = get_domain_categories()
        print(f"✅ Domain mapping refreshed - {len(get_all_domains())} domains loaded")

# Global instance
intelligent_router = IntelligentRouter() 
