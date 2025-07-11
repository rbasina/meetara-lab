import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Assuming TrinityIntelligenceHub will be imported from its own module
# For now, we'll use a placeholder or assume it's available in the same context for type hinting
# from trinity_core.agents.intelligence_hub import TrinityIntelligenceHub # Uncomment in actual usage

logger = logging.getLogger(__name__)

class TrinityDomainRouter:
    """Domain Router component of Trinity Intelligence Hub"""
    
    def __init__(self, hub: Any):
        self.hub = hub
        self.routing_history = []
    
    async def route_intelligently(self, request: Dict[str, Any], 
                                intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Route requests intelligently using Trinity optimization"""
        
        # Check routing cache first
        input_hash = hash(str(request.get("user_input", "")))
        cache_key = f"routing_{input_hash}"
        
        if cache_key in self.hub.shared_context["routing_cache"]:
            self.hub.trinity_stats["cache_hits"] += 1
            cached_routing = self.hub.shared_context["routing_cache"][cache_key]
            return {
                "success": True,
                "routing_source": "cache",
                "primary_domain": cached_routing.get("primary_domain"),
                "confidence": cached_routing.get("confidence"),
                "alternative_domains": cached_routing.get("alternative_domains", []),
                "routing_pattern": cached_routing,
                "decisions": cached_routing.get("decisions", [])
            }
        
        # Perform intelligent routing
        primary_domain = intelligence.get("domain_analysis", {}).get("primary_domain", "general")
        confidence = intelligence.get("domain_analysis", {}).get("confidence", 0.5)
        
        # Identify alternative domains
        alternative_domains = await self._identify_alternative_domains(intelligence)
        
        # Generate routing decisions
        routing_decisions = await self._generate_routing_decisions(
            primary_domain, alternative_domains, intelligence
        )
        
        routing_pattern = {
            "input_hash": input_hash,
            "primary_domain": primary_domain,
            "confidence": confidence,
            "alternative_domains": alternative_domains,
            "routing_logic": await self._explain_routing_logic(intelligence),
            "decisions": routing_decisions,
            "timestamp": datetime.now().isoformat()
        }
        
        result = {
            "success": True,
            "routing_source": "intelligent",
            "primary_domain": primary_domain,
            "confidence": confidence,
            "alternative_domains": alternative_domains,
            "routing_pattern": routing_pattern,
            "decisions": routing_decisions,
            "alternative_paths": await self._suggest_alternative_paths(alternative_domains)
        }
        
        # Cache for future use
        self.hub.shared_context["routing_cache"][cache_key] = routing_pattern
        
        return result
    
    async def _identify_alternative_domains(self, intelligence: Dict[str, Any]) -> List[str]:
        """Identify alternative domains for routing"""
        
        alternative_domains = []
        
        # Check predictions for alternative domains
        predictions = intelligence.get("predictions", {})
        if predictions.get("potential_concerns"):
            for concern in predictions["potential_concerns"]:
                if "health" in concern.lower():
                    alternative_domains.append("health")
                elif "business" in concern.lower():
                    alternative_domains.append("business")
                elif "education" in concern.lower():
                    alternative_domains.append("education")
        
        # Check human needs for alternative domains
        human_needs = intelligence.get("human_needs", {})
        if human_needs.get("contextual_needs"):
            for need in human_needs["contextual_needs"]:
                if "technical" in need.lower():
                    alternative_domains.append("technology")
                elif "emotional" in need.lower():
                    alternative_domains.append("psychology")
        
        return list(set(alternative_domains))[:3]  # Limit to 3 alternatives
    
    async def _generate_routing_decisions(self, primary_domain: str, 
                                        alternative_domains: List[str],
                                        intelligence: Dict[str, Any]) -> List[str]:
        """Generate routing decisions based on intelligence"""
        
        decisions = []
        
        # Primary domain decision
        confidence = intelligence.get("domain_analysis", {}).get("confidence", 0.5)
        if confidence > 0.8:
            decisions.append(f"High confidence routing to {primary_domain}")
        elif confidence > 0.6:
            decisions.append(f"Medium confidence routing to {primary_domain}")
        else:
            decisions.append(f"Low confidence routing to {primary_domain} - consider alternatives")
        
        # Alternative domain decisions
        for alt_domain in alternative_domains:
            decisions.append(f"Consider {alt_domain} as alternative approach")
        
        # Urgency-based decisions
        urgency = intelligence.get("domain_analysis", {}).get("urgency_level", "normal")
        if urgency == "high":
            decisions.append("Prioritize immediate response due to high urgency")
        
        return decisions
    
    async def _explain_routing_logic(self, intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the routing logic for transparency"""
        
        return {
            "primary_factors": [
                "Domain analysis confidence",
                "User intent classification",
                "Urgency level assessment"
            ],
            "intelligence_factors": [
                "Emotional state analysis",
                "Complexity assessment",
                "Predictive insights"
            ],
            "optimization_factors": [
                "Cache utilization",
                "Context sharing",
                "Performance optimization"
            ]
        }
    
    async def _suggest_alternative_paths(self, alternative_domains: List[str]) -> List[str]:
        """Suggest alternative paths for routing"""
        
        paths = []
        
        for domain in alternative_domains:
            paths.append(f"Route to {domain} for specialized handling")
            paths.append(f"Use {domain} context for enhanced response")
        
        return paths