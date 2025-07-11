"""
MeeTARA Lab - Domain Experts with Trinity Architecture
Specialized knowledge for 60+ domains with expert system integration and Trinity intelligence
"""

import asyncio
import json
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import trinity_core components
import sys
sys.path.append('../trinity_core')
from trinity_core.agents.coordination.lightweight_mcp_v2 import BaseAgent, AgentType, MessageType, MCPMessage
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

class ExpertiseLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    SPECIALIST = "specialist"

class DomainCategory(Enum):
    HEALTHCARE = "healthcare"
    DAILY_LIFE = "daily_life" 
    BUSINESS = "business"
    EDUCATION = "education"
    CREATIVE = "creative"
    TECHNOLOGY = "technology"
    SPECIALIZED = "specialized"

@dataclass
class DomainExpertise:
    domain_name: str
    category: DomainCategory
    expertise_level: ExpertiseLevel
    knowledge_base: Dict[str, Any]
    model_recommendations: Dict[str, str]
    optimization_strategies: Dict[str, Any]
    quality_thresholds: Dict[str, float]

class DomainExperts:
    """
    Represents the domain expert intelligence system.
    Refactored to use the modern SmartTrinityConfigManager.
    """
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        self.all_domains = self.config_manager.get_all_domains_flat()
        self.domains_by_category = self._get_domains_by_category()

    def _get_domains_by_category(self):
        """Gets a dictionary of available domains grouped by category."""
        domains_by_category = {}
        for domain, details in self.all_domains.items():
            category = details.get('category', 'unknown')
            if category not in domains_by_category:
                domains_by_category[category] = []
            domains_by_category[category].append(domain)
        return domains_by_category

    def get_expert_details(self, domain: str) -> Dict[str, Any]:
        """
        Retrieves the expert details (parameters) for a given domain.
        """
        if domain not in self.all_domains:
            return {"error": f"Domain '{domain}' not found."}
        
        return self.config_manager.get_tara_proven_params(domain)

    def list_experts_by_category(self, category: str) -> List[str]:
        """
        Lists all expert domains within a specific category.
        """
        return self.domains_by_category.get(category, [])

if __name__ == '__main__':
    # Example usage:
    expert_system = DomainExperts()
    
    print("--- Listing experts for 'daily_life' ---")
    daily_experts = expert_system.list_experts_by_category('daily_life')
    print(daily_experts)
    
    print("\n--- Getting details for 'shopping' expert ---")
    shopping_details = expert_system.get_expert_details('shopping')
    print(shopping_details) 
