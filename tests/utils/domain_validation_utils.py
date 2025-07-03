 #!/usr/bin/env python3
"""
MeeTARA Lab - Domain Validation Utilities
Reusable utilities for dynamic domain testing across all test suites
"""

import yaml
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

class DomainConfigManager:
    """Manager for dynamic domain configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path"""
        self.config_path = Path(config_path) if config_path else Path("config/cloud-optimized-domain-mapping.yaml")
        self._config = None
        self._domains = None
        self._categories = None
    
    @property
    def config(self) -> Dict:
        """Load and cache domain configuration"""
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Domain configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        
        return self._config
    
    @property
    def domains(self) -> Set[str]:
        """Get all configured domains dynamically"""
        if self._domains is None:
            self._domains = set()
            for category_data in self.config.values():
                if isinstance(category_data, dict):
                    self._domains.update(category_data.keys())
        
        return self._domains
    
    @property
    def categories(self) -> Set[str]:
        """Get all domain categories"""
        if self._categories is None:
            self._categories = set(self.config.keys())
        
        return self._categories
    
    @property
    def domain_count(self) -> int:
        """Get total domain count"""
        return len(self.domains)
    
    def get_category_domains(self, category: str) -> Set[str]:
        """Get domains for a specific category"""
        if category not in self.config:
            return set()
        
        category_data = self.config[category]
        if isinstance(category_data, dict):
            return set(category_data.keys())
        
        return set()
    
    def get_domain_category(self, domain: str) -> Optional[str]:
        """Get the category for a specific domain"""
        for category, category_data in self.config.items():
            if isinstance(category_data, dict) and domain in category_data:
                return category
        
        return None
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get domain count per category"""
        return {
            category: len(self.get_category_domains(category))
            for category in self.categories
        }
    
    def get_sample_domains(self, max_per_category: int = 1) -> Dict[str, List[str]]:
        """Get sample domains from each category for testing"""
        samples = {}
        
        for category in self.categories:
            category_domains = list(self.get_category_domains(category))
            sample_count = min(max_per_category, len(category_domains))
            samples[category] = category_domains[:sample_count]
        
        return samples

class DomainTestValidators:
    """Collection of validation functions for domain testing"""
    
    @staticmethod
    def validate_domain_completeness(agent_domains: Set[str], expected_domains: Set[str], agent_name: str) -> None:
        """Validate that agent supports all expected domains"""
        assert len(agent_domains) == len(expected_domains), \
            f"{agent_name}: Expected {len(expected_domains)} domains, found {len(agent_domains)}"
        
        assert agent_domains == expected_domains, \
            f"{agent_name}: Domain set mismatch. Missing: {expected_domains - agent_domains}, Extra: {agent_domains - expected_domains}"
    
    @staticmethod
    def validate_domain_categories(agent_categories: Dict[str, str], expected_domains: Set[str], agent_name: str) -> None:
        """Validate that all domains have category mappings"""
        for domain in expected_domains:
            assert domain in agent_categories, \
                f"{agent_name}: Domain '{domain}' not in category mapping"
            
            assert agent_categories[domain] is not None, \
                f"{agent_name}: Domain '{domain}' has null category"
    
    @staticmethod
    def validate_quality_requirements(agent, domains: Set[str], agent_name: str) -> None:
        """Validate quality requirements for all domains"""
        for domain in domains:
            if hasattr(agent, '_get_quality_requirements'):
                requirements = agent._get_quality_requirements(domain)
                assert requirements is not None, \
                    f"{agent_name}: No quality requirements for domain '{domain}'"
                
                if 'min_validation_score' in requirements:
                    score = requirements['min_validation_score']
                    assert 70 <= score <= 100, \
                        f"{agent_name}: Invalid validation score {score}% for domain '{domain}'"
    
    @staticmethod
    def validate_domain_keywords(agent, domains: Set[str], agent_name: str) -> None:
        """Validate domain keyword mappings"""
        if not hasattr(agent, 'domain_keywords'):
            return  # Skip if agent doesn't use keywords
        
        for domain in domains:
            if domain in agent.domain_keywords:
                keywords = agent.domain_keywords[domain]
                assert len(keywords) > 0, \
                    f"{agent_name}: Empty keywords for domain '{domain}'"
                
                assert all(isinstance(kw, str) for kw in keywords), \
                    f"{agent_name}: Non-string keywords found for domain '{domain}'"
    
    @staticmethod
    def validate_compatibility_scores(agent, sample_domains: List[str], agent_name: str) -> None:
        """Validate domain compatibility matrix"""
        if not hasattr(agent, '_calculate_domain_compatibility'):
            return  # Skip if agent doesn't use compatibility
        
        import asyncio
        
        async def check_compatibility():
            for domain1 in sample_domains:
                for domain2 in sample_domains:
                    compatibility = await agent._calculate_domain_compatibility(domain1, domain2)
                    assert 0.0 <= compatibility <= 1.0, \
                        f"{agent_name}: Invalid compatibility score {compatibility} for {domain1}-{domain2}"
        
        # Run async validation
        asyncio.run(check_compatibility())

def get_domain_config() -> DomainConfigManager:
    """Get domain configuration manager instance"""
    return DomainConfigManager()

def validate_agent_domain_support(agent, agent_name: str, config_manager: DomainConfigManager) -> None:
    """Comprehensive validation of agent domain support"""
    # Load agent configuration
    if hasattr(agent, '_load_domain_configuration'):
        agent._load_domain_configuration()
    
    # Get agent domains
    agent_domains = set()
    if hasattr(agent, 'domain_mapping'):
        for category, domains in agent.domain_mapping.items():
            agent_domains.update(domains)
    
    # Validate completeness
    DomainTestValidators.validate_domain_completeness(
        agent_domains, config_manager.domains, agent_name
    )
    
    # Validate categories
    if hasattr(agent, 'domain_categories'):
        DomainTestValidators.validate_domain_categories(
            agent.domain_categories, config_manager.domains, agent_name
        )
    
    # Validate quality requirements
    DomainTestValidators.validate_quality_requirements(
        agent, config_manager.domains, agent_name
    )
    
    # Validate keywords
    if hasattr(agent, '_initialize_domain_keywords'):
        agent._initialize_domain_keywords()
    
    DomainTestValidators.validate_domain_keywords(
        agent, config_manager.domains, agent_name
    )
    
    print(f"✅ {agent_name} validated for {config_manager.domain_count} domains")

# Convenience functions for common test patterns
def get_test_fixtures():
    """Get common test fixtures for domain testing"""
    config_manager = get_domain_config()
    
    return {
        'domain_config': config_manager.config,
        'expected_domains': config_manager.domains,
        'expected_categories': config_manager.categories,
        'domain_count': config_manager.domain_count,
        'category_counts': config_manager.get_category_counts(),
        'sample_domains': config_manager.get_sample_domains()
    }

def run_basic_domain_validation(agent, agent_name: str) -> bool:
    """Run basic domain validation for an agent"""
    try:
        config_manager = get_domain_config()
        validate_agent_domain_support(agent, agent_name, config_manager)
        return True
    except Exception as e:
        print(f"❌ {agent_name} validation failed: {e}")
        return False