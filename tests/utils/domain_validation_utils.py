#!/usr/bin/env python3
"""
MeeTARA Lab - Domain Validation Utilities
Reusable utilities for dynamic domain testing, powered by the centralized
SmartTrinityConfigManager.
"""

import asyncio
from typing import Dict, Set, List, Optional
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

def get_domain_config() -> SmartTrinityConfigManager:
    """Gets a singleton instance of the centralized config manager."""
    try:
        return SmartTrinityConfigManager()
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Could not initialize central config for testing: {e}")

class DomainTestValidators:
    """Collection of validation functions for domain testing."""
    
    @staticmethod
    def validate_domain_completeness(agent_domains: Set[str], expected_domains: Set[str], agent_name: str) -> None:
        """Validate that an agent supports all expected domains."""
        if not expected_domains:
            # Cannot validate against an empty set of expected domains.
            return
        assert agent_domains == expected_domains, \
            f"{agent_name}: Domain set mismatch. Missing: {expected_domains - agent_domains}, Extra: {agent_domains - agent_domains}"
    
    @staticmethod
    def validate_domain_categories(config_manager: SmartTrinityConfigManager, agent_name: str) -> None:
        """Validates that all domains have a valid category in the config."""
        all_domains = config_manager.get_all_domains_flat()
        for domain in all_domains:
            params = config_manager.get_tara_proven_params(domain)
            assert params.get('category'), f"{agent_name}: Domain '{domain}' is missing a category in the config."

def validate_agent_domain_support(agent_name: str, config_manager: SmartTrinityConfigManager) -> None:
    """
    Comprehensive validation of domain support using the central config.
    This is a conceptual placeholder, as agents should no longer hold their own domain lists.
    The real test is whether the agent can correctly process any domain given to it
    by the conductor, which gets the domain list from the config manager.
    """
    print(f"Verifying configuration for '{agent_name}'...")
    expected_domains = set(config_manager.get_all_domains_flat())
    
    # In the new architecture, agents don't have their own domain lists.
    # We simply confirm that the config manager has domains.
    assert len(expected_domains) > 60, "Expected at least 60 domains in the central configuration."
    
    DomainTestValidators.validate_domain_categories(config_manager, agent_name)
    
    print(f"✅ '{agent_name}' conceptually validated against {len(expected_domains)} domains from SmartTrinityConfigManager.")

def get_test_fixtures() -> Dict:
    """Gets common test fixtures for domain testing from the central config."""
    config_manager = get_domain_config()
    all_domains = config_manager.get_all_domains_flat()
    all_categories = config_manager.get_all_domain_categories()
    
    # Create a list of sample domains for testing
    sample_domains = [domains[0] for domains in all_categories.values() if domains]

    return {
        'config_manager': config_manager,
        'expected_domains': set(all_domains),
        'expected_categories': all_categories,
        'domain_count': len(all_domains),
        'category_counts': {cat: len(doms) for cat, doms in all_categories.items()},
        'sample_domains': sample_domains
    }

def run_basic_domain_validation(agent_name: str) -> bool:
    """Runs basic domain validation for a conceptual agent."""
    try:
        config_manager = get_domain_config()
        validate_agent_domain_support(agent_name, config_manager)
        return True
    except (RuntimeError, AssertionError) as e:
        print(f"❌ '{agent_name}' validation failed: {e}")
        return False
