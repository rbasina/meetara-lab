"""
MeeTARA Lab - Comprehensive Domain Integration
This module serves as a clean interface to the centralized SmartTrinityConfigManager,
ensuring all parts of the system access domain information consistently.
"""

from typing import Dict, Any, List, Optional
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Initialize a single instance of the config manager to be used by all functions
_manager = SmartTrinityConfigManager()

def get_domain_categories() -> List[str]:
    """
    Gets a list of all unique domain categories.
    e.g., ['healthcare', 'business', 'daily_life', ...]
    """
    if _manager._domain_config:
        return sorted(list(_manager._domain_config.keys()))
    return []

def get_all_domains() -> List[str]:
    """
    Gets a flat list of all available domain names.
    e.g., ['health_advisor', 'shopping', 'story_writer', ...]
    """
    # get_all_domains_flat() already returns a list of keys.
    return sorted(_manager.get_all_domains_flat())

def get_domains_for_category(category: str) -> List[str]:
    """
    Gets all domain names for a specific category.
    """
    if _manager._domain_config and category in _manager._domain_config:
        category_data = _manager._domain_config[category].get('domains', {})
        if category_data:
            return sorted(list(category_data.keys()))
    return []

def validate_domain(domain: str) -> bool:
    """
    Checks if a domain exists in the configuration.
    """
    return domain in _manager.get_all_domains_flat()

def get_domain_stats() -> Dict[str, int]:
    """
    Returns statistics about the configured domains.
    """
    total_categories = 0
    if _manager._domain_config:
        total_categories = len(_manager._domain_config)
    return {
        "total_domains": len(_manager.get_all_domains_flat()),
        "total_categories": total_categories
    } 
