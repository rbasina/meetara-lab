"""
MeeTARA Lab - Comprehensive Domain Integration
Aligns MCP Agent Ecosystem with all 62 domains and 10 Enhanced TARA Features
SINGLE SOURCE OF TRUTH for all domain mappings - Config-driven approach
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from functools import lru_cache

class DomainIntegration:
    """Comprehensive integration of all 62 domains with 10 Enhanced TARA Features"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/trinity_domain_model_mapping_config.yaml"
        
        # Load all 62 domains from cloud-optimized mapping
        self.domain_mapping = self._load_domain_mapping()
        
        # Cache for performance
        self._domain_categories = None
        self._all_domains_flat = None
        
        # 10 Enhanced TARA Features - MUST BE PRESERVED
        self.enhanced_features = {
            "tts_manager": {
                "description": "6 voice categories with Edge-TTS + pyttsx3 integration",
                "file": "trinity-core/tts_manager.py",
                "domains": "ALL",
                "voice_categories": ["meditative", "therapeutic", "professional", "educational", "creative", "casual"]
            },
            "emotion_detector": {
                "description": "RoBERTa-based emotion detection with professional context",
                "file": "trinity-core/emotion_detector.py", 
                "domains": "ALL",
                "emotion_categories": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
            },
            "intelligent_router": {
                "description": "Multi-domain analysis with RoBERTa-powered routing",
                "file": "trinity-core/intelligent_router.py",
                "domains": "ALL",
                "routing_methods": ["single_domain", "multi_domain", "cross_domain"]
            },
            "universal_gguf_factory": {
                "description": "Real GGUF creation with quality assurance",
                "file": "model-factory/trinity_master_gguf_factory.py",
                "domains": "ALL",
                "output_format": "Q4_K_M"
            },
            "training_orchestrator": {
                "description": "Multi-domain training coordination",
                "file": "cloud-training/training_orchestrator.py",
                "domains": "ALL",
                "training_modes": ["parallel", "sequential", "adaptive"]
            },
            "monitoring_recovery": {
                "description": "Connection recovery with dashboard tracking",
                "file": "cloud-training/monitoring_system.py",
                "domains": "ALL",
                "monitoring_types": ["training", "performance", "costs"]
            },
            "security_privacy": {
                "description": "Local processing with GDPR/HIPAA compliance",
                "file": "trinity-core/security_manager.py",
                "domains": "ALL",
                "compliance_standards": ["GDPR", "HIPAA", "SOC2"]
            },
            "domain_experts": {
                "description": "Specialized domain knowledge",
                "file": "intelligence-hub/domain_experts.py",
                "domains": "ALL",
                "expert_types": ["context_aware", "knowledge_base", "reasoning"]
            },
            "utilities_validation": {
                "description": "Data quality validation",
                "file": "trinity-core/validation_utils.py",
                "domains": "ALL",
                "validation_types": ["data_quality", "model_performance", "output_validation"]
            },
            "configuration_management": {
                "description": "Domain model mapping with training configurations",
                "file": "trinity-core/config_manager.py",
                "domains": "ALL",
                "config_types": ["domain_mapping", "training_params", "model_selection"]
            }
        }
        
    @lru_cache(maxsize=1)
    def _load_domain_mapping(self) -> Dict[str, Any]:
        """Load all 62 domains from trinity_domain_model_mapping_config.yaml - Config-driven only"""
        # Dynamic path resolution - works on any environment
        base_paths = [
            self.config_path,  # User-specified path
            Path(__file__).parent.parent / "config" / "trinity_domain_model_mapping_config.yaml",  # Relative to script
            Path.cwd() / "config" / "trinity_domain_model_mapping_config.yaml",  # Current working directory
            Path(__file__).resolve().parent.parent / "config" / "trinity_domain_model_mapping_config.yaml",  # Absolute relative
        ]
        
        # Add environment-specific paths
        try:
            # Check if we're in Google Colab
            import google.colab
            # Colab-specific paths
            base_paths.extend([
                Path("/content/meetara-lab/config/trinity_domain_model_mapping_config.yaml"),
                Path("/content/config/trinity_domain_model_mapping_config.yaml"),
                Path("/content/drive/MyDrive/meetara-lab/config/trinity_domain_model_mapping_config.yaml")
            ])
        except ImportError:
            # Not in Colab, add local development paths
            home = Path.home()
            base_paths.extend([
                home / "Documents" / "meetara-lab" / "config" / "trinity_domain_model_mapping_config.yaml",
                home / "Desktop" / "meetara-lab" / "config" / "trinity_domain_model_mapping_config.yaml",
            ])
        
        # Try each path until we find the config
        for path in base_paths:
            try:
                # Convert string paths to Path objects
                path_obj = Path(path) if isinstance(path, str) else path
                if path_obj and path_obj.exists():
                    with open(path_obj, 'r', encoding='utf-8') as f:
                        mapping = yaml.safe_load(f)
                        print(f"✅ Domain config loaded from: {path_obj}")
                        
                    # Extract all domains
                    all_domains = {}
                    domain_categories = ["healthcare", "daily_life", "business", "education", "creative", "technology", "specialized"]
                    
                    for category in domain_categories:
                        if category in mapping:
                            all_domains[category] = mapping[category]
                            
                    return {
                        "domains": all_domains,
                        "total_domain_count": sum(len(domains) for domains in all_domains.values()),
                        "config_loaded": True,
                        "config_path": str(path_obj)
                    }
                    
            except (FileNotFoundError, UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Could not load {path}: {e}")
                continue
                
        # No fallback - force user to fix config file
        raise FileNotFoundError(
            f"❌ CRITICAL: Could not load domain config from any path!\n"
            f"   Tried paths: {[str(p) for p in base_paths]}\n"
            f"   Please ensure config/trinity_domain_model_mapping_config.yaml exists and is accessible.\n"
            f"   This is a config-driven system - no hardcoded fallbacks!\n"
            f"   Current working directory: {Path.cwd()}\n"
            f"   Script location: {Path(__file__).parent}"
        )
    
    @property
    def domain_categories(self) -> Dict[str, List[str]]:
        """Get domain categories with lists of domains"""
        if self._domain_categories is None:
            self._domain_categories = {}
            for category, domains in self.domain_mapping["domains"].items():
                self._domain_categories[category] = list(domains.keys())
                
        return self._domain_categories
    
    @property
    def all_domains_flat(self) -> List[str]:
        """Get flat list of all domains"""
        if self._all_domains_flat is None:
            self._all_domains_flat = []
            for domains in self.domain_categories.values():
                self._all_domains_flat.extend(domains)
                
        return self._all_domains_flat
    
    def get_domain_category(self, domain: str) -> str:
        """Get category for a specific domain"""
        for category, domains in self.domain_categories.items():
            if domain in domains:
                return category
        return "daily_life"  # Default fallback
    
    def get_model_for_domain(self, domain: str) -> str:
        """Get model recommendation for a specific domain"""
        category = self.get_domain_category(domain)
        domain_mapping = self.domain_mapping["domains"].get(category, {})
        return domain_mapping.get(domain, "microsoft/Phi-3.5-mini-instruct")  # Default fallback
    
    def get_domains_by_category(self, category: str) -> List[str]:
        """Get all domains in a specific category"""
        return self.domain_categories.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.domain_categories.keys())
    
    def get_total_domain_count(self) -> int:
        """Get total number of domains"""
        return len(self.all_domains_flat)
    
    def validate_domain(self, domain: str) -> bool:
        """Check if domain exists in mapping"""
        return domain in self.all_domains_flat
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Get comprehensive domain statistics"""
        return {
            "total_domains": self.get_total_domain_count(),
            "total_categories": len(self.get_all_categories()),
            "domains_per_category": {
                category: len(domains) 
                for category, domains in self.domain_categories.items()
            },
            "categories": self.get_all_categories(),
            "enhanced_features": len(self.enhanced_features),
            "config_loaded": self.domain_mapping.get("config_loaded", False),
            "config_path": self.domain_mapping.get("config_path", "Unknown")
        }
    
    def get_tts_voice_for_domain(self, domain: str) -> str:
        """Get recommended TTS voice category for a domain"""
        category = self.get_domain_category(domain)
        
        # Domain-specific voice mapping
        voice_mapping = {
            "healthcare": "therapeutic",
            "daily_life": "casual", 
            "business": "professional",
            "education": "educational",
            "creative": "creative",
            "technology": "professional",
            "specialized": "professional"
        }
        
        return voice_mapping.get(category, "casual")
    
    def get_enhanced_feature_for_domain(self, domain: str, feature: str) -> Dict[str, Any]:
        """Get enhanced feature configuration for a specific domain"""
        if feature not in self.enhanced_features:
            return {}
            
        feature_config = self.enhanced_features[feature].copy()
        feature_config["domain"] = domain
        feature_config["category"] = self.get_domain_category(domain)
        feature_config["model"] = self.get_model_for_domain(domain)
        
        # Add domain-specific configurations
        if feature == "tts_manager":
            feature_config["recommended_voice"] = self.get_tts_voice_for_domain(domain)
            
        return feature_config
    
    def refresh_config(self):
        """Force reload of config file"""
        self._load_domain_mapping.cache_clear()
        self.domain_mapping = self._load_domain_mapping()
        self._domain_categories = None
        self._all_domains_flat = None
    
    def get_cross_domain_compatibility(self, domains: List[str]) -> Dict[str, Any]:
        """Analyze cross-domain compatibility for multi-domain queries"""
        if not domains:
            return {"compatible": False, "reason": "No domains provided"}
            
        # Validate all domains exist
        invalid_domains = [d for d in domains if not self.validate_domain(d)]
        if invalid_domains:
            return {"compatible": False, "reason": f"Invalid domains: {invalid_domains}"}
        
        # Get categories for all domains
        categories = [self.get_domain_category(d) for d in domains]
        unique_categories = list(set(categories))
        
        # Cross-domain analysis
        compatibility_score = 1.0
        if len(unique_categories) > 3:
            compatibility_score = 0.6  # Too many categories
        elif len(unique_categories) > 2:
            compatibility_score = 0.8  # Multiple categories
            
        return {
            "compatible": compatibility_score > 0.5,
            "compatibility_score": compatibility_score,
            "domains": domains,
            "categories": unique_categories,
            "recommended_approach": "parallel" if len(domains) <= 3 else "sequential"
        }

# Global instance - single source of truth
domain_integration = DomainIntegration()

# Convenience functions for easy importing (maintaining backward compatibility)
def get_domain_mapping() -> Dict[str, Dict[str, str]]:
    """Get complete domain mapping"""
    return domain_integration.domain_mapping["domains"]

def get_domain_categories() -> Dict[str, List[str]]:
    """Get domain categories with domain lists"""
    return domain_integration.domain_categories

def get_all_domains() -> List[str]:
    """Get flat list of all domains"""
    return domain_integration.all_domains_flat

def get_domain_category(domain: str) -> str:
    """Get category for a domain"""
    return domain_integration.get_domain_category(domain)

def get_model_for_domain(domain: str) -> str:
    """Get model for a domain"""
    return domain_integration.get_model_for_domain(domain)

def validate_domain(domain: str) -> bool:
    """Validate if domain exists"""
    return domain_integration.validate_domain(domain)

def get_domain_stats() -> Dict[str, Any]:
    """Get domain statistics"""
    return domain_integration.get_domain_stats()

def get_tts_voice_for_domain(domain: str) -> str:
    """Get TTS voice for domain"""
    return domain_integration.get_tts_voice_for_domain(domain)

def get_enhanced_feature_for_domain(domain: str, feature: str) -> Dict[str, Any]:
    """Get enhanced feature for domain"""
    return domain_integration.get_enhanced_feature_for_domain(domain, feature) 
