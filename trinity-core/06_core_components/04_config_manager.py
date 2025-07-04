"""
MeeTARA Lab - Configuration Manager with Trinity Architecture
Centralized configuration management for 62-domain training and Trinity systems
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import centralized domain mapping
from domain_integration import get_domain_categories, get_all_domains, get_domain_stats

@dataclass
class DomainConfig:
    """Configuration for a specific domain"""
    domain_name: str
    category: str
    model_tier: str
    model_name: str
    quality_thresholds: Dict[str, float]
    training_params: Dict[str, Any]
    optimization_strategy: Dict[str, Any]
    safety_critical: bool = False

@dataclass
class ModelTierConfig:
    """Configuration for model tiers"""
    name: str
    model_path: str
    cost_per_hour: float
    recommended_gpu: str
    batch_size: int
    sequence_length: int
    performance_tier: str

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    encryption_required: bool = True
    access_control_enabled: bool = True
    audit_logging: bool = True
    local_processing_only: bool = True
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 3
    data_retention_days: int = 30

@dataclass
class QualityConfig:
    """Quality assurance configuration"""
    target_validation_score: float = 101.0
    minimum_quality_threshold: float = 80.0
    data_filter_success_rate: float = 31.0
    samples_per_domain: int = 2000
    max_loss_threshold: float = 0.5
    convergence_minimum: float = 0.1

class TrinityConfigManager:
    """Trinity Architecture enhanced configuration manager"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration files
        self.domain_mapping_file = self.config_dir / "trinity_domain_model_mapping_config.yaml"
        self.model_tiers_file = self.config_dir / "model-tiers.json"
        self.security_config_file = self.config_dir / "security-config.json"
        self.quality_standards_file = self.config_dir / "quality-standards.json"
        
        # Configuration caches
        self._domain_config_cache = {}
        self._model_tier_cache = {}
        self._security_config_cache = None
        self._quality_config_cache = None
        
        # Config-driven parameters - loaded from domain mapping file
        self.tara_proven_params = {}
        self.trinity_config = {}
        self.model_tier_mappings = {}
        self.category_model_mappings = {}
        
        # Load all configurations
        self._load_all_configurations()
        
    def _load_all_configurations(self):
        """Load all configuration files"""
        try:
            self._load_domain_configurations()
            self._load_tara_and_trinity_configs()
            self._load_model_tier_configurations()
            self._load_security_configuration()
            self._load_quality_configuration()
            logger.info("? All configurations loaded successfully")
        except Exception as e:
            logger.error(f"? Error loading configurations: {e}")
            self._initialize_default_configurations()
            
    def _load_tara_and_trinity_configs(self):
        """Load TARA proven parameters and Trinity config from domain mapping file"""
        try:
            if self.domain_mapping_file.exists():
                with open(self.domain_mapping_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                # Load TARA proven parameters from config
                if 'tara_proven_params' in config:
                    self.tara_proven_params = config['tara_proven_params']
                    logger.info("? TARA proven parameters loaded from config")
                else:
                    self._initialize_default_tara_params()
                    
                # Load model tier mappings from config
                if 'model_tiers' in config:
                    self.model_tier_mappings = config['model_tiers']
                    logger.info("? Model tier mappings loaded from config")
                    
                # Load category model mappings from config  
                if 'category_model_mappings' in config:
                    self.category_model_mappings = config['category_model_mappings']
                    logger.info("? Category model mappings loaded from config")
                    
                # Initialize Trinity config (can be extended from config later)
                self.trinity_config = {
                    "arc_reactor_efficiency": 0.90,
                    "perplexity_intelligence": True,
                    "einstein_fusion_multiplier": 5.04,
                    "optimization_enabled": True,
                    "context_awareness": True,
                    "adaptive_routing": True
                }
                
                logger.info("? Config-driven parameters loaded successfully")
                
            else:
                logger.warning("?? Domain mapping file not found, using defaults")
                self._initialize_default_tara_params()
                
        except Exception as e:
            logger.error(f"? Error loading TARA/Trinity configs: {e}")
            self._initialize_default_tara_params()
            
    def _load_domain_configurations(self):
        """Load domain configurations from YAML using centralized mapping"""
        try:
            # Use centralized domain mapping
            domain_categories = get_domain_categories()
            domain_stats = get_domain_stats()
            
            logger.info(f"? Using centralized domain mapping:")
            logger.info(f"   ? Total domains: {domain_stats['total_domains']}")
            logger.info(f"   ? Categories: {domain_stats['total_categories']}")
            logger.info(f"   ? Config path: {domain_stats.get('config_path', 'Dynamic')}")
            
            # Create domain configurations from centralized mapping
            for category, domains in domain_categories.items():
                for domain_name in domains:
                    domain_config = DomainConfig(
                        domain_name=domain_name,
                        category=category,
                        model_tier=self._get_model_tier_from_category(category),
                        model_name=self._get_model_name_from_category(category),
                        quality_thresholds=self._get_category_quality_thresholds(category),
                        training_params=self._get_training_params_for_domain(domain_name),
                        optimization_strategy=self._get_optimization_strategy_for_domain(domain_name),
                        safety_critical=self._is_safety_critical_domain(category)
                    )
                    self._domain_config_cache[domain_name] = domain_config
                    
            logger.info(f"? Loaded {len(self._domain_config_cache)} domain configurations from centralized mapping")
            
        except Exception as e:
            logger.error(f"? Error loading domain configurations: {e}")
            self._initialize_default_domain_configurations()
            
    def _load_model_tier_configurations(self):
        """Load model tier configurations"""
        try:
            if self.model_tiers_file.exists():
                with open(self.model_tiers_file, 'r') as f:
                    config = json.load(f)
                    
                for tier_name, tier_config in config.items():
                    model_tier = ModelTierConfig(
                        name=tier_name,
                        model_path=tier_config.get("model_path", ""),
                        cost_per_hour=tier_config.get("cost_per_hour", 0.0),
                        recommended_gpu=tier_config.get("recommended_gpu", "T4"),
                        batch_size=tier_config.get("batch_size", 16),
                        sequence_length=tier_config.get("sequence_length", 128),
                        performance_tier=tier_config.get("performance_tier", "balanced")
                    )
                    self._model_tier_cache[tier_name] = model_tier
                    
                logger.info(f"? Loaded {len(self._model_tier_cache)} model tier configurations")
            else:
                self._initialize_default_model_tiers()
                
        except Exception as e:
            logger.error(f"? Error loading model tier configurations: {e}")
            self._initialize_default_model_tiers()
            
    def _load_security_configuration(self):
        """Load security configuration"""
        try:
            if self.security_config_file.exists():
                with open(self.security_config_file, 'r') as f:
                    config = json.load(f)
                    
                self._security_config_cache = SecurityConfig(**config)
                logger.info("? Security configuration loaded")
            else:
                self._security_config_cache = SecurityConfig()
                logger.info("? Default security configuration initialized")
                
        except Exception as e:
            logger.error(f"? Error loading security configuration: {e}")
            self._security_config_cache = SecurityConfig()
            
    def _load_quality_configuration(self):
        """Load quality configuration"""
        try:
            if self.quality_standards_file.exists():
                with open(self.quality_standards_file, 'r') as f:
                    config = json.load(f)
                    
                self._quality_config_cache = QualityConfig(**config)
                logger.info("? Quality configuration loaded")
            else:
                self._quality_config_cache = QualityConfig()
                logger.info("? Default quality configuration initialized")
                
        except Exception as e:
            logger.error(f"? Error loading quality configuration: {e}")
            self._quality_config_cache = QualityConfig()
            
    def get_domain_config(self, domain_name: str) -> Optional[DomainConfig]:
        """Get configuration for a specific domain"""
        return self._domain_config_cache.get(domain_name)
        
    def get_all_domain_configs(self) -> Dict[str, DomainConfig]:
        """Get all domain configurations"""
        return self._domain_config_cache.copy()
        
    def get_domains_by_category(self, category: str) -> List[DomainConfig]:
        """Get all domains in a specific category"""
        return [config for config in self._domain_config_cache.values() if config.category == category]
        
    def get_model_tier_config(self, tier_name: str) -> Optional[ModelTierConfig]:
        """Get model tier configuration"""
        return self._model_tier_cache.get(tier_name)
        
    def get_all_model_tiers(self) -> Dict[str, ModelTierConfig]:
        """Get all model tier configurations"""
        return self._model_tier_cache.copy()
        
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self._security_config_cache
        
    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration"""
        return self._quality_config_cache
        
    def get_tara_proven_params(self) -> Dict[str, Any]:
        """Get TARA proven parameters (config-driven)"""
        return self.tara_proven_params.copy()
        
    def get_trinity_config(self) -> Dict[str, Any]:
        """Get Trinity Architecture configuration (config-driven)"""
        return self.trinity_config.copy()
        
    def get_model_tier_mappings(self) -> Dict[str, str]:
        """Get model tier mappings (config-driven)"""
        return self.model_tier_mappings.copy()
        
    def get_category_model_mappings(self) -> Dict[str, str]:
        """Get category to model tier mappings (config-driven)"""
        return self.category_model_mappings.copy()
        
    def update_domain_config(self, domain_name: str, config_updates: Dict[str, Any]):
        """Update domain configuration"""
        if domain_name in self._domain_config_cache:
            domain_config = self._domain_config_cache[domain_name]
            
            # Update configuration fields
            for field, value in config_updates.items():
                if hasattr(domain_config, field):
                    setattr(domain_config, field, value)
                    
            logger.info(f"? Updated configuration for domain: {domain_name}")
        else:
            logger.warning(f"?? Domain not found: {domain_name}")
            
    def update_security_config(self, config_updates: Dict[str, Any]):
        """Update security configuration"""
        for field, value in config_updates.items():
            if hasattr(self._security_config_cache, field):
                setattr(self._security_config_cache, field, value)
                
        logger.info("? Security configuration updated")
        
    def update_quality_config(self, config_updates: Dict[str, Any]):
        """Update quality configuration"""
        for field, value in config_updates.items():
            if hasattr(self._quality_config_cache, field):
                setattr(self._quality_config_cache, field, value)
                
        logger.info("? Quality configuration updated")
        
    def save_all_configurations(self):
        """Save all configurations to files"""
        try:
            # Save domain configurations
            self._save_domain_configurations()
            
            # Save model tier configurations
            self._save_model_tier_configurations()
            
            # Save security configuration
            self._save_security_configuration()
            
            # Save quality configuration
            self._save_quality_configuration()
            
            logger.info("? All configurations saved successfully")
            
        except Exception as e:
            logger.error(f"? Error saving configurations: {e}")
            
    def _save_domain_configurations(self):
        """Save domain configurations to YAML"""
        domain_mapping = {}
        
        for domain_config in self._domain_config_cache.values():
            category = domain_config.category
            if category not in domain_mapping:
                domain_mapping[category] = {}
            domain_mapping[category][domain_config.domain_name] = domain_config.model_name
            
        with open(self.domain_mapping_file, 'w') as f:
            yaml.dump(domain_mapping, f, default_flow_style=False)
            
    def _save_model_tier_configurations(self):
        """Save model tier configurations to JSON"""
        model_tiers = {}
        
        for tier_config in self._model_tier_cache.values():
            model_tiers[tier_config.name] = asdict(tier_config)
            
        with open(self.model_tiers_file, 'w') as f:
            json.dump(model_tiers, f, indent=2)
            
    def _save_security_configuration(self):
        """Save security configuration to JSON"""
        with open(self.security_config_file, 'w') as f:
            json.dump(asdict(self._security_config_cache), f, indent=2)
            
    def _save_quality_configuration(self):
        """Save quality configuration to JSON"""
        with open(self.quality_standards_file, 'w') as f:
            json.dump(asdict(self._quality_config_cache), f, indent=2)
            
    def _get_model_tier_from_category(self, category: str) -> str:
        """Get model tier from category using config-driven mappings"""
        return self.category_model_mappings.get(category, "balanced")
    
    def _get_model_name_from_category(self, category: str) -> str:
        """Get model name from category using config-driven mappings"""
        model_tier = self._get_model_tier_from_category(category)
        return self.model_tier_mappings.get(model_tier, "microsoft/Phi-3.5-mini-instruct")
        
    def _get_category_quality_thresholds(self, category: str) -> Dict[str, float]:
        """Get quality thresholds for category"""
        thresholds = {
            "healthcare": {"accuracy": 95.0, "safety": 99.0, "relevance": 90.0},
            "specialized": {"expertise": 98.0, "precision": 96.0, "authority": 94.0},
            "business": {"accuracy": 90.0, "practicality": 92.0, "insight": 88.0},
            "education": {"pedagogical_quality": 93.0, "clarity": 95.0, "engagement": 87.0},
            "technology": {"technical_accuracy": 95.0, "practicality": 93.0, "depth": 90.0},
            "daily_life": {"relevance": 85.0, "helpfulness": 90.0, "empathy": 85.0},
            "creative": {"creativity": 88.0, "originality": 85.0, "inspiration": 90.0}
        }
        return thresholds.get(category, {"accuracy": 85.0, "relevance": 80.0})
        
    def _get_training_params_for_domain(self, domain_name: str) -> Dict[str, Any]:
        """Get training parameters for domain"""
        return {
            "batch_size": self.tara_proven_params["batch_size"],
            "lora_r": self.tara_proven_params["lora_r"],
            "max_steps": self.tara_proven_params["max_steps"],
            "learning_rate": self.tara_proven_params["learning_rate"],
            "sequence_length": self.tara_proven_params["sequence_length"],
            "early_stopping": True,
            "checkpoint_frequency": 50
        }
        
    def _get_optimization_strategy_for_domain(self, domain_name: str) -> Dict[str, Any]:
        """Get optimization strategy for domain"""
        return {
            "gpu_optimization": True,
            "cost_optimization": "moderate",
            "quality_target": "high",
            "speed_priority": "balanced"
        }
        
    def _is_safety_critical_domain(self, category: str) -> bool:
        """Check if domain category is safety critical"""
        return category in ["healthcare", "specialized"]
        
    def _initialize_default_configurations(self):
        """Initialize default configurations"""
        self._initialize_default_domain_configurations()
        self._initialize_default_model_tiers()
        self._security_config_cache = SecurityConfig()
        self._quality_config_cache = QualityConfig()
        
    def _initialize_default_domain_configurations(self):
        """Initialize default domain configurations"""
        # Healthcare domains
        healthcare_domains = ["general_health", "mental_health", "nutrition", "fitness", "sleep", "stress_management"]
        for domain in healthcare_domains:
            self._domain_config_cache[domain] = DomainConfig(
                domain_name=domain,
                category="healthcare",
                model_tier="quality",
                model_name="meta-llama/Llama-3.2-8B",
                quality_thresholds={"accuracy": 95.0, "safety": 99.0, "relevance": 90.0},
                training_params=self._get_training_params_for_domain(domain),
                optimization_strategy=self._get_optimization_strategy_for_domain(domain),
                safety_critical=True
            )
            
        # Business domains
        business_domains = ["entrepreneurship", "marketing", "sales", "customer_service"]
        for domain in business_domains:
            self._domain_config_cache[domain] = DomainConfig(
                domain_name=domain,
                category="business",
                model_tier="balanced",
                model_name="Qwen/Qwen2.5-7B",
                quality_thresholds={"accuracy": 90.0, "practicality": 92.0, "insight": 88.0},
                training_params=self._get_training_params_for_domain(domain),
                optimization_strategy=self._get_optimization_strategy_for_domain(domain),
                safety_critical=False
            )
            
        logger.info(f"? Initialized {len(self._domain_config_cache)} default domain configurations")
        
    def _initialize_default_model_tiers(self):
        """Initialize default model tier configurations"""
        self._model_tier_cache = {
            "lightning": ModelTierConfig(
                name="lightning",
                model_path="HuggingFaceTB/SmolLM2-1.7B",
                cost_per_hour=0.40,
                recommended_gpu="T4",
                batch_size=16,
                sequence_length=128,
                performance_tier="fast"
            ),
            "balanced": ModelTierConfig(
                name="balanced",
                model_path="Qwen/Qwen2.5-7B",
                cost_per_hour=2.50,
                recommended_gpu="V100",
                batch_size=32,
                sequence_length=256,
                performance_tier="balanced"
            ),
            "quality": ModelTierConfig(
                name="quality",
                model_path="meta-llama/Llama-3.2-8B",
                cost_per_hour=4.00,
                recommended_gpu="A100",
                batch_size=64,
                sequence_length=512,
                performance_tier="quality"
            )
        }
        
        logger.info(f"? Initialized {len(self._model_tier_cache)} default model tier configurations")

    def _initialize_default_tara_params(self):
        """Initialize default TARA proven parameters if config not available"""
        self.tara_proven_params = {
            "batch_size": 2,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "max_steps": 846,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 10,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "fp16": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8
        }
        
        self.trinity_config = {
            "arc_reactor_efficiency": 0.90,
            "perplexity_intelligence": True,
            "einstein_fusion_multiplier": 5.04,
            "optimization_enabled": True,
            "context_awareness": True,
            "adaptive_routing": True
        }
        
        logger.info("? Default TARA proven parameters initialized")

    def refresh_domain_configurations(self):
        """Refresh domain configurations from centralized mapping"""
        try:
            # Clear existing cache
            self._domain_config_cache.clear()
            
            # Reload from centralized mapping
            self._load_domain_configurations()
            
            logger.info("? Domain configurations refreshed from centralized mapping")
            
        except Exception as e:
            logger.error(f"? Error refreshing domain configurations: {e}")

    def get_centralized_domain_stats(self) -> Dict[str, Any]:
        """Get statistics from centralized domain mapping"""
        try:
            return get_domain_stats()
        except Exception as e:
            logger.error(f"? Error getting centralized domain stats: {e}")
            return {
                "total_domains": len(self._domain_config_cache),
                "total_categories": len(set(config.category for config in self._domain_config_cache.values())),
                "config_loaded": False,
                "error": str(e)
            }

    def refresh_config_from_file(self):
        """Refresh all configurations from files"""
        try:
            # Clear existing caches
            self._domain_config_cache.clear()
            self._model_tier_cache.clear()
            
            # Reload all configurations
            self._load_all_configurations()
            
            logger.info("? All configurations refreshed from files")
            
        except Exception as e:
            logger.error(f"? Error refreshing configurations: {e}")

    def get_config_driven_stats(self) -> Dict[str, Any]:
        """Get statistics about config-driven parameters"""
        return {
            "tara_params_loaded": len(self.tara_proven_params) > 0,
            "trinity_config_loaded": len(self.trinity_config) > 0,
            "model_tiers_loaded": len(self.model_tier_mappings) > 0,
            "category_mappings_loaded": len(self.category_model_mappings) > 0,
            "total_tara_params": len(self.tara_proven_params),
            "total_model_tiers": len(self.model_tier_mappings),
            "total_category_mappings": len(self.category_model_mappings),
            "config_file_exists": self.domain_mapping_file.exists()
        }

# Create global configuration manager instance
config_manager = TrinityConfigManager()

# Convenience functions for common configuration tasks
def get_domain_config(domain_name: str) -> Optional[DomainConfig]:
    """Quick access to domain configuration"""
    return config_manager.get_domain_config(domain_name)

def get_model_tier_config(tier_name: str) -> Optional[ModelTierConfig]:
    """Quick access to model tier configuration"""
    return config_manager.get_model_tier_config(tier_name)

def get_security_config() -> SecurityConfig:
    """Quick access to security configuration"""
    return config_manager.get_security_config()

def get_quality_config() -> QualityConfig:
    """Quick access to quality configuration"""
    return config_manager.get_quality_config()

def get_tara_proven_params() -> Dict[str, Any]:
    """Quick access to TARA proven parameters (config-driven)"""
    return config_manager.get_tara_proven_params()
