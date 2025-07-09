"""
MeeTARA Lab - Validation Utils with Trinity Architecture
Comprehensive validation utilities for 62-domain training and quality assurance
"""

import json
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import yaml
import re

from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# ==============================================================================
# Domain & Category Validation
# ==============================================================================
def validate_domain_exists(domain: str) -> bool:
    """
    Validates if a domain exists in the main config.
    """
    manager = SmartTrinityConfigManager()
    all_domains = manager.get_all_domains_flat()
    return domain in all_domains

def get_domains_for_category(category: str) -> List[str]:
    """
    Gets all domain names for a specific category from the main config.
    """
    manager = SmartTrinityConfigManager()
    all_domains = manager.get_all_domains_flat()
    
    return [
        domain_name for domain_name, details in all_domains.items() 
        if details.get("category") == category
    ]

# ==============================================================================
# GGUF & Model Validation
# ==============================================================================

def validate_gguf_file(file_path: Path) -> bool:
    """
    Validates if a GGUF file exists and is of a reasonable size.
    This is a placeholder for a more robust check.
    """
    # This is a placeholder for a more robust check
    return file_path.exists() and file_path.stat().st_size > 1_000_000 # > 1MB

def get_expected_gguf_filename(domain: str) -> str:
    """
    Generates the expected GGUF filename based on config parameters.
    """
    manager = SmartTrinityConfigManager()
    params = manager.get_tara_proven_params(domain)
    if not params:
        return f"{domain}_INVALID_CONFIG.gguf"
        
    quantization = params.get("quantization", "Q4_K_M")
    return f"{domain}-{quantization}.gguf"


# ==============================================================================
# Enhanced Validation with Trinity Intelligence
# ==============================================================================

class TrinityQualityValidator:
    """Trinity Architecture enhanced validation utilities"""
    
    def __init__(self):
        # TARA proven validation standards
        self.tara_standards = {
            "target_validation_score": 101.0,  # 101% validation target
            "minimum_quality_threshold": 80.0,  # 80% minimum quality
            "data_filter_success_rate": 31.0,   # 31% filter success rate
            "samples_per_domain": 2000,         # 2000+ samples per domain
            "max_loss_threshold": 0.5,          # Maximum loss threshold
            "convergence_minimum": 0.1          # Minimum convergence rate
        }
        
        # Validation patterns for different data types
        self.validation_patterns = {
            "domain_name": r"^[a-z_]+$",
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "model_name": r"^[a-zA-Z0-9/_.-]+$",
            "score": r"^[0-9]+(\.[0-9]+)?$"
        }
        
        # Domain-specific validation requirements
        self.domain_validation_requirements = {
            "healthcare": {
                "safety_critical": True,
                "min_validation_score": 95.0,
                "crisis_detection": True,
                "regulatory_compliance": ["FDA", "HIPAA"]
            },
            "specialized": {
                "safety_critical": True,
                "min_validation_score": 92.0,
                "precision_required": True,
                "authority_validation": True
            },
            "business": {
                "safety_critical": False,
                "min_validation_score": 88.0,
                "roi_validation": True,
                "practicality_check": True
            },
            "education": {
                "safety_critical": False,
                "min_validation_score": 87.0,
                "pedagogical_validation": True,
                "age_appropriateness": True
            },
            "technology": {
                "safety_critical": False,
                "min_validation_score": 87.0,
                "technical_accuracy": True,
                "security_validation": True
            },
            "daily_life": {
                "safety_critical": False,
                "min_validation_score": 85.0,
                "empathy_validation": True,
                "cultural_sensitivity": True
            },
            "creative": {
                "safety_critical": False,
                "min_validation_score": 82.0,
                "creativity_validation": True,
                "originality_check": True
            }
        }
        
        self.qa_score = None
        self.validation_report = {}

    def run_validation_suite(self, gguf_path: Path, domain: str) -> Dict[str, Any]:
        """
        Runs the complete validation suite for a given GGUF file and domain.
        """
        self.validation_report = {
            "domain": domain,
            "gguf_path": str(gguf_path),
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # 1. File Integrity
        self.validation_report["checks"]["file_integrity"] = self._validate_file_integrity(gguf_path)
        
        # 2. Config Alignment
        expected_filename = get_expected_gguf_filename(domain)
        self.validation_report["checks"]["config_alignment"] = self._validate_config_alignment(
            gguf_path.name, expected_filename
        )
        
        # 3. Performance Simulation
        self.validation_report["checks"]["performance_simulation"] = self._simulate_performance(gguf_path)
        
        # 4. Quality Assessment (Simulated)
        self.validation_report["checks"]["quality_assessment"] = self._assess_quality_simulation(domain)

        return self.validation_report

    def _validate_file_integrity(self, gguf_path: Path) -> Dict[str, Any]:
        """Validate the basic integrity of the GGUF file."""
        if not gguf_path.exists():
            return {"status": "FAIL", "reason": "File does not exist."}
        
        if gguf_path.stat().st_size < 1_000_000: # Less than 1MB
            return {"status": "FAIL", "reason": f"File size ({gguf_path.stat().st_size} bytes) is suspiciously small."}
            
        return {"status": "PASS", "details": f"File size: {gguf_path.stat().st_size} bytes."}

    def _validate_config_alignment(self, actual_filename: str, expected_filename: str) -> Dict[str, Any]:
        """Check if the filename aligns with configuration parameters."""
        if actual_filename == expected_filename:
            return {"status": "PASS", "details": f"Filename '{actual_filename}' matches config."}
        else:
            return {
                "status": "FAIL", 
                "reason": "Filename does not match configuration.",
                "details": {
                    "actual": actual_filename,
                    "expected": expected_filename
                }
            }
            
    def _simulate_performance(self, gguf_path: Path) -> Dict[str, Any]:
        """Simulate a performance check."""
        # In a real scenario, this would involve loading the model and timing inference.
        # For demonstration, we'll simulate a value.
        import random # Added missing import for random
        simulated_ms_per_token = random.uniform(20, 100)
        return {"status": "PASS", "details": f"Simulated {simulated_ms_per_token:.2f} ms/token."}

    def _assess_quality_simulation(self, domain: str) -> Dict[str, Any]:
        """Simulate a quality check using perplexity or other metrics."""
        manager = SmartTrinityConfigManager()
        params = manager.get_tara_proven_params(domain)
        target_score = params.get("validation_target", 99.0)
        
        # Simulate a score that is usually close to the target
        simulated_score = target_score * random.uniform(0.98, 1.02)
        
        status = "PASS" if simulated_score >= target_score else "FAIL"
        
        return {
            "status": status, 
            "details": f"Simulated score: {simulated_score:.2f} (Target: {target_score})"
        }

# This is a legacy function and should be removed or updated if still needed.
# For now, it's an example of what NOT to do.
def get_hardcoded_domain_list_for_testing_purpose_only(category: str) -> List[str]:
    """
    DEPRECATED: Returns a hardcoded list of domains for a category.
    This is a bad practice and is here only for historical reference during refactoring.
    """
    hardcoded_map = {
        "healthcare": ["health_advisor", "medical_scribe", "fitness_coach"],
        "business": ["business_analyst", "sales_pitch_generator", "market_researcher"]
    }
    return hardcoded_map.get(category, [])
