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
from trinity_core.config_manager import get_all_domain_categories

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class TrinityValidationUtils:
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
        
    def validate_training_data(self, data: Dict[str, Any], domain: str) -> Tuple[bool, List[str], float]:
        """Validate training data for a specific domain"""
        errors = []
        warnings = []
        
        # Basic structure validation
        if not self._validate_data_structure(data):
            errors.append("Invalid data structure")
            
        # Domain-specific validation
        domain_category = self._get_domain_category(domain)
        domain_reqs = self.domain_validation_requirements.get(domain_category, {})
        
        # Quality score validation
        quality_score = self._calculate_data_quality_score(data, domain_reqs)
        
        if quality_score < self.tara_standards["minimum_quality_threshold"]:
            errors.append(f"Quality score {quality_score:.2f} below minimum threshold")
            
        # Sample count validation
        if "samples" in data:
            sample_count = len(data["samples"])
            if sample_count < self.tara_standards["samples_per_domain"]:
                warnings.append(f"Sample count {sample_count} below recommended {self.tara_standards['samples_per_domain']}")
                
        # Safety-critical validation
        if domain_reqs.get("safety_critical", False):
            safety_errors = self._validate_safety_critical_data(data, domain)
            errors.extend(safety_errors)
            
        is_valid = len(errors) == 0
        return is_valid, errors + warnings, quality_score
        
    def validate_training_metrics(self, metrics: Dict[str, Any], domain: str) -> Tuple[bool, List[str], Dict[str, float]]:
        """Validate training metrics and performance"""
        errors = []
        validation_scores = {}
        
        # Validation score check
        validation_score = metrics.get("validation_score", 0.0)
        domain_category = self._get_domain_category(domain)
        min_score = self.domain_validation_requirements.get(domain_category, {}).get("min_validation_score", 85.0)
        
        if validation_score < min_score:
            errors.append(f"Validation score {validation_score:.2f} below minimum {min_score}")
            
        validation_scores["validation_score"] = validation_score
        
        # Loss validation
        loss = metrics.get("loss", float('inf'))
        if loss > self.tara_standards["max_loss_threshold"]:
            errors.append(f"Loss {loss:.4f} exceeds maximum threshold {self.tara_standards['max_loss_threshold']}")
            
        validation_scores["loss"] = loss
        
        # Convergence validation
        convergence_rate = metrics.get("convergence_rate", 0.0)
        if convergence_rate < self.tara_standards["convergence_minimum"]:
            errors.append(f"Convergence rate {convergence_rate:.4f} below minimum {self.tara_standards['convergence_minimum']}")
            
        validation_scores["convergence_rate"] = convergence_rate
        
        # TARA benchmark comparison
        tara_comparison = self._compare_to_tara_benchmarks(validation_scores)
        validation_scores.update(tara_comparison)
        
        is_valid = len(errors) == 0
        return is_valid, errors, validation_scores
        
    def validate_model_output(self, output: Dict[str, Any], domain: str, context: Dict[str, Any] = None) -> Tuple[bool, List[str], float]:
        """Validate model output quality and safety"""
        errors = []
        
        # Basic output validation
        if not output or not isinstance(output, dict):
            errors.append("Invalid output format")
            return False, errors, 0.0
            
        # Content validation
        if "response" not in output:
            errors.append("Missing response in output")
            
        # Domain-specific output validation
        domain_category = self._get_domain_category(domain)
        domain_reqs = self.domain_validation_requirements.get(domain_category, {})
        
        # Safety validation for critical domains
        if domain_reqs.get("safety_critical", False):
            safety_errors = self._validate_output_safety(output, domain)
            errors.extend(safety_errors)
            
        # Crisis detection validation
        if domain_reqs.get("crisis_detection", False):
            crisis_errors = self._validate_crisis_handling(output, context)
            errors.extend(crisis_errors)
            
        # Quality score calculation
        quality_score = self._calculate_output_quality_score(output, domain_reqs)
        
        is_valid = len(errors) == 0
        return is_valid, errors, quality_score
        
    def validate_domain_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate domain configuration"""
        errors = []
        
        # Required fields validation
        required_fields = ["domain_name", "category", "model_tier", "quality_thresholds"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        # Domain name validation
        if "domain_name" in config:
            if not re.match(self.validation_patterns["domain_name"], config["domain_name"]):
                errors.append("Invalid domain name format")
                
        # Category validation
        if "category" in config:
            valid_categories = list(self.domain_validation_requirements.keys())
            if config["category"] not in valid_categories:
                errors.append(f"Invalid category. Must be one of: {valid_categories}")
                
        # Quality thresholds validation
        if "quality_thresholds" in config:
            threshold_errors = self._validate_quality_thresholds(config["quality_thresholds"])
            errors.extend(threshold_errors)
            
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def calculate_validation_score(self, metrics: Dict[str, Any], weights: Dict[str, float] = None) -> float:
        """Calculate comprehensive validation score"""
        if weights is None:
            weights = {
                "accuracy": 0.25,
                "quality": 0.25,
                "safety": 0.20,
                "empathy": 0.15,
                "consistency": 0.15
            }
            
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score = float(metrics[metric])
                # Normalize score to 0-100 range
                normalized_score = min(max(score, 0.0), 100.0)
                total_score += normalized_score * weight
                total_weight += weight
                
        if total_weight == 0:
            return 0.0
            
        final_score = total_score / total_weight
        return round(final_score, 2)
        
    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against schema"""
        errors = []
        
        # Type validation
        for field, field_schema in schema.items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(data[field], expected_type):
                    errors.append(f"Field '{field}' has wrong type. Expected {expected_type.__name__}")
                    
                # Range validation
                if "range" in field_schema and isinstance(data[field], (int, float)):
                    min_val, max_val = field_schema["range"]
                    if not (min_val <= data[field] <= max_val):
                        errors.append(f"Field '{field}' value {data[field]} out of range [{min_val}, {max_val}]")
                        
            elif field_schema.get("required", False):
                errors.append(f"Required field '{field}' is missing")
                
        is_valid = len(errors) == 0
        return is_valid, errors
        
    def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate basic data structure"""
        if not isinstance(data, dict):
            return False
            
        # Check for essential fields
        essential_fields = ["domain", "samples", "metadata"]
        return all(field in data for field in essential_fields)
        
    def _get_domain_category(self, domain: str) -> str:
        """Get domain category for validation"""
        # SMART: Load from YAML config instead of hardcoding
        domain_categories = get_all_domain_categories()
        
        if domain in domain_categories:
            return domain
        else:
            return "daily_life"  # Default category
            
    def _calculate_data_quality_score(self, data: Dict[str, Any], domain_reqs: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        score = 0.0
        max_score = 0.0
        
        # Sample quality check
        if "samples" in data:
            samples = data["samples"]
            if isinstance(samples, list) and len(samples) > 0:
                score += 25.0  # Base score for having samples
                
                # Sample diversity check
                if len(samples) >= 100:
                    score += 25.0
                    
        max_score += 50.0
        
        # Metadata quality check
        if "metadata" in data:
            metadata = data["metadata"]
            if isinstance(metadata, dict):
                score += 20.0
                
                # Domain-specific metadata
                if "domain_specific" in metadata:
                    score += 15.0
                    
        max_score += 35.0
        
        # Quality validation check
        if "quality_metrics" in data:
            score += 15.0
            
        max_score += 15.0
        
        return (score / max_score) * 100.0 if max_score > 0 else 0.0
        
    def _validate_safety_critical_data(self, data: Dict[str, Any], domain: str) -> List[str]:
        """Validate safety-critical data"""
        errors = []
        
        # Check for safety keywords
        safety_keywords = ["emergency", "crisis", "danger", "harm", "suicide", "violence"]
        
        if "samples" in data:
            for sample in data["samples"]:
                if isinstance(sample, dict) and "text" in sample:
                    text = sample["text"].lower()
                    for keyword in safety_keywords:
                        if keyword in text:
                            # Check if proper safety handling is present
                            if "safety_response" not in sample:
                                errors.append(f"Missing safety response for sample containing '{keyword}'")
                                
        return errors
        
    def _validate_output_safety(self, output: Dict[str, Any], domain: str) -> List[str]:
        """Validate output safety"""
        errors = []
        
        # Check for harmful content
        if "response" in output:
            response = output["response"].lower()
            harmful_indicators = ["dangerous", "harmful", "illegal", "inappropriate"]
            
            for indicator in harmful_indicators:
                if indicator in response:
                    errors.append(f"Potentially harmful content detected: {indicator}")
                    
        return errors
        
    def _validate_crisis_handling(self, output: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """Validate crisis handling capability"""
        errors = []
        
        if context and context.get("crisis_detected", False):
            # Check if crisis response is present
            if "crisis_response" not in output:
                errors.append("Missing crisis response for detected crisis situation")
                
            # Check if professional referral is provided
            if "professional_referral" not in output:
                errors.append("Missing professional referral for crisis situation")
                
        return errors
        
    def _calculate_output_quality_score(self, output: Dict[str, Any], domain_reqs: Dict[str, Any]) -> float:
        """Calculate output quality score"""
        score = 0.0
        max_score = 0.0
        
        # Response quality
        if "response" in output and output["response"]:
            score += 30.0
            
            # Length check
            if len(output["response"]) >= 50:
                score += 20.0
                
        max_score += 50.0
        
        # Confidence score
        if "confidence" in output:
            confidence = float(output["confidence"])
            score += confidence * 0.25  # 25% weight
            
        max_score += 25.0
        
        # Empathy score (if required)
        if domain_reqs.get("empathy_validation", False):
            if "empathy_score" in output:
                empathy = float(output["empathy_score"])
                score += empathy * 0.25
            else:
                score += 10.0  # Default empathy score
                
        max_score += 25.0
        
        return (score / max_score) * 100.0 if max_score > 0 else 0.0
        
    def _validate_quality_thresholds(self, thresholds: Dict[str, Any]) -> List[str]:
        """Validate quality thresholds"""
        errors = []
        
        # Check for required thresholds
        required_thresholds = ["accuracy", "safety", "relevance"]
        for threshold in required_thresholds:
            if threshold not in thresholds:
                errors.append(f"Missing required threshold: {threshold}")
                
        # Validate threshold values
        for threshold_name, threshold_value in thresholds.items():
            if not isinstance(threshold_value, (int, float)):
                errors.append(f"Threshold '{threshold_name}' must be numeric")
            elif not (0 <= threshold_value <= 100):
                errors.append(f"Threshold '{threshold_name}' must be between 0 and 100")
                
        return errors
        
    def _compare_to_tara_benchmarks(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Compare scores to TARA benchmarks"""
        comparison = {}
        
        # Validation score comparison
        if "validation_score" in scores:
            tara_target = self.tara_standards["target_validation_score"]
            comparison["tara_validation_ratio"] = (scores["validation_score"] / tara_target) * 100
            
        # Quality threshold comparison
        if "quality_score" in scores:
            tara_quality = self.tara_standards["minimum_quality_threshold"]
            comparison["tara_quality_ratio"] = (scores["quality_score"] / tara_quality) * 100
            
        return comparison

    def get_domain_categories_for_validation(self) -> Dict[str, List[str]]:
        """Get domain categories for validation - LOADS FROM YAML CONFIG"""
        domain_categories = get_all_domain_categories()
        
        print(f"✅ Loaded {len(domain_categories)} categories from YAML config")
        for category, domains in domain_categories.items():
            print(f"   {category}: {len(domains)} domains")
        
        return domain_categories

# Create global validation instance
validation_utils = TrinityValidationUtils()

# Convenience functions for common validation tasks
def validate_training_data(data: Dict[str, Any], domain: str) -> Tuple[bool, List[str], float]:
    """Quick validation for training data"""
    return validation_utils.validate_training_data(data, domain)

def validate_training_metrics(metrics: Dict[str, Any], domain: str) -> Tuple[bool, List[str], Dict[str, float]]:
    """Quick validation for training metrics"""
    return validation_utils.validate_training_metrics(metrics, domain)

def calculate_validation_score(metrics: Dict[str, Any], weights: Dict[str, float] = None) -> float:
    """Quick validation score calculation"""
    return validation_utils.calculate_validation_score(metrics, weights)

def validate_domain_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Quick domain configuration validation"""
    return validation_utils.validate_domain_configuration(config)
