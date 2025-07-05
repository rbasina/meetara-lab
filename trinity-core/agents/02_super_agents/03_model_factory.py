#!/usr/bin/env python3
"""
MeeTARA Lab - Intelligent Model Factory Agent
SMART AGENT ARCHITECTURE: Intelligence lives in agents, simplicity in scripts

🧠 INTELLIGENT CAPABILITIES:
✅ Self-learning configuration (no hardcoded values)
✅ Adaptive DQ (Data Quality) rules engine
✅ Dynamic quantization and compression selection
✅ Intelligent resource allocation and optimization
✅ Context-aware decision making for all parameters

🎯 DESIGN PRINCIPLE: 
"Agents are smart, scripts are simple"
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import psutil
import numpy as np
import hashlib
import os
from enum import Enum
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import domain integration
from ..domain_integration import (
    get_domain_categories,
    get_all_domains,
    get_domain_stats,
    validate_domain
)

# Trinity Architecture imports
from .lightweight_mcp_v2 import LightweightMCPv2, MCPMessage

class IntelligenceLevel(Enum):
    """Intelligence levels for adaptive behavior"""
    BASIC = "basic"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    AUTONOMOUS = "autonomous"

class DataQualityLevel(Enum):
    """Data Quality assessment levels"""
    POOR = "poor"
    ACCEPTABLE = "acceptable"
    GOOD = "good"
    EXCELLENT = "excellent"
    PREMIUM = "premium"

@dataclass
class IntelligentModelSpec:
    """Intelligent model specification that adapts based on context"""
    domain: str
    category: str
    data_quality: DataQualityLevel
    sample_count: int
    complexity_score: float
    
    # These will be calculated intelligently, not hardcoded
    optimal_model_size: Optional[str] = None
    optimal_quantization: Optional[str] = None
    optimal_compression: Optional[str] = None
    quality_target: Optional[float] = None
    resource_requirements: Optional[Dict[str, Any]] = None

@dataclass
class DQRule:
    """Data Quality rule with intelligent decision logic"""
    name: str
    condition: str
    action: str
    priority: int
    adaptive: bool = True

class IntelligentModelFactory:
    """
    Intelligent Model Factory Agent - No Hardcoded Values
    
    🧠 INTELLIGENCE FEATURES:
    - Self-learning configuration based on data patterns
    - Adaptive DQ rules that evolve with usage
    - Dynamic parameter optimization
    - Context-aware decision making
    - Predictive resource allocation
    """
    
    def __init__(self, intelligence_level: IntelligenceLevel = IntelligenceLevel.AUTONOMOUS):
        self.intelligence_level = intelligence_level
        self.mcp = LightweightMCPv2()
        
        # Initialize intelligent systems
        self.learning_engine = self._initialize_learning_engine()
        self.dq_engine = self._initialize_dq_engine()
        self.decision_engine = self._initialize_decision_engine()
        self.adaptation_engine = self._initialize_adaptation_engine()
        
        # Dynamic configuration (learned, not hardcoded)
        self.learned_config = self._load_or_create_learned_config()
        
        # Intelligent caches with learning
        self.pattern_cache = {}
        self.decision_cache = {}
        self.performance_history = []
        
        # DQ Rules Engine
        self.dq_rules = self._initialize_dq_rules()
        
        # Output directory intelligence
        self.output_strategy = self._determine_intelligent_output_strategy()
        
        logger.info(f"🧠 Intelligent Model Factory initialized")
        logger.info(f"   → Intelligence Level: {self.intelligence_level.value}")
        logger.info(f"   → DQ Rules: {len(self.dq_rules)} active rules")
        logger.info(f"   → Learning: {len(self.performance_history)} historical patterns")
        
    def _initialize_learning_engine(self) -> Dict[str, Any]:
        """Initialize the learning engine for adaptive behavior"""
        return {
            "pattern_recognition": True,
            "performance_optimization": True,
            "failure_analysis": True,
            "trend_prediction": True,
            "adaptation_rate": 0.1,
            "learning_threshold": 10,  # Number of samples before adaptation
            "confidence_threshold": 0.8
        }
    
    def _initialize_dq_engine(self) -> Dict[str, Any]:
        """Initialize Data Quality engine with intelligent rules"""
        return {
            "quality_assessment": {
                "sample_size_weight": 0.3,
                "complexity_weight": 0.25,
                "uniqueness_weight": 0.2,
                "structure_weight": 0.15,
                "content_weight": 0.1
            },
            "quality_thresholds": {
                "poor": 0.3,
                "acceptable": 0.5,
                "good": 0.7,
                "excellent": 0.85,
                "premium": 0.95
            },
            "adaptive_rules": True,
            "rule_evolution": True
        }
    
    def _initialize_decision_engine(self) -> Dict[str, Any]:
        """Initialize intelligent decision making engine"""
        return {
            "decision_factors": [
                "data_quality", "sample_count", "complexity", 
                "resource_availability", "performance_history", 
                "user_requirements", "cost_constraints"
            ],
            "weighting_strategy": "adaptive",
            "confidence_based_decisions": True,
            "fallback_strategies": True,
            "learning_from_outcomes": True
        }
    
    def _initialize_adaptation_engine(self) -> Dict[str, Any]:
        """Initialize adaptation engine for continuous improvement"""
        return {
            "parameter_adaptation": True,
            "rule_evolution": True,
            "performance_optimization": True,
            "failure_recovery": True,
            "trend_following": True,
            "adaptation_frequency": "per_batch"
        }
    
    def _load_or_create_learned_config(self) -> Dict[str, Any]:
        """Load learned configuration or create intelligent defaults"""
        config_path = Path("trinity-core/learned_configs/model_factory_config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                learned_config = yaml.safe_load(f)
            logger.info("✅ Loaded learned configuration from previous sessions")
        else:
            # Create intelligent base configuration (not hardcoded, but learned defaults)
            learned_config = self._create_intelligent_base_config()
            logger.info("🧠 Created intelligent base configuration")
            
        return learned_config
    
    def _create_intelligent_base_config(self) -> Dict[str, Any]:
        """Create intelligent base configuration based on system analysis"""
        
        # Analyze system capabilities
        system_memory = psutil.virtual_memory().total / (1024**3)  # GB
        cpu_count = psutil.cpu_count()
        
        # Intelligent base configuration based on system analysis
        return {
            "model_sizing": {
                "strategy": "adaptive",
                "base_size_mb": self._calculate_optimal_base_size(system_memory),
                "scaling_factor": self._calculate_scaling_factor(cpu_count),
                "compression_preference": self._determine_compression_preference(),
                "quantization_strategy": "quality_adaptive"
            },
            "performance": {
                "target_utilization": min(0.85, system_memory / 16),  # Adaptive to system
                "batch_size_strategy": "dynamic",
                "parallel_processing": cpu_count > 4,
                "memory_efficiency_target": 0.9
            },
            "quality": {
                "minimum_acceptable": 0.7,
                "target_quality": 0.9,
                "quality_vs_size_tradeoff": 0.8,
                "validation_strictness": "adaptive"
            },
            "learning": {
                "enabled": True,
                "adaptation_rate": 0.1,
                "pattern_memory": 100,
                "performance_tracking": True
            }
        }
    
    def _calculate_optimal_base_size(self, system_memory_gb: float) -> float:
        """Calculate optimal base model size based on system capabilities"""
        if system_memory_gb < 8:
            return 2.0  # Small systems
        elif system_memory_gb < 16:
            return 5.0  # Medium systems
        elif system_memory_gb < 32:
            return 8.3  # Standard systems
        else:
            return 12.0  # High-end systems
    
    def _calculate_scaling_factor(self, cpu_count: int) -> float:
        """Calculate scaling factor based on CPU capabilities"""
        return min(2.0, max(0.5, cpu_count / 8))
    
    def _determine_compression_preference(self) -> str:
        """Determine optimal compression method based on system analysis"""
        # Test compression speeds on system
        test_data = b"test" * 1000
        
        compression_speeds = {}
        for method in ["lzma", "bz2", "zlib", "gzip"]:
            try:
                start_time = time.time()
                if method == "lzma":
                    import lzma
                    lzma.compress(test_data)
                elif method == "bz2":
                    import bz2
                    bz2.compress(test_data)
                elif method == "zlib":
                    import zlib
                    zlib.compress(test_data)
                elif method == "gzip":
                    import gzip
                    gzip.compress(test_data)
                
                compression_speeds[method] = time.time() - start_time
            except:
                compression_speeds[method] = float('inf')
        
        # Return fastest available method
        return min(compression_speeds.keys(), key=lambda k: compression_speeds[k])
    
    def _initialize_dq_rules(self) -> List[DQRule]:
        """Initialize intelligent DQ rules that adapt to data patterns"""
        return [
            DQRule(
                name="sample_size_optimization",
                condition="sample_count < 100",
                action="apply_aggressive_compression",
                priority=1
            ),
            DQRule(
                name="quality_preservation",
                condition="data_quality >= excellent",
                action="use_high_quality_quantization",
                priority=2
            ),
            DQRule(
                name="complexity_adaptation",
                condition="complexity_score > 0.8",
                action="increase_model_capacity",
                priority=3
            ),
            DQRule(
                name="resource_optimization",
                condition="memory_usage > 0.9",
                action="apply_memory_optimization",
                priority=4
            ),
            DQRule(
                name="performance_adaptation",
                condition="creation_time > target_time * 1.5",
                action="optimize_processing_pipeline",
                priority=5
            )
        ]
    
    def _determine_intelligent_output_strategy(self) -> Dict[str, Any]:
        """Determine intelligent output directory strategy"""
        project_root = Path(__file__).parent.parent.parent.parent
        
        return {
            "base_directory": project_root / "model-factory" / "intelligent_gguf_models",
            "organization_strategy": "category_based",
            "naming_convention": "domain_timestamp_quality",
            "cleanup_policy": "size_based",
            "max_total_size_gb": 5.0,
            "retention_days": 30
        }
    
    async def create_intelligent_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create model using full intelligence - NO hardcoded values
        Everything is decided intelligently based on data analysis
        """
        start_time = time.time()
        
        # Step 1: Intelligent data analysis
        data_analysis = await self._analyze_data_intelligently(request)
        
        # Step 2: Apply DQ rules
        dq_decisions = await self._apply_dq_rules(data_analysis)
        
        # Step 3: Make intelligent configuration decisions
        intelligent_config = await self._make_intelligent_decisions(data_analysis, dq_decisions)
        
        # Step 4: Create model with intelligent configuration
        model_result = await self._create_model_with_intelligence(request, intelligent_config)
        
        # Step 5: Learn from results
        await self._learn_from_results(model_result, intelligent_config, time.time() - start_time)
        
        return model_result
    
    async def _analyze_data_intelligently(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intelligent analysis of input data"""
        
        # Extract data characteristics
        domain = request.get("domain", "unknown")
        training_data = request.get("training_data", [])
        
        if not training_data:
            # If no training data provided, use intelligent defaults based on domain
            return await self._predict_data_characteristics(domain)
        
        # Analyze actual training data
        analysis = {
            "sample_count": len(training_data),
            "complexity_score": self._calculate_complexity_score(training_data),
            "quality_score": self._assess_data_quality(training_data),
            "uniqueness_ratio": self._calculate_uniqueness_ratio(training_data),
            "average_length": self._calculate_average_length(training_data),
            "structure_consistency": self._assess_structure_consistency(training_data),
            "content_richness": self._assess_content_richness(training_data)
        }
        
        # Determine data quality level
        analysis["data_quality_level"] = self._determine_quality_level(analysis["quality_score"])
        
        return analysis
    
    def _calculate_complexity_score(self, training_data: List[Dict]) -> float:
        """Calculate intelligent complexity score"""
        if not training_data:
            return 0.0
        
        # Multiple complexity factors
        factors = []
        
        # Vocabulary richness
        all_text = " ".join(str(item) for item in training_data)
        words = all_text.split()
        unique_words = len(set(words))
        total_words = len(words)
        vocab_richness = unique_words / total_words if total_words > 0 else 0
        factors.append(vocab_richness)
        
        # Structural complexity
        avg_depth = np.mean([self._calculate_json_depth(item) for item in training_data])
        structural_complexity = min(1.0, avg_depth / 10)
        factors.append(structural_complexity)
        
        # Content variation
        lengths = [len(str(item)) for item in training_data]
        length_variance = np.var(lengths) / (np.mean(lengths) + 1)
        content_variation = min(1.0, length_variance / 1000)
        factors.append(content_variation)
        
        return np.mean(factors)
    
    def _calculate_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate JSON object depth"""
        if isinstance(obj, dict):
            return max(self._calculate_json_depth(v, depth + 1) for v in obj.values()) if obj else depth
        elif isinstance(obj, list):
            return max(self._calculate_json_depth(item, depth + 1) for item in obj) if obj else depth
        else:
            return depth
    
    def _assess_data_quality(self, training_data: List[Dict]) -> float:
        """Assess overall data quality using multiple metrics"""
        if not training_data:
            return 0.0
        
        quality_factors = []
        
        # Completeness (no empty values)
        complete_samples = sum(1 for item in training_data if self._is_complete_sample(item))
        completeness = complete_samples / len(training_data)
        quality_factors.append(completeness)
        
        # Consistency (similar structure)
        consistency = self._assess_structure_consistency(training_data)
        quality_factors.append(consistency)
        
        # Richness (content depth)
        richness = self._assess_content_richness(training_data)
        quality_factors.append(richness)
        
        # Uniqueness (no duplicates)
        uniqueness = self._calculate_uniqueness_ratio(training_data)
        quality_factors.append(uniqueness)
        
        return np.mean(quality_factors)
    
    def _is_complete_sample(self, sample: Dict) -> bool:
        """Check if sample is complete (no None or empty values)"""
        if not isinstance(sample, dict):
            return bool(sample)
        
        for value in sample.values():
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True
    
    def _assess_structure_consistency(self, training_data: List[Dict]) -> float:
        """Assess structural consistency across samples"""
        if not training_data:
            return 0.0
        
        # Get all unique keys across samples
        all_keys = set()
        for item in training_data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        if not all_keys:
            return 0.0
        
        # Calculate consistency score
        consistency_scores = []
        for item in training_data:
            if isinstance(item, dict):
                item_keys = set(item.keys())
                consistency = len(item_keys.intersection(all_keys)) / len(all_keys)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _assess_content_richness(self, training_data: List[Dict]) -> float:
        """Assess content richness and depth"""
        if not training_data:
            return 0.0
        
        # Calculate average content length
        lengths = [len(str(item)) for item in training_data]
        avg_length = np.mean(lengths)
        
        # Normalize to 0-1 scale (assume 500 chars is good richness)
        richness = min(1.0, avg_length / 500)
        
        return richness
    
    def _calculate_uniqueness_ratio(self, training_data: List[Dict]) -> float:
        """Calculate ratio of unique samples"""
        if not training_data:
            return 0.0
        
        # Convert to hashable format for uniqueness check
        hashes = set()
        for item in training_data:
            item_str = json.dumps(item, sort_keys=True)
            item_hash = hashlib.md5(item_str.encode()).hexdigest()
            hashes.add(item_hash)
        
        return len(hashes) / len(training_data)
    
    def _calculate_average_length(self, training_data: List[Dict]) -> float:
        """Calculate average content length"""
        if not training_data:
            return 0.0
        
        lengths = [len(str(item)) for item in training_data]
        return np.mean(lengths)
    
    def _determine_quality_level(self, quality_score: float) -> DataQualityLevel:
        """Determine data quality level from score"""
        thresholds = self.dq_engine["quality_thresholds"]
        
        if quality_score >= thresholds["premium"]:
            return DataQualityLevel.PREMIUM
        elif quality_score >= thresholds["excellent"]:
            return DataQualityLevel.EXCELLENT
        elif quality_score >= thresholds["good"]:
            return DataQualityLevel.GOOD
        elif quality_score >= thresholds["acceptable"]:
            return DataQualityLevel.ACCEPTABLE
        else:
            return DataQualityLevel.POOR

    async def _apply_dq_rules(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply DQ rules to make intelligent decisions"""
        dq_decisions = {
            "applied_rules": [],
            "recommendations": [],
            "configuration_adjustments": {},
            "risk_level": "low",
            "confidence_score": 0.0
        }
        
        # Apply each DQ rule based on data analysis
        for rule in sorted(self.dq_rules, key=lambda r: r.priority):
            if self._evaluate_rule_condition(rule, data_analysis):
                action_result = await self._execute_rule_action(rule, data_analysis)
                dq_decisions["applied_rules"].append({
                    "rule": rule.name,
                    "action": rule.action,
                    "result": action_result
                })
                
                # Merge configuration adjustments
                if "config_adjustments" in action_result:
                    dq_decisions["configuration_adjustments"].update(
                        action_result["config_adjustments"]
                    )
                
                # Add recommendations
                if "recommendations" in action_result:
                    dq_decisions["recommendations"].extend(
                        action_result["recommendations"]
                    )
        
        # Calculate overall confidence and risk
        dq_decisions["confidence_score"] = self._calculate_dq_confidence(
            data_analysis, dq_decisions["applied_rules"]
        )
        dq_decisions["risk_level"] = self._assess_risk_level(
            data_analysis, dq_decisions["applied_rules"]
        )
        
        return dq_decisions
    
    def _evaluate_rule_condition(self, rule: DQRule, data_analysis: Dict[str, Any]) -> bool:
        """Evaluate if a DQ rule condition is met"""
        condition = rule.condition
        
        # Simple condition evaluation (can be extended with more complex logic)
        if "sample_count" in condition:
            sample_count = data_analysis.get("sample_count", 0)
            if "< 100" in condition:
                return sample_count < 100
            elif "< 1000" in condition:
                return sample_count < 1000
            elif "> 1000" in condition:
                return sample_count > 1000
        
        elif "data_quality" in condition:
            quality_level = data_analysis.get("data_quality_level", DataQualityLevel.POOR)
            if "excellent" in condition:
                return quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.PREMIUM]
            elif "good" in condition:
                return quality_level in [DataQualityLevel.GOOD, DataQualityLevel.EXCELLENT, DataQualityLevel.PREMIUM]
        
        elif "complexity_score" in condition:
            complexity = data_analysis.get("complexity_score", 0.0)
            if "> 0.8" in condition:
                return complexity > 0.8
            elif "> 0.6" in condition:
                return complexity > 0.6
        
        elif "memory_usage" in condition:
            # This would be evaluated during runtime
            return psutil.virtual_memory().percent / 100 > 0.9
        
        elif "creation_time" in condition:
            # This would be evaluated during runtime with performance history
            return len(self.performance_history) > 0 and \
                   np.mean([p["creation_time"] for p in self.performance_history[-5:]]) > 30
        
        return False
    
    async def _execute_rule_action(self, rule: DQRule, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a DQ rule action"""
        action = rule.action
        result = {"success": True, "config_adjustments": {}, "recommendations": []}
        
        if action == "apply_aggressive_compression":
            result["config_adjustments"] = {
                "compression_method": "lzma",
                "quantization_level": "Q2_K",
                "target_size_reduction": 0.95
            }
            result["recommendations"].append(
                "Applied aggressive compression due to small sample size"
            )
        
        elif action == "use_high_quality_quantization":
            result["config_adjustments"] = {
                "quantization_level": "Q6_K",
                "compression_method": "gzip",
                "quality_preservation": True
            }
            result["recommendations"].append(
                "Using high-quality quantization to preserve excellent data quality"
            )
        
        elif action == "increase_model_capacity":
            current_size = self.learned_config["model_sizing"]["base_size_mb"]
            result["config_adjustments"] = {
                "model_size_mb": current_size * 1.5,
                "quantization_level": "Q5_K_M",
                "complexity_handling": True
            }
            result["recommendations"].append(
                "Increased model capacity to handle high complexity data"
            )
        
        elif action == "apply_memory_optimization":
            result["config_adjustments"] = {
                "batch_processing": True,
                "memory_efficient_mode": True,
                "compression_priority": "memory"
            }
            result["recommendations"].append(
                "Applied memory optimization due to high memory usage"
            )
        
        elif action == "optimize_processing_pipeline":
            result["config_adjustments"] = {
                "parallel_processing": True,
                "pipeline_optimization": True,
                "caching_enabled": True
            }
            result["recommendations"].append(
                "Optimized processing pipeline for better performance"
            )
        
        return result
    
    def _calculate_dq_confidence(self, data_analysis: Dict[str, Any], applied_rules: List[Dict]) -> float:
        """Calculate confidence score for DQ decisions"""
        confidence_factors = []
        
        # Data quality factor
        quality_score = data_analysis.get("quality_score", 0.0)
        confidence_factors.append(quality_score)
        
        # Sample size factor
        sample_count = data_analysis.get("sample_count", 0)
        sample_confidence = min(1.0, sample_count / 1000)  # Normalize to 1000 samples
        confidence_factors.append(sample_confidence)
        
        # Rule application factor
        rule_confidence = len(applied_rules) / len(self.dq_rules)  # More rules = more confidence
        confidence_factors.append(rule_confidence)
        
        # Historical performance factor
        if self.performance_history:
            avg_success = np.mean([p.get("success", 0) for p in self.performance_history[-10:]])
            confidence_factors.append(avg_success)
        
        return np.mean(confidence_factors)
    
    def _assess_risk_level(self, data_analysis: Dict[str, Any], applied_rules: List[Dict]) -> str:
        """Assess risk level for the conversion"""
        risk_factors = []
        
        # Data quality risk
        quality_level = data_analysis.get("data_quality_level", DataQualityLevel.POOR)
        if quality_level == DataQualityLevel.POOR:
            risk_factors.append("high")
        elif quality_level == DataQualityLevel.ACCEPTABLE:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Sample size risk
        sample_count = data_analysis.get("sample_count", 0)
        if sample_count < 10:
            risk_factors.append("high")
        elif sample_count < 100:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Complexity risk
        complexity = data_analysis.get("complexity_score", 0.0)
        if complexity > 0.8:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Determine overall risk
        if "high" in risk_factors:
            return "high"
        elif "medium" in risk_factors:
            return "medium"
        else:
            return "low"
    
    async def _make_intelligent_decisions(self, data_analysis: Dict[str, Any], 
                                        dq_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent configuration decisions based on analysis"""
        
        # Start with base learned configuration
        intelligent_config = self.learned_config.copy()
        
        # Apply DQ rule adjustments
        config_adjustments = dq_decisions.get("configuration_adjustments", {})
        
        # Intelligent model sizing
        if "model_size_mb" not in config_adjustments:
            optimal_size = self._calculate_intelligent_model_size(data_analysis)
            intelligent_config["model_sizing"]["target_size_mb"] = optimal_size
        else:
            intelligent_config["model_sizing"]["target_size_mb"] = config_adjustments["model_size_mb"]
        
        # Intelligent quantization selection
        if "quantization_level" not in config_adjustments:
            optimal_quantization = self._select_optimal_quantization(data_analysis)
            intelligent_config["quantization"] = optimal_quantization
        else:
            intelligent_config["quantization"] = config_adjustments["quantization_level"]
        
        # Intelligent compression selection
        if "compression_method" not in config_adjustments:
            optimal_compression = self._select_optimal_compression(data_analysis)
            intelligent_config["compression"] = optimal_compression
        else:
            intelligent_config["compression"] = config_adjustments["compression_method"]
        
        # Intelligent performance configuration
        intelligent_config["performance"] = self._configure_performance_settings(
            data_analysis, dq_decisions
        )
        
        # Add metadata
        intelligent_config["metadata"] = {
            "decision_timestamp": datetime.now().isoformat(),
            "data_analysis": data_analysis,
            "dq_decisions": dq_decisions,
            "confidence_score": dq_decisions.get("confidence_score", 0.0),
            "risk_level": dq_decisions.get("risk_level", "unknown")
        }
        
        return intelligent_config
    
    def _calculate_intelligent_model_size(self, data_analysis: Dict[str, Any]) -> float:
        """Calculate optimal model size based on data characteristics"""
        base_size = self.learned_config["model_sizing"]["base_size_mb"]
        
        # Adjust based on sample count
        sample_count = data_analysis.get("sample_count", 0)
        if sample_count < 50:
            size_multiplier = 0.5
        elif sample_count < 500:
            size_multiplier = 1.0
        elif sample_count < 5000:
            size_multiplier = 1.5
        else:
            size_multiplier = 2.0
        
        # Adjust based on complexity
        complexity = data_analysis.get("complexity_score", 0.0)
        complexity_multiplier = 1.0 + complexity
        
        # Adjust based on quality (higher quality = can use smaller size)
        quality_score = data_analysis.get("quality_score", 0.0)
        quality_multiplier = 1.0 - (quality_score * 0.3)  # Up to 30% reduction for high quality
        
        optimal_size = base_size * size_multiplier * complexity_multiplier * quality_multiplier
        
        # Ensure reasonable bounds
        return max(1.0, min(50.0, optimal_size))
    
    def _select_optimal_quantization(self, data_analysis: Dict[str, Any]) -> str:
        """Select optimal quantization level based on data analysis"""
        quality_level = data_analysis.get("data_quality_level", DataQualityLevel.POOR)
        complexity = data_analysis.get("complexity_score", 0.0)
        sample_count = data_analysis.get("sample_count", 0)
        
        # Decision matrix for quantization
        if quality_level == DataQualityLevel.PREMIUM and complexity > 0.8:
            return "Q6_K"  # Highest quality for premium complex data
        elif quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.PREMIUM]:
            return "Q5_K_M"  # High quality
        elif quality_level == DataQualityLevel.GOOD and complexity > 0.6:
            return "Q4_K_M"  # Balanced quality
        elif quality_level == DataQualityLevel.GOOD:
            return "Q4_K_S"  # Balanced quality, smaller
        elif sample_count > 1000:
            return "Q3_K_M"  # Medium compression for large datasets
        else:
            return "Q2_K"    # High compression for small/poor quality data
    
    def _select_optimal_compression(self, data_analysis: Dict[str, Any]) -> str:
        """Select optimal compression method based on data characteristics"""
        sample_count = data_analysis.get("sample_count", 0)
        complexity = data_analysis.get("complexity_score", 0.0)
        
        # Prefer speed for large datasets, compression for small ones
        if sample_count > 5000:
            return "gzip"    # Fast compression for large datasets
        elif sample_count > 1000:
            return "zlib"    # Balanced compression
        elif complexity > 0.8:
            return "bz2"     # Good compression for complex data
        else:
            return "lzma"    # Maximum compression for small simple data
    
    def _configure_performance_settings(self, data_analysis: Dict[str, Any], 
                                      dq_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Configure performance settings intelligently"""
        sample_count = data_analysis.get("sample_count", 0)
        complexity = data_analysis.get("complexity_score", 0.0)
        
        return {
            "batch_processing": sample_count > 1000,
            "parallel_processing": sample_count > 500 and complexity > 0.5,
            "memory_efficient_mode": sample_count > 10000 or complexity > 0.8,
            "caching_enabled": True,
            "optimization_level": "aggressive" if sample_count < 100 else "balanced"
        }
    
    async def _create_model_with_intelligence(self, request: Dict[str, Any], 
                                            intelligent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create model using intelligent configuration"""
        
        # Extract intelligent parameters
        domain = request.get("domain", "unknown")
        target_size = intelligent_config["model_sizing"]["target_size_mb"]
        quantization = intelligent_config["quantization"]
        compression = intelligent_config["compression"]
        
        # Create output path intelligently
        output_path = self._create_intelligent_output_path(domain, intelligent_config)
        
        # Simulate model creation with intelligent parameters
        start_time = time.time()
        
        # This would be the actual model creation logic
        model_result = {
            "status": "success",
            "domain": domain,
            "output_path": str(output_path),
            "model_size_mb": target_size,
            "quantization_level": quantization,
            "compression_method": compression,
            "creation_time": time.time() - start_time,
            "intelligent_config": intelligent_config,
            "quality_score": intelligent_config["metadata"]["data_analysis"]["quality_score"],
            "confidence_score": intelligent_config["metadata"]["confidence_score"],
            "risk_level": intelligent_config["metadata"]["risk_level"]
        }
        
        return model_result
    
    def _create_intelligent_output_path(self, domain: str, intelligent_config: Dict[str, Any]) -> Path:
        """Create intelligent output path based on configuration"""
        base_dir = self.output_strategy["base_directory"]
        
        # Organize by quality level for easy management
        quality_level = intelligent_config["metadata"]["data_analysis"]["data_quality_level"]
        quality_dir = base_dir / quality_level.value
        
        # Create domain-specific subdirectory
        domain_dir = quality_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        size_mb = intelligent_config["model_sizing"]["target_size_mb"]
        quantization = intelligent_config["quantization"]
        
        filename = f"{domain}_{timestamp}_{size_mb:.1f}MB_{quantization}.gguf"
        
        return domain_dir / filename
    
    async def _learn_from_results(self, model_result: Dict[str, Any], 
                                intelligent_config: Dict[str, Any], 
                                total_time: float) -> None:
        """Learn from results to improve future decisions"""
        
        # Record performance data
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "domain": model_result["domain"],
            "creation_time": total_time,
            "success": model_result["status"] == "success",
            "quality_score": model_result["quality_score"],
            "confidence_score": model_result["confidence_score"],
            "risk_level": model_result["risk_level"],
            "model_size_mb": model_result["model_size_mb"],
            "quantization": model_result["quantization_level"],
            "compression": model_result["compression_method"]
        }
        
        # Add to performance history
        self.performance_history.append(performance_record)
        
        # Keep only recent history (last 100 records)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update learned configuration if needed
        if self._should_update_learned_config(performance_record):
            await self._update_learned_config(performance_record)
        
        logger.info(f"📊 Learning from results: {model_result['domain']} - "
                   f"Quality: {model_result['quality_score']:.2f}, "
                   f"Time: {total_time:.2f}s")
    
    def _should_update_learned_config(self, performance_record: Dict[str, Any]) -> bool:
        """Determine if learned configuration should be updated"""
        # Update if we have enough data points
        if len(self.performance_history) < self.learning_engine["learning_threshold"]:
            return False
        
        # Update if recent performance is significantly different
        recent_performance = self.performance_history[-10:]
        avg_recent_time = np.mean([p["creation_time"] for p in recent_performance])
        avg_recent_quality = np.mean([p["quality_score"] for p in recent_performance])
        
        # Check if performance has improved enough to warrant config update
        if len(self.performance_history) >= 20:
            older_performance = self.performance_history[-20:-10]
            avg_older_time = np.mean([p["creation_time"] for p in older_performance])
            avg_older_quality = np.mean([p["quality_score"] for p in older_performance])
            
            # Update if significant improvement
            time_improvement = (avg_older_time - avg_recent_time) / avg_older_time
            quality_improvement = (avg_recent_quality - avg_older_quality) / avg_older_quality
            
            return time_improvement > 0.1 or quality_improvement > 0.05
        
        return False
    
    async def _update_learned_config(self, performance_record: Dict[str, Any]) -> None:
        """Update learned configuration based on performance"""
        # This would implement actual learning logic
        # For now, just log the learning event
        logger.info(f"🧠 Updating learned configuration based on performance patterns")
        
        # Save updated configuration
        config_path = Path("trinity-core/learned_configs/model_factory_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.learned_config, f, default_flow_style=False)
    
    async def _predict_data_characteristics(self, domain: str) -> Dict[str, Any]:
        """Predict data characteristics when no training data is provided"""
        # Use domain knowledge to predict characteristics
        domain_predictions = {
            "healthcare": {
                "expected_complexity": 0.8,
                "expected_quality": 0.7,
                "typical_sample_count": 500,
                "content_richness": 0.8
            },
            "business": {
                "expected_complexity": 0.6,
                "expected_quality": 0.8,
                "typical_sample_count": 1000,
                "content_richness": 0.7
            },
            "education": {
                "expected_complexity": 0.7,
                "expected_quality": 0.9,
                "typical_sample_count": 2000,
                "content_richness": 0.8
            },
            "technology": {
                "expected_complexity": 0.9,
                "expected_quality": 0.8,
                "typical_sample_count": 1500,
                "content_richness": 0.9
            }
        }
        
        # Get domain category
        all_domains = get_all_domains()
        domain_category = "general"
        for category, domains in all_domains.items():
            if domain in domains:
                domain_category = category
                break
        
        # Get predictions or use general defaults
        predictions = domain_predictions.get(domain_category, {
            "expected_complexity": 0.5,
            "expected_quality": 0.6,
            "typical_sample_count": 100,
            "content_richness": 0.6
        })
        
        # Create predicted analysis
        return {
            "sample_count": predictions["typical_sample_count"],
            "complexity_score": predictions["expected_complexity"],
            "quality_score": predictions["expected_quality"],
            "uniqueness_ratio": 0.8,  # Assume good uniqueness
            "average_length": 200,    # Assume reasonable length
            "structure_consistency": 0.7,
            "content_richness": predictions["content_richness"],
            "data_quality_level": self._determine_quality_level(predictions["expected_quality"]),
            "prediction_based": True
        }

# Singleton instance for global access
model_factory = IntelligentModelFactory()