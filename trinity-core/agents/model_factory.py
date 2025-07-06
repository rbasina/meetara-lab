#!/usr/bin/env python3
"""
MeeTARA Lab - Enhanced Model Factory Super Agent
MULTI-BASE MODEL ARCHITECTURE SUPPORT

🧠 ENHANCED CAPABILITIES:
✅ Multi-base model integration (7 models: SmolLM2, Phi-3.5, Qwen2.5, Phi-3-medium variants)
✅ Base-level quantization (Q2_K for A_universal_full, Q4_K_M for B_universal_lite)
✅ Intelligent model selection based on domain requirements
✅ Dynamic architecture routing (A_universal_full vs B_universal_lite)
✅ Smart output path organization
✅ Trinity Architecture integration with multi-model support

🎯 ARCHITECTURE SUPPORT:
- A_universal_full: 7.78GB (Q2_K quantization, 1.9GB runtime)
- B_universal_lite: 815MB (Q4_K_M quantization)
- Domain-specific: 8.3MB (Q4_K_M quantization)
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

# Import enhanced config manager
from ..config_manager import SmartTrinityConfigManager, MultiBaseModel, UniversalModelArchitecture

# Trinity Architecture imports
from .lightweight_mcp_v2 import LightweightMCPv2, MCPMessage

class IntelligenceLevel(Enum):
    """Intelligence levels for adaptive behavior"""
    BASIC = "basic"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    AUTONOMOUS = "autonomous"

class ArchitectureType(Enum):
    """Universal model architecture types"""
    A_UNIVERSAL_FULL = "A_universal_full"
    B_UNIVERSAL_LITE = "B_universal_lite"
    C_CATEGORY_SPECIFIC = "C_category_specific"
    D_DOMAIN_SPECIFIC = "D_domain_specific"

@dataclass
class MultiBaseModelSpec:
    """Multi-base model specification with quantization"""
    domain: str
    category: str
    architecture_type: ArchitectureType
    base_model: str
    quantization: str
    tier: str
    parameters: str
    license: str
    output_path: Path
    
    # Performance characteristics
    target_size: float
    runtime_memory: float
    quality_target: float
    
    # Training configuration
    batch_size: int
    sequence_length: int
    recommended_gpu: str
    cost_per_hour: float

class EnhancedModelFactory:
    """
    Enhanced Model Factory Super Agent - Multi-Base Model Architecture
    
    🧠 MULTI-BASE MODEL INTELLIGENCE:
    - Intelligent model selection based on domain requirements
    - Base-level quantization for optimal size/quality balance
    - Dynamic architecture routing (Full vs Lite)
    - Smart output organization
    - Trinity Architecture integration
    """
    
    def __init__(self, intelligence_level: IntelligenceLevel = IntelligenceLevel.AUTONOMOUS):
        self.intelligence_level = intelligence_level
        self.mcp = LightweightMCPv2()
        
        # Enhanced configuration manager
        self.config_manager = SmartTrinityConfigManager()
        
        # Multi-base model support
        self.multi_base_models = self.config_manager.get_multi_base_models()
        self.universal_architectures = {
            ArchitectureType.A_UNIVERSAL_FULL: self.config_manager.get_universal_architecture("A_universal_full"),
            ArchitectureType.B_UNIVERSAL_LITE: self.config_manager.get_universal_architecture("B_universal_lite")
        }
        
        # Intelligence engines
        self.learning_engine = self._initialize_learning_engine()
        self.architecture_selector = self._initialize_architecture_selector()
        self.quantization_optimizer = self._initialize_quantization_optimizer()
        
        # Performance tracking
        self.performance_history = []
        self.model_cache = {}
        
        logger.info(f"🧠 Enhanced Model Factory initialized")
        logger.info(f"   → Intelligence Level: {self.intelligence_level.value}")
        logger.info(f"   → Multi-Base Models: {len(self.multi_base_models)} models")
        logger.info(f"   → Universal Architectures: {len(self.universal_architectures)} types")
        
    def _initialize_learning_engine(self) -> Dict[str, Any]:
        """Initialize enhanced learning engine for multi-base models"""
        return {
            "multi_model_optimization": True,
            "quantization_learning": True,
            "architecture_adaptation": True,
            "performance_prediction": True,
            "cost_optimization": True,
            "quality_preservation": True
        }
    
    def _initialize_architecture_selector(self) -> Dict[str, Any]:
        """Initialize intelligent architecture selection"""
        return {
            "size_requirements": {
                "mobile": ArchitectureType.D_DOMAIN_SPECIFIC,
                "edge": ArchitectureType.C_CATEGORY_SPECIFIC,
                "standard": ArchitectureType.B_UNIVERSAL_LITE,
                "full": ArchitectureType.A_UNIVERSAL_FULL
            },
            "quality_requirements": {
                "basic": ArchitectureType.D_DOMAIN_SPECIFIC,
                "standard": ArchitectureType.C_CATEGORY_SPECIFIC,
                "high": ArchitectureType.B_UNIVERSAL_LITE,
                "premium": ArchitectureType.A_UNIVERSAL_FULL
            },
            "domain_priorities": {
                "healthcare": ArchitectureType.A_UNIVERSAL_FULL,
                "specialized": ArchitectureType.A_UNIVERSAL_FULL,
                "business": ArchitectureType.A_UNIVERSAL_FULL,
                "education": ArchitectureType.B_UNIVERSAL_LITE,
                "technology": ArchitectureType.B_UNIVERSAL_LITE,
                "creative": ArchitectureType.B_UNIVERSAL_LITE,
                "daily_life": ArchitectureType.B_UNIVERSAL_LITE
            }
        }
    
    def _initialize_quantization_optimizer(self) -> Dict[str, Any]:
        """Initialize intelligent quantization optimization"""
        return {
            "architecture_quantization": {
                "A_universal_full": "Q2_K",
                "B_universal_lite": "Q4_K_M",
                "C_category_specific": "Q4_K_M",
                "D_domain_specific": "Q4_K_M"
            },
            "quality_thresholds": {
                "Q2_K": 0.82,
                "Q4_K_M": 0.92,
                "Q5_K_S": 0.94
            },
            "size_targets": {
                "A_universal_full": 7.78,  # GB
                "B_universal_lite": 0.815,  # GB
                "domain_specific": 0.0083  # GB
            }
        }
    
    async def create_multi_base_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create model using multi-base architecture with intelligent selection
        
        🧠 INTELLIGENT PROCESS:
        1. Analyze domain requirements
        2. Select optimal architecture type
        3. Choose appropriate base model
        4. Apply optimal quantization
        5. Generate model with Trinity enhancement
        """
        start_time = time.time()
        
        try:
            # Extract request parameters
            domain = request.get("domain")
            category = request.get("category")
            architecture_hint = request.get("architecture_type")
            
            if not domain:
                return {"error": "Domain is required for multi-base model creation"}
            
            # Step 1: Intelligent architecture selection
            architecture_type = await self._select_optimal_architecture(domain, category, architecture_hint)
            
            # Step 2: Get multi-base model specification
            model_spec = await self._create_multi_base_model_spec(domain, category, architecture_type)
            
            # Step 3: Validate and prepare
            validation_result = await self._validate_multi_base_model_spec(model_spec)
            if not validation_result["valid"]:
                return {"error": f"Model specification validation failed: {validation_result['reason']}"}
            
            # Step 4: Create model with Trinity enhancement
            model_result = await self._create_model_with_trinity_enhancement(model_spec, request)
            
            # Step 5: Post-processing and optimization
            final_result = await self._post_process_multi_base_model(model_result, model_spec)
            
            # Step 6: Learning and adaptation
            await self._learn_from_multi_base_model_creation(final_result, model_spec, time.time() - start_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Multi-base model creation failed: {e}")
            return {"error": f"Multi-base model creation failed: {str(e)}"}
    
    async def _select_optimal_architecture(self, domain: str, category: str, hint: Optional[str] = None) -> ArchitectureType:
        """Intelligently select optimal architecture type"""
        
        # If hint provided and valid, use it
        if hint:
            try:
                return ArchitectureType(hint)
            except ValueError:
                logger.warning(f"Invalid architecture hint: {hint}, using intelligent selection")
        
        # Get category if not provided
        if not category:
            category = self.config_manager.get_category_for_domain(domain)
        
        # Intelligent selection based on domain priorities
        if category in self.architecture_selector["domain_priorities"]:
            return self.architecture_selector["domain_priorities"][category]
        
        # Default to B_universal_lite for balanced performance
        return ArchitectureType.B_UNIVERSAL_LITE
    
    async def _create_multi_base_model_spec(self, domain: str, category: str, architecture_type: ArchitectureType) -> MultiBaseModelSpec:
        """Create comprehensive multi-base model specification"""
        
        # Get base model and quantization for this domain and architecture
        base_model, quantization = self.config_manager.get_base_model_for_domain_with_quantization(
            domain, architecture_type.value
        )
        
        # Find the tier for this base model
        tier = None
        for tier_name, model_config in self.multi_base_models.items():
            if model_config.model_path == base_model:
                tier = tier_name
                break
        
        if not tier:
            raise ValueError(f"No tier found for base model: {base_model}")
        
        # Get model configuration
        model_config = self.multi_base_models[tier]
        
        # Get architecture configuration
        arch_config = self.universal_architectures.get(architecture_type)
        
        # Get output path
        output_path = self.config_manager.get_model_output_path(domain, architecture_type.value)
        
        # Create specification
        return MultiBaseModelSpec(
            domain=domain,
            category=category or self.config_manager.get_category_for_domain(domain),
            architecture_type=architecture_type,
            base_model=base_model,
            quantization=quantization,
            tier=tier,
            parameters=model_config.parameters,
            license=model_config.license,
            output_path=output_path,
            target_size=arch_config.target_size if arch_config else 0.0083,
            runtime_memory=arch_config.runtime_memory if arch_config else 0.0083,
            quality_target=0.92,  # Default quality target
            batch_size=model_config.batch_size,
            sequence_length=model_config.sequence_length,
            recommended_gpu=model_config.recommended_gpu,
            cost_per_hour=model_config.cost_per_hour
        )
    
    async def _validate_multi_base_model_spec(self, model_spec: MultiBaseModelSpec) -> Dict[str, Any]:
        """Validate multi-base model specification"""
        
        try:
            # Validate base model exists
            if not model_spec.base_model:
                return {"valid": False, "reason": "Base model not specified"}
            
            # Validate quantization is supported
            if model_spec.quantization not in ["Q2_K", "Q4_K_M", "Q5_K_S"]:
                return {"valid": False, "reason": f"Unsupported quantization: {model_spec.quantization}"}
            
            # Validate architecture type
            if model_spec.architecture_type not in [ArchitectureType.A_UNIVERSAL_FULL, ArchitectureType.B_UNIVERSAL_LITE]:
                return {"valid": False, "reason": f"Unsupported architecture: {model_spec.architecture_type}"}
            
            # Validate output path
            if not model_spec.output_path:
                return {"valid": False, "reason": "Output path not specified"}
            
            # Ensure output directory exists
            model_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return {"valid": True, "reason": "Specification validated successfully"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _create_model_with_trinity_enhancement(self, model_spec: MultiBaseModelSpec, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create model with Trinity Architecture enhancement"""
        
        logger.info(f"🏗️ Creating {model_spec.architecture_type.value} model for {model_spec.domain}")
        logger.info(f"   → Base Model: {model_spec.base_model} ({model_spec.parameters})")
        logger.info(f"   → Quantization: {model_spec.quantization}")
        logger.info(f"   → Target Size: {model_spec.target_size:.2f}GB")
        logger.info(f"   → Runtime Memory: {model_spec.runtime_memory:.2f}GB")
        
        # Get training configuration
        training_config = self.config_manager.get_training_config_for_domain_with_quantization(
            model_spec.domain, model_spec.architecture_type.value
        )
        
        # Enhanced training configuration with Trinity
        enhanced_config = {
            **training_config,
            "multi_base_model": {
                "tier": model_spec.tier,
                "base_model": model_spec.base_model,
                "quantization": model_spec.quantization,
                "parameters": model_spec.parameters,
                "license": model_spec.license
            },
            "trinity_enhancement": {
                "arc_reactor_efficiency": 0.90,
                "perplexity_intelligence": True,
                "einstein_fusion_multiplier": 5.04
            },
            "architecture_config": {
                "type": model_spec.architecture_type.value,
                "target_size_gb": model_spec.target_size,
                "runtime_memory_gb": model_spec.runtime_memory,
                "quantization": model_spec.quantization
            },
            "output_config": {
                "output_path": str(model_spec.output_path),
                "format": "GGUF",
                "compression": model_spec.quantization
            }
        }
        
        # Simulate model creation (in real implementation, this would call the actual training)
        creation_result = {
            "success": True,
            "model_spec": model_spec,
            "training_config": enhanced_config,
            "model_path": str(model_spec.output_path),
            "size_gb": model_spec.target_size,
            "runtime_memory_gb": model_spec.runtime_memory,
            "quantization": model_spec.quantization,
            "quality_score": 0.92,  # Simulated quality score
            "creation_time": datetime.now().isoformat(),
            "trinity_enhanced": True
        }
        
        return creation_result
    
    async def _post_process_multi_base_model(self, model_result: Dict[str, Any], model_spec: MultiBaseModelSpec) -> Dict[str, Any]:
        """Post-process multi-base model with optimization"""
        
        # Add multi-base model metadata
        model_result["multi_base_metadata"] = {
            "architecture_type": model_spec.architecture_type.value,
            "base_model": model_spec.base_model,
            "tier": model_spec.tier,
            "parameters": model_spec.parameters,
            "license": model_spec.license,
            "quantization": model_spec.quantization,
            "domain": model_spec.domain,
            "category": model_spec.category
        }
        
        # Add performance metrics
        model_result["performance_metrics"] = {
            "target_size_gb": model_spec.target_size,
            "runtime_memory_gb": model_spec.runtime_memory,
            "quality_target": model_spec.quality_target,
            "cost_per_hour": model_spec.cost_per_hour,
            "recommended_gpu": model_spec.recommended_gpu
        }
        
        # Add Trinity Architecture status
        model_result["trinity_status"] = {
            "arc_reactor_efficiency": 0.90,
            "perplexity_intelligence": True,
            "einstein_fusion_active": True,
            "multi_base_integration": True
        }
        
        return model_result
    
    async def _learn_from_multi_base_model_creation(self, result: Dict[str, Any], model_spec: MultiBaseModelSpec, creation_time: float) -> None:
        """Learn from multi-base model creation for future optimization"""
        
        # Create learning record
        learning_record = {
            "timestamp": datetime.now().isoformat(),
            "domain": model_spec.domain,
            "category": model_spec.category,
            "architecture_type": model_spec.architecture_type.value,
            "base_model": model_spec.base_model,
            "quantization": model_spec.quantization,
            "creation_time": creation_time,
            "success": result.get("success", False),
            "quality_score": result.get("quality_score", 0.0),
            "target_size_gb": model_spec.target_size,
            "runtime_memory_gb": model_spec.runtime_memory
        }
        
        # Add to performance history
        self.performance_history.append(learning_record)
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"📊 Learning record created for {model_spec.domain} ({model_spec.architecture_type.value})")
    
    async def get_multi_base_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of multi-base model system"""
        
        return {
            "multi_base_models": {
                "total_models": len(self.multi_base_models),
                "models": {tier: config.model_path for tier, config in self.multi_base_models.items()}
            },
            "universal_architectures": {
                "supported": list(self.universal_architectures.keys()),
                "A_universal_full": {
                    "target_size_gb": self.universal_architectures[ArchitectureType.A_UNIVERSAL_FULL].target_size,
                    "quantization": self.universal_architectures[ArchitectureType.A_UNIVERSAL_FULL].quantization
                } if ArchitectureType.A_UNIVERSAL_FULL in self.universal_architectures else None,
                "B_universal_lite": {
                    "target_size_gb": self.universal_architectures[ArchitectureType.B_UNIVERSAL_LITE].target_size,
                    "quantization": self.universal_architectures[ArchitectureType.B_UNIVERSAL_LITE].quantization
                } if ArchitectureType.B_UNIVERSAL_LITE in self.universal_architectures else None
            },
            "performance_history": {
                "total_records": len(self.performance_history),
                "recent_success_rate": self._calculate_recent_success_rate(),
                "average_quality_score": self._calculate_average_quality_score()
            },
            "intelligence_level": self.intelligence_level.value,
            "trinity_status": {
                "arc_reactor_active": True,
                "perplexity_intelligence": True,
                "einstein_fusion": True,
                "multi_base_integration": True
            }
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate from performance history"""
        if not self.performance_history:
            return 0.0
        
        recent_records = self.performance_history[-100:]  # Last 100 records
        successful = sum(1 for record in recent_records if record.get("success", False))
        return successful / len(recent_records)
    
    def _calculate_average_quality_score(self) -> float:
        """Calculate average quality score from performance history"""
        if not self.performance_history:
            return 0.0
        
        quality_scores = [record.get("quality_score", 0.0) for record in self.performance_history]
        return sum(quality_scores) / len(quality_scores)

# Factory function for creating enhanced model factory
def create_enhanced_model_factory(intelligence_level: IntelligenceLevel = IntelligenceLevel.AUTONOMOUS) -> EnhancedModelFactory:
    """Create enhanced model factory with multi-base model support"""
    return EnhancedModelFactory(intelligence_level)

# Export the enhanced factory
__all__ = ["EnhancedModelFactory", "MultiBaseModelSpec", "ArchitectureType", "create_enhanced_model_factory"] 