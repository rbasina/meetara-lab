#!/usr/bin/env python3
"""
Test Multi-Base Model Implementation
Validates the enhanced architecture with 7 base models and quantization
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from trinity_core.config_manager import SmartTrinityConfigManager
from trinity_core.agents.model_factory import EnhancedModelFactory, ArchitectureType

async def test_multi_base_implementation():
    """Test the multi-base model implementation"""
    
    print("🧪 Testing Multi-Base Model Implementation")
    print("=" * 50)
    
    # Initialize components
    config_manager = SmartTrinityConfigManager()
    model_factory = EnhancedModelFactory()
    
    # Test 1: Configuration loading
    print("\n1️⃣ Testing Configuration Loading")
    multi_base_models = config_manager.get_multi_base_models()
    print(f"   → Multi-base models loaded: {len(multi_base_models)}")
    
    for tier, model in multi_base_models.items():
        print(f"   → {tier}: {model.model_path} ({model.parameters})")
    
    # Test 2: Architecture configurations
    print("\n2️⃣ Testing Architecture Configurations")
    arch_full = config_manager.get_universal_architecture("A_universal_full")
    arch_lite = config_manager.get_universal_architecture("B_universal_lite")
    
    if arch_full:
        print(f"   → A_universal_full: {arch_full.target_size:.2f}GB, {arch_full.quantization}")
    if arch_lite:
        print(f"   → B_universal_lite: {arch_lite.target_size:.3f}GB, {arch_lite.quantization}")
    
    # Test 3: Domain-to-model mapping
    print("\n3️⃣ Testing Domain-to-Model Mapping")
    test_domains = ["general_health", "programming", "writing", "parenting"]
    
    for domain in test_domains:
        base_model, quantization = config_manager.get_base_model_for_domain_with_quantization(
            domain, "A_universal_full"
        )
        category = config_manager.get_category_for_domain(domain)
        print(f"   → {domain} ({category}): {base_model} with {quantization}")
    
    # Test 4: Model factory status
    print("\n4️⃣ Testing Model Factory Status")
    status = await model_factory.get_multi_base_model_status()
    print(f"   → Total models: {status['multi_base_models']['total_models']}")
    print(f"   → Intelligence level: {status['intelligence_level']}")
    print(f"   → Trinity status: {status['trinity_status']}")
    
    # Test 5: Create a test model
    print("\n5️⃣ Testing Model Creation")
    test_request = {
        "domain": "general_health",
        "category": "healthcare",
        "architecture_type": "A_universal_full"
    }
    
    result = await model_factory.create_multi_base_model(test_request)
    
    if result.get("success"):
        print(f"   ✅ Model created successfully")
        print(f"   → Base model: {result['multi_base_metadata']['base_model']}")
        print(f"   → Quantization: {result['multi_base_metadata']['quantization']}")
        print(f"   → Target size: {result['performance_metrics']['target_size_gb']:.2f}GB")
        print(f"   → Trinity enhanced: {result['trinity_status']['multi_base_integration']}")
    else:
        print(f"   ❌ Model creation failed: {result.get('error')}")
    
    # Test 6: All base models for architecture
    print("\n6️⃣ Testing All Base Models for Architecture")
    all_models = config_manager.get_all_base_models_for_architecture("A_universal_full")
    print(f"   → Models for A_universal_full:")
    for tier, model_path, quantization in all_models:
        print(f"     • {tier}: {model_path} ({quantization})")
    
    print("\n✅ Multi-Base Model Implementation Test Complete!")
    return True

if __name__ == "__main__":
    asyncio.run(test_multi_base_implementation()) 