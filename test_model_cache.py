#!/usr/bin/env python3
"""
Test script to verify model caching is working properly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from trinity_core.agents.model_factory import get_model_factory_singleton
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

async def test_model_caching():
    """Test that the same base model is not downloaded multiple times"""
    print("🧪 Testing Model Caching...")
    
    # Get the singleton model factory
    config_manager = SmartTrinityConfigManager()
    model_factory = get_model_factory_singleton(config_manager)
    
    # Test domains that should use the same base model
    test_domains = [
        "project_management",  # Should use Qwen2.5-14B-Instruct
        "financial_planning",  # Should use Qwen2.5-14B-Instruct  
        "marketing_strategy",  # Should use Qwen2.5-14B-Instruct
    ]
    
    print(f"📋 Testing {len(test_domains)} domains with same base model...")
    
    for i, domain in enumerate(test_domains, 1):
        print(f"\n🔍 Test {i}/{len(test_domains)}: {domain}")
        
        # Create request
        request = {
            "domain": domain,
            "category": "business",
            "training_data": [],
            "simulation": True,  # Use simulation to avoid actual training
            "target_size_mb": 8.3,
            "environment": "dev"
        }
        
        # Process the domain
        result = await model_factory.create_intelligent_model(request)
        
        if result.get("error"):
            print(f"❌ Error for {domain}: {result['error']}")
        else:
            print(f"✅ Success for {domain}")
            print(f"   → Base model: {result.get('base_model')}")
            print(f"   → Quality score: {result.get('simulated_quality_score', 0):.2f}")
    
    # Log final cache status
    print(f"\n📊 Final Cache Status:")
    cache_status = model_factory.get_cache_status()
    print(f"   → Models cached: {cache_status['model_cache_size']}")
    print(f"   → Tokenizers cached: {cache_status['tokenizer_cache_size']}")
    if cache_status['cached_models']:
        print(f"   → Cached models: {', '.join(cache_status['cached_models'])}")
    
    print("\n✅ Model caching test completed!")

if __name__ == "__main__":
    asyncio.run(test_model_caching()) 