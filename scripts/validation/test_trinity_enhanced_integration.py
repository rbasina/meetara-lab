#!/usr/bin/env python3
"""
🚀 Trinity Enhanced Integration Test
Tests the complete integration of Enhanced Model Factory + Speech Models Factory

🎯 WHAT THIS TESTS:
✅ Enhanced Model Factory with multi-base model support
✅ Speech Models Factory with all 7 voice categories
✅ Auto-coordination between both super agents
✅ A_universal_full + B_universal_lite + 62 domains
✅ Smart quantization (hybrid Q2_K/Q4_K_M)
✅ Complete speech models bundle creation
✅ Trinity Architecture enhancements
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add trinity_core to path
sys.path.append(str(Path(__file__).parent.parent / "trinity_core"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_model_factory():
    """Test the enhanced model factory with multi-base model support"""
    logger.info("🧪 Testing Enhanced Model Factory...")
    
    try:
        from trinity_core.agents.model_factory import IntelligentModelFactory
        
        # Initialize enhanced model factory
        model_factory = IntelligentModelFactory()
        
        # Test multi-base model creation
        request = {
            "domain": "healthcare",
            "category": "healthcare", 
            "architecture_type": "A_universal_full",
            "include_62_domains": True,
            "quality_target": 0.95
        }
        
        logger.info("🏭 Creating multi-base model with 62 domains...")
        result = await model_factory.create_multi_base_model(request)
        
        if result.get("success"):
            logger.info("✅ Enhanced Model Factory test PASSED")
            logger.info(f"   → Architecture: {result.get('multi_base_model_spec', {}).get('architecture_type')}")
            logger.info(f"   → Quantization: {result.get('multi_base_model_spec', {}).get('quantization_strategy')}")
            logger.info(f"   → Size: {result.get('multi_base_model_spec', {}).get('target_size_gb')}GB")
            logger.info(f"   → Domains: {result.get('multi_base_model_spec', {}).get('domains_included')}")
            return True
        else:
            logger.error(f"❌ Enhanced Model Factory test FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Model Factory test ERROR: {e}")
        return False

async def test_speech_models_factory():
    """Test the speech models factory"""
    logger.info("🧪 Testing Speech Models Factory...")
    
    try:
        from agents.02_super_agents.04_speech_models_factory import SpeechModelsFactory
        
        # Initialize speech models factory
        speech_factory = SpeechModelsFactory()
        
        # Test speech models creation
        request = {
            "domain": "healthcare",
            "category": "healthcare",
            "output_path": "models/test_speech_models",
            "create_all_voices": True,
            "trinity_enhanced": True,
            "tara_compatible": True
        }
        
        logger.info("🎤 Creating complete speech models bundle...")
        result = await speech_factory.create_speech_models(request)
        
        if result.get("success"):
            logger.info("✅ Speech Models Factory test PASSED")
            logger.info(f"   → Total files: {result.get('total_files_created')}")
            logger.info(f"   → Emotion models: {result.get('models_summary', {}).get('emotion_models')}")
            logger.info(f"   → Voice profiles: {result.get('models_summary', {}).get('voice_profiles')}")
            logger.info(f"   → Routing models: {result.get('models_summary', {}).get('routing_models')}")
            logger.info(f"   → Creation time: {result.get('creation_time'):.2f}s")
            return True
        else:
            logger.error(f"❌ Speech Models Factory test FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Speech Models Factory test ERROR: {e}")
        return False

async def test_trinity_orchestrator_coordination():
    """Test the Trinity Orchestrator coordination"""
    logger.info("🧪 Testing Trinity Orchestrator Coordination...")
    
    try:
        from agents.10_trinity_orchestrator_master import TrinityOrchestratorMaster
        
        # Initialize Trinity Orchestrator
        orchestrator = TrinityOrchestratorMaster()
        
        # Initialize Trinity Architecture
        logger.info("🔱 Initializing Trinity Architecture...")
        init_success = await orchestrator.initialize_trinity_architecture()
        
        if not init_success:
            logger.error("❌ Trinity Architecture initialization failed")
            return False
        
        # Test coordinated model creation
        request = {
            "domain": "education",
            "category": "education",
            "architecture_type": "B_universal_lite",
            "include_62_domains": True,
            "create_speech_models": True,
            "quality_target": 0.92
        }
        
        logger.info("🔱 Testing coordinated model creation...")
        result = await orchestrator.create_complete_intelligent_model(request)
        
        if result.get("success"):
            logger.info("✅ Trinity Orchestrator Coordination test PASSED")
            logger.info(f"   → GGUF Success: {result.get('gguf_model', {}).get('success')}")
            logger.info(f"   → Speech Success: {result.get('speech_models', {}).get('success')}")
            logger.info(f"   → Architecture: {result.get('gguf_model', {}).get('architecture_type')}")
            logger.info(f"   → Domains: {result.get('gguf_model', {}).get('domains_included')}")
            logger.info(f"   → Speech Files: {result.get('speech_models', {}).get('total_files')}")
            logger.info(f"   → Auto-coordinated: {result.get('auto_coordinated')}")
            logger.info(f"   → Total time: {result.get('creation_time'):.2f}s")
            return True
        else:
            logger.error(f"❌ Trinity Orchestrator Coordination test FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Trinity Orchestrator Coordination test ERROR: {e}")
        return False

async def test_architecture_selection():
    """Test intelligent architecture selection"""
    logger.info("🧪 Testing Intelligent Architecture Selection...")
    
    try:
        from agents.02_super_agents.03_model_factory import IntelligentModelFactory
        
        model_factory = IntelligentModelFactory()
        
        # Test different domain priorities
        test_cases = [
            {"domain": "healthcare", "expected": "A_universal_full"},
            {"domain": "specialized", "expected": "A_universal_full"},
            {"domain": "business", "expected": "A_universal_full"},
            {"domain": "education", "expected": "B_universal_lite"},
            {"domain": "creative", "expected": "B_universal_lite"},
            {"domain": "daily_life", "expected": "B_universal_lite"}
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            domain = test_case["domain"]
            expected = test_case["expected"]
            
            # Test architecture selection
            selected = await model_factory._select_optimal_architecture(domain, domain, None, {})
            
            if selected.value == expected:
                logger.info(f"   ✅ {domain} → {selected.value} (correct)")
            else:
                logger.error(f"   ❌ {domain} → {selected.value} (expected {expected})")
                all_passed = False
        
        if all_passed:
            logger.info("✅ Architecture Selection test PASSED")
            return True
        else:
            logger.error("❌ Architecture Selection test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Architecture Selection test ERROR: {e}")
        return False

async def test_quantization_strategy():
    """Test smart quantization strategy"""
    logger.info("🧪 Testing Smart Quantization Strategy...")
    
    try:
        from agents.02_super_agents.03_model_factory import IntelligentModelFactory, ArchitectureType
        
        model_factory = IntelligentModelFactory()
        
        # Test quantization for different architectures
        test_cases = [
            {"arch": ArchitectureType.A_UNIVERSAL_FULL, "component": "base_models", "expected": "Q2_K"},
            {"arch": ArchitectureType.A_UNIVERSAL_FULL, "component": "domain_models", "expected": "Q4_K_M"},
            {"arch": ArchitectureType.B_UNIVERSAL_LITE, "component": "everything", "expected": "Q4_K_M"},
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            arch = test_case["arch"]
            component = test_case["component"]
            expected = test_case["expected"]
            
            # Test quantization selection
            selected = model_factory._get_quantization_for_component(arch, component)
            
            if selected == expected:
                logger.info(f"   ✅ {arch.value} + {component} → {selected} (correct)")
            else:
                logger.error(f"   ❌ {arch.value} + {component} → {selected} (expected {expected})")
                all_passed = False
        
        if all_passed:
            logger.info("✅ Quantization Strategy test PASSED")
            return True
        else:
            logger.error("❌ Quantization Strategy test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quantization Strategy test ERROR: {e}")
        return False

async def main():
    """Run all Trinity enhanced integration tests"""
    logger.info("🚀 Starting Trinity Enhanced Integration Tests")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # Run all tests
    tests = [
        ("Enhanced Model Factory", test_enhanced_model_factory),
        ("Speech Models Factory", test_speech_models_factory),
        ("Architecture Selection", test_architecture_selection),
        ("Quantization Strategy", test_quantization_strategy),
        ("Trinity Orchestrator Coordination", test_trinity_orchestrator_coordination),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 TRINITY ENHANCED INTEGRATION TEST RESULTS")
    logger.info("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"   → Tests Passed: {passed}/{total}")
    logger.info(f"   → Success Rate: {success_rate:.1f}%")
    logger.info(f"   → Total Time: {total_time:.2f}s")
    
    if success_rate == 100:
        logger.info("🎉 ALL TESTS PASSED - Trinity Enhanced Integration is working perfectly!")
    elif success_rate >= 80:
        logger.info("⚠️ MOSTLY WORKING - Some minor issues to address")
    else:
        logger.info("❌ SIGNIFICANT ISSUES - Integration needs work")
    
    logger.info("=" * 80)
    
    return success_rate == 100

if __name__ == "__main__":
    asyncio.run(main()) 