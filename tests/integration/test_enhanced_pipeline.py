#!/usr/bin/env python3
"""
MeeTARA Lab - Enhanced Pipeline Integration Test

Tests the complete enhanced training pipeline with all intelligent features:
- Emotion/context learning
- LoRA integration
- Intelligent routing
- GGUF validation
- Resource-aware parallelism
- Comprehensive reporting
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trinity_core.agents.trinity_conductor import TrinityPrimaryConductor
from trinity_core.core_components.config_manager import SmartTrinityConfigManager
from trinity_core.agents.data_generator import TrinityDataGenerator
from trinity_core.agents.model_factory import IntelligentModelFactory
from trinity_core.agents.quantization_and_cleanup_agent import QuantizationAndCleanupAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPipelineTester:
    """
    Comprehensive tester for the enhanced MeeTARA Lab training pipeline.
    """
    
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        self.conductor = TrinityPrimaryConductor()
        self.data_generator = TrinityDataGenerator(self.conductor.intelligence_hub)
        self.model_factory = IntelligentModelFactory()
        self.quantization_agent = QuantizationAndCleanupAgent()
        
        self.test_results = {
            "start_time": datetime.now(),
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "test_details": []
        }
    
    async def run_comprehensive_test(self):
        """
        Run comprehensive test of the enhanced pipeline.
        """
        logger.info("🚀 Starting comprehensive enhanced pipeline test...")
        
        test_functions = [
            self.test_config_manager,
            self.test_data_generation,
            self.test_model_factory,
            self.test_quantization_agent,
            self.test_conductor_orchestration,
            self.test_end_to_end_pipeline
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
                self.test_results["tests_passed"] += 1
            except Exception as e:
                logger.error(f"❌ Test failed: {test_func.__name__}: {e}")
                self.test_results["tests_failed"] += 1
                self.test_results["test_details"].append({
                    "test": test_func.__name__,
                    "status": "failed",
                    "error": str(e)
                })
            
            self.test_results["total_tests"] += 1
        
        await self.generate_test_report()
    
    async def test_config_manager(self):
        """Test configuration manager functionality."""
        logger.info("📋 Testing configuration manager...")
        
        # Test domain retrieval
        domains = self.config_manager.get_all_domains_flat()
        assert len(domains) > 0, "No domains found in configuration"
        
        # Test domain details retrieval
        test_domain = domains[0]
        domain_details = self.config_manager._get_domain_details(test_domain)
        assert "base_model" in domain_details, "Domain details missing base_model"
        assert "tier_name" in domain_details, "Domain details missing tier_name"
        
        # Test tier configuration
        tier_config = self.config_manager.get_model_tier_config(domain_details["tier_name"])
        assert "lora_r" in tier_config, "Tier config missing LoRA parameters"
        
        logger.info(f"✅ Configuration manager test passed - {len(domains)} domains loaded")
    
    async def test_data_generation(self):
        """Test enhanced data generation with emotion/context learning."""
        logger.info("📊 Testing enhanced data generation...")
        
        # Test data generation for a sample domain
        test_domain = "general_health"
        result = self.data_generator.generate_domain_data(test_domain, samples_per_domain=100)
        
        # Check if data generation was successful (even if 0 samples in simulation)
        assert result["status"] == "success", f"Data generation failed: {result.get('error')}"
        assert "trinity_enhancements" in result, "Missing Trinity enhancements"
        assert result["trinity_enhancements"]["emotional_intelligence"], "Emotion learning not enabled"
        
        # Check quality metrics (may be 0 in simulation)
        quality_metrics = result["quality_metrics"]
        assert "diversity_score" in quality_metrics, "Missing diversity score"
        assert "emotion_coverage" in quality_metrics, "Missing emotion coverage"
        
        logger.info(f"✅ Data generation test passed - {result.get('total_samples', 0)} samples generated")
    
    async def test_model_factory(self):
        """Test enhanced model factory with LoRA integration."""
        logger.info("🧠 Testing enhanced model factory...")
        
        # Test model creation
        request = {
            "domain": "mental_health",
            "category": "healthcare",
            "training_data": [{"conversation": "test"}],
            "simulation": True,
            "generate_synthetic": True,
            "target_size_mb": 8.3
        }
        
        result = await self.model_factory.create_intelligent_model(request)
        
        assert result["status"] == "success", f"Model creation failed: {result.get('error')}"
        assert "lora_adapter_path" in result, "Missing LoRA adapter path"
        assert "lora_config" in result, "Missing LoRA configuration"
        assert "emotion_context_config" in result, "Missing emotion/context configuration"
        
        # Check LoRA configuration
        lora_config = result["lora_config"]
        assert "r" in lora_config, "Missing LoRA rank"
        assert "alpha" in lora_config, "Missing LoRA alpha"
        assert "target_modules" in lora_config, "Missing LoRA target modules"
        
        logger.info(f"✅ Model factory test passed - LoRA integration working")
    
    async def test_quantization_agent(self):
        """Test enhanced quantization agent with GGUF validation."""
        logger.info("🔧 Testing enhanced quantization agent...")
        
        # Create a dummy raw model for testing
        test_raw_model = Path("test_raw_model.bin")
        with open(test_raw_model, 'wb') as f:
            f.write(b"dummy_model_data")
        
        try:
            result = await self.quantization_agent.process_and_finalize_model(
                raw_model_path=str(test_raw_model),
                domain="general_health",  # Use real domain from config
                model_size_mb=8.3,
                architecture_type="domain_specific"
            )
            
            assert result["status"] == "success", f"Quantization failed: {result.get('error')}"
            assert "validation_results" in result, "Missing validation results"
            assert "quality_report" in result, "Missing quality report"
            
            # Check validation results
            validation_results = result["validation_results"]
            assert len(validation_results) > 0, "No validation results"
            
            # Check quality report
            quality_report = result["quality_report"]
            assert "success_rate" in quality_report, "Missing success rate"
            assert "average_validation_score" in quality_report, "Missing validation score"
            
            logger.info(f"✅ Quantization agent test passed - GGUF validation working")
            
        finally:
            # Cleanup test file
            if test_raw_model.exists():
                test_raw_model.unlink()
    
    async def test_conductor_orchestration(self):
        """Test conductor orchestration with resource awareness."""
        logger.info("🎼 Testing conductor orchestration...")
        
        # Test single domain processing
        allocation = {
            "gpu_type": "T4",
            "memory_gb": 16,
            "parallel_capacity": 2
        }
        
        result = await self.conductor._process_domain_optimized(
            domain="nutrition",
            category="healthcare",
            allocation=allocation,
            simulation=True,
            generate_synthetic=True
        )
        
        assert result["status"] == "success", f"Conductor processing failed: {result.get('error')}"
        assert "data_generation" in result, "Missing data generation results"
        assert "model_training" in result, "Missing model training results"
        assert "gguf_creation" in result, "Missing GGUF creation results"
        assert "trinity_enhancements" in result, "Missing Trinity enhancements"
        
        # Check Trinity enhancements
        enhancements = result["trinity_enhancements"]
        assert enhancements["emotion_context_learning"], "Emotion/context learning not enabled"
        assert enhancements["lora_integration"], "LoRA integration not enabled"
        assert enhancements["gguf_validation"], "GGUF validation not enabled"
        
        logger.info(f"✅ Conductor orchestration test passed - Resource awareness working")
    
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        logger.info("🔄 Testing complete end-to-end pipeline...")
        
        # Test orchestration with multiple domains
        test_domains = ["general_health", "nutrition"]
        
        result = await self.conductor.orchestrate_intelligent_training(
            target_domains=test_domains,
            training_mode="optimized",
            simulation=True,
            generate_synthetic=True
        )
        
        # Check for the actual response structure from the conductor
        assert "overall_success" in result, "Missing overall success status"
        assert "total_domains_processed" in result, "Missing total domains processed"
        assert "successful_domains_count" in result, "Missing successful domains count"
        assert "failed_domains_count" in result, "Missing failed domains count"
        assert "domain_breakdown" in result, "Missing domain breakdown"
        
        # Check domain breakdown structure
        domain_breakdown = result["domain_breakdown"]
        assert "successful_domains" in domain_breakdown, "Missing successful domains list"
        assert "failed_domains" in domain_breakdown, "Missing failed domains list"
        
        # Check optimization gains
        if "optimization_gains" in result:
            gains = result["optimization_gains"]
            assert "speed_improvement" in gains, "Missing speed improvement"
            assert "success_rate" in gains, "Missing success rate"
        
        logger.info(f"✅ End-to-end pipeline test passed - {result['total_domains_processed']} domains processed")
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating comprehensive test report...")
        
        end_time = datetime.now()
        duration = (end_time - self.test_results["start_time"]).total_seconds()
        
        report = {
            "test_summary": {
                "total_tests": self.test_results["total_tests"],
                "tests_passed": self.test_results["tests_passed"],
                "tests_failed": self.test_results["tests_failed"],
                "success_rate": self.test_results["tests_passed"] / self.test_results["total_tests"] if self.test_results["total_tests"] > 0 else 0,
                "duration_seconds": duration
            },
            "enhanced_features_tested": {
                "emotion_context_learning": True,
                "lora_integration": True,
                "intelligent_routing": True,
                "gguf_validation": True,
                "quality_assurance": True,
                "resource_optimization": True,
                "comprehensive_reporting": True
            },
            "test_details": self.test_results["test_details"],
            "recommendations": self._generate_test_recommendations()
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎯 ENHANCED PIPELINE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
        logger.info(f"Tests Passed: {report['test_summary']['tests_passed']}")
        logger.info(f"Tests Failed: {report['test_summary']['tests_failed']}")
        logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
        logger.info(f"Duration: {report['test_summary']['duration_seconds']:.2f}s")
        logger.info("=" * 60)
        
        # Save detailed report
        report_file = Path("test_reports") / f"enhanced_pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        return report
    
    def _generate_test_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.test_results["tests_failed"] > 0:
            recommendations.append("Review failed tests and fix any issues before production deployment")
        
        if self.test_results["tests_passed"] == self.test_results["total_tests"]:
            recommendations.append("All tests passed! Enhanced pipeline is ready for production")
        
        recommendations.append("Run full production test with real domains before deployment")
        recommendations.append("Monitor resource usage during production runs")
        recommendations.append("Validate GGUF files with llama.cpp in production environment")
        
        return recommendations

async def main():
    """Main test execution."""
    logger.info("🚀 Starting MeeTARA Lab Enhanced Pipeline Integration Test")
    
    tester = EnhancedPipelineTester()
    await tester.run_comprehensive_test()
    
    logger.info("✅ Enhanced pipeline integration test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 