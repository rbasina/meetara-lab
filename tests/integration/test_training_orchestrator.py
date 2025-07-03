#!/usr/bin/env python3
"""
Integration Tests for Training Orchestrator
Tests the cloud training orchestration with Trinity Architecture integration
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add cloud-training to path
sys.path.append('../../cloud-training')
sys.path.append('../../trinity-core')

from cloud_training.training_orchestrator import TrainingOrchestrator

class TestTrainingOrchestrator:
    """Integration tests for Training Orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance for testing"""
        orchestrator = TrainingOrchestrator()
        yield orchestrator
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'domain_categories')
        assert hasattr(orchestrator, 'cloud_providers')
        assert hasattr(orchestrator, 'cost_optimization')
        assert len(orchestrator.domain_categories) == 7
        assert len(orchestrator.cloud_providers) == 4
        
    @pytest.mark.asyncio
    async def test_orchestrator_statistics(self, orchestrator):
        """Test orchestrator statistics reporting"""
        stats = await orchestrator.get_orchestration_statistics()
        
        assert stats is not None
        assert "orchestration_ready" in stats
        assert "domain_categories_supported" in stats
        assert "cloud_providers_available" in stats
        assert "monthly_budget_remaining" in stats
        assert "trinity_orchestration_enabled" in stats
        
        assert stats["orchestration_ready"] is True
        assert stats["domain_categories_supported"] == 7
        assert stats["cloud_providers_available"] == 4
        assert stats["monthly_budget_remaining"] == 50.0
        assert stats["trinity_orchestration_enabled"] is True
        
    @pytest.mark.asyncio
    async def test_sample_domain_training(self, orchestrator):
        """Test training orchestration with sample domains"""
        test_domains = ["parenting", "general_health", "marketing"]
        
        result = await orchestrator.orchestrate_universal_training(
            target_domains=test_domains,
            training_mode="balanced"
        )
        
        assert result is not None
        assert "success" in result
        assert "total_domains" in result
        assert "successful_domains" in result
        assert "total_cost" in result
        assert "speed_improvement" in result
        
        if result.get("success", False):
            assert result["total_domains"] == len(test_domains)
            assert result["successful_domains"] <= result["total_domains"]
            assert isinstance(result["total_cost"], (int, float))
            assert result["speed_improvement"]  # Should have some speed improvement value
            
    @pytest.mark.asyncio
    async def test_cost_optimization(self, orchestrator):
        """Test cost optimization features"""
        # Check cost limits are set correctly
        assert orchestrator.cost_optimization["daily_limit"] == 5.0
        assert orchestrator.cost_optimization["monthly_target"] == 50.0
        assert orchestrator.cost_optimization["auto_shutdown"] is True
        
        # Test budget compliance
        stats = await orchestrator.get_orchestration_statistics()
        assert stats["monthly_budget_remaining"] <= 50.0
        
    @pytest.mark.asyncio
    async def test_cloud_provider_selection(self, orchestrator):
        """Test cloud provider selection logic"""
        # Create a mock training plan
        training_plan = {
            "estimated_total_cost": 10.0,
            "estimated_total_time": 5 * 60  # 5 hours in minutes
        }
        
        provider = await orchestrator._select_optimal_provider(training_plan)
        assert provider in ["google_colab", "vast_ai"]
        
        # Test with higher cost - should select spot instances
        training_plan["estimated_total_cost"] = 20.0
        provider = await orchestrator._select_optimal_provider(training_plan)
        assert provider in orchestrator.cloud_providers.keys()
        
    @pytest.mark.asyncio
    async def test_trinity_architecture_integration(self, orchestrator):
        """Test Trinity Architecture integration"""
        assert orchestrator.trinity_orchestration["arc_reactor_efficiency"] is True
        assert orchestrator.trinity_orchestration["perplexity_intelligence"] is True
        assert orchestrator.trinity_orchestration["einstein_fusion"] is True
        
        # Test Trinity coordination enhancement
        mock_results = {
            "completed_domains": ["test_domain"],
            "failed_domains": [],
            "total_cost": 5.0,
            "total_time": 100,
            "speed_improvement": "50x",
            "average_quality": 101
        }
        
        enhanced_results = await orchestrator._apply_trinity_coordination(mock_results)
        assert enhanced_results["arc_reactor_coordination"] is True
        assert enhanced_results["perplexity_coordination"] is True
        assert enhanced_results["einstein_coordination"] is True
        assert "trinity_signature" in enhanced_results

if __name__ == "__main__":
    # Allow running directly for quick testing
    async def run_tests():
        print("🚀 MeeTARA Lab - Training Orchestrator Integration Tests")
        print("=" * 60)
        
        orchestrator = TrainingOrchestrator()
        test_instance = TestTrainingOrchestrator()
        
        print("📊 Testing orchestrator statistics...")
        await test_instance.test_orchestrator_statistics(orchestrator)
        print("✅ Statistics test passed")
        
        print("🎯 Testing sample domain training...")
        await test_instance.test_sample_domain_training(orchestrator)
        print("✅ Sample training test completed")
        
        print("💰 Testing cost optimization...")
        await test_instance.test_cost_optimization(orchestrator)
        print("✅ Cost optimization test passed")
        
        print("🎼 Testing Trinity Architecture integration...")
        await test_instance.test_trinity_architecture_integration(orchestrator)
        print("✅ Trinity integration test passed")
        
        print("\n🎉 All integration tests completed successfully!")
    
    asyncio.run(run_tests()) 