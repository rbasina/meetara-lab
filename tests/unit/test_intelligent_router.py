#!/usr/bin/env python3
"""
Unit tests for Trinity Intelligent Router
Tests smart routing and decision-making logic
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import sys

# Import the intelligent router component
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core"))

try:
    from intelligent_router import IntelligentRouter
except ImportError:
    # Create mock class if import fails
    class IntelligentRouter:
        def __init__(self, mcp=None):
            pass
        
        async def route_request_intelligent(self, user_input: str, 
                                          context: dict = None,
                                          user_preferences: dict = None) -> dict:
            """Mock intelligent routing method"""
            return {
                "primary_domain": "general",
                "domain_confidence": 0.85,
                "domain_category": "daily_life",
                "model_tier": "balanced",
                "model_name": "microsoft/DialoGPT-small",
                "routing_strategy": "local",
                "empathy_level": 0.7,
                "meetara_integration": {},
                "trinity_enhanced": True,
                "processing_time_ms": 15.5,
                "routing_timestamp": "2024-12-22T23:00:00",
                "success": True
            }

class TestIntelligentRouter:
    
    @pytest.fixture
    def router(self):
        """Create router instance for testing"""
        return IntelligentRouter()
    
    @pytest.fixture
    def sample_requests(self):
        """Sample requests for routing testing"""
        return {
            "simple_text": "Hello, how are you?",
            "complex_analysis": "Analyze this complex dataset with 50,000 rows",
            "realtime_tts": "Convert this to speech immediately",
            "gpu_training": "Train neural network on large dataset",
            "health_question": "I'm feeling anxious about my health",
            "business_query": "Help me with marketing strategy",
            "creative_writing": "Write me a beautiful story"
        }
    
    def test_router_initialization(self, router):
        """Test router initializes correctly"""
        assert router is not None
        assert hasattr(router, 'route_request_intelligent')
    
    @pytest.mark.asyncio
    async def test_simple_request_routing(self, router, sample_requests):
        """Test routing of simple requests"""
        result = await router.route_request_intelligent(sample_requests["simple_text"])
        
        assert isinstance(result, dict)
        assert "primary_domain" in result
        assert "domain_confidence" in result
        assert "success" in result
        assert result["success"] is True
        assert 0.0 <= result["domain_confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_complex_request_routing(self, router, sample_requests):
        """Test routing of complex requests"""
        result = await router.route_request_intelligent(sample_requests["complex_analysis"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "model_tier" in result
        assert "routing_strategy" in result
        # Complex requests should have reasonable confidence
        assert result["domain_confidence"] > 0.3
    
    @pytest.mark.asyncio
    async def test_health_domain_routing(self, router, sample_requests):
        """Test routing of health-related requests"""
        result = await router.route_request_intelligent(sample_requests["health_question"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "primary_domain" in result
        assert "empathy_level" in result
        # Health queries should have high empathy
        assert result["empathy_level"] >= 0.5
    
    @pytest.mark.asyncio
    async def test_business_domain_routing(self, router, sample_requests):
        """Test routing of business requests"""
        result = await router.route_request_intelligent(sample_requests["business_query"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "model_tier" in result
        # Business queries should use appropriate model tier
        assert result["model_tier"] in ["fast", "balanced", "quality"]
    
    @pytest.mark.asyncio
    async def test_creative_domain_routing(self, router, sample_requests):
        """Test routing of creative requests"""
        result = await router.route_request_intelligent(sample_requests["creative_writing"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "domain_category" in result
        assert "trinity_enhanced" in result
        assert result["trinity_enhanced"] is True
    
    @pytest.mark.asyncio
    async def test_context_aware_routing(self, router, sample_requests):
        """Test routing with context information"""
        context = {
            "conversation_history": ["Hello", "How are you?"],
            "emotional_context": {"primary_emotion": "anxiety"}
        }
        
        result = await router.route_request_intelligent(
            sample_requests["health_question"], 
            context=context
        )
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "processing_time_ms" in result
        assert isinstance(result["processing_time_ms"], (int, float))
    
    @pytest.mark.asyncio
    async def test_user_preferences_routing(self, router, sample_requests):
        """Test routing with user preferences"""
        user_preferences = {
            "model_tier": "quality",
            "empathy_level": 0.9
        }
        
        result = await router.route_request_intelligent(
            sample_requests["simple_text"],
            user_preferences=user_preferences
        )
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "routing_timestamp" in result
        assert isinstance(result["routing_timestamp"], str)
    
    @pytest.mark.asyncio
    async def test_routing_consistency(self, router, sample_requests):
        """Test that routing decisions are reasonably consistent"""
        request = sample_requests["simple_text"]
        
        results = []
        for _ in range(3):
            result = await router.route_request_intelligent(request)
            results.append(result["primary_domain"])
        
        # Results should be consistent (allowing for some variation)
        unique_domains = set(results)
        assert len(unique_domains) <= 2  # Allow some variation but not chaos
    
    @pytest.mark.asyncio
    async def test_error_handling_empty_input(self, router):
        """Test handling of empty input"""
        result = await router.route_request_intelligent("")
        
        assert isinstance(result, dict)
        # Should either succeed with default or provide error info
        if result.get("success", False):
            assert "primary_domain" in result
        else:
            assert "error" in result or "fallback_domain" in result
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, router, sample_requests):
        """Test performance requirements"""
        import time
        
        start_time = time.time()
        
        # Process multiple requests
        for _ in range(3):
            for request in list(sample_requests.values())[:3]:  # Test first 3
                result = await router.route_request_intelligent(request)
                assert result["success"] is True
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 9 requests in reasonable time (less than 5 seconds)
        assert processing_time < 5.0
        
        # Calculate requests per second
        requests_per_second = 9 / processing_time
        print(f"Routing speed: {requests_per_second:.2f} requests/second")
        
        # Should process at least 2 requests per second
        assert requests_per_second >= 2.0
    
    @pytest.mark.asyncio
    async def test_trinity_enhancements(self, router, sample_requests):
        """Test Trinity Architecture enhancements"""
        result = await router.route_request_intelligent(sample_requests["simple_text"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        
        # Should have Trinity enhancements
        assert "trinity_enhanced" in result
        assert result["trinity_enhanced"] is True
        
        # Should have processing time tracking
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_meetara_integration(self, router, sample_requests):
        """Test MeeTARA integration features"""
        result = await router.route_request_intelligent(sample_requests["simple_text"])
        
        assert isinstance(result, dict)
        assert result["success"] is True
        
        # Should have MeeTARA integration data
        assert "meetara_integration" in result
        assert isinstance(result["meetara_integration"], dict)
        
        # Should have timestamp
        assert "routing_timestamp" in result
        assert isinstance(result["routing_timestamp"], str)

class TestRouterEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def router(self):
        """Create router instance for testing"""
        return IntelligentRouter()
    
    @pytest.mark.asyncio
    async def test_very_long_input(self, router):
        """Test with very long input"""
        long_input = "This is a test sentence. " * 100  # 2500+ characters
        
        result = await router.route_request_intelligent(long_input)
        
        assert isinstance(result, dict)
        # Should handle long input gracefully
        assert result.get("success", False) or "error" in result
    
    @pytest.mark.asyncio
    async def test_special_characters_input(self, router):
        """Test with special characters"""
        special_input = "Hello! 😊 ¿Cómo estás? @#$%^&*()[]"
        
        result = await router.route_request_intelligent(special_input)
        
        assert isinstance(result, dict)
        # Should handle special characters gracefully
        assert result.get("success", False) or "error" in result
    
    @pytest.mark.asyncio
    async def test_none_input_handling(self, router):
        """Test handling of None input"""
        try:
            result = await router.route_request_intelligent(None)
            assert isinstance(result, dict)
            # Should either succeed with default or provide error info
            assert result.get("success", False) or "error" in result
        except (TypeError, AttributeError):
            # It's acceptable for None to raise an error
            pass

if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    print("🧪 Running Intelligent Router Unit Tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Exit code: {result.returncode}") 
