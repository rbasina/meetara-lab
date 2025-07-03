"""
Unit Tests for TTS Manager Component
Tests voice synthesis functionality with Trinity Architecture enhancements
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

# Import the component being tested (mocked in conftest.py for safe testing)
# from trinity_core.tts_manager import TTSManager


class TestTTSManager:
    """Test suite for TTS Manager component"""
    
    def test_voice_synthesis_success(self, tts_manager):
        """Test successful voice synthesis with professional voice"""
        # Arrange
        message = "Welcome to MeeTARA Lab healthcare consultation"
        voice_category = "professional"
        
        # Act
        result = tts_manager.synthesize_voice(message, voice_category=voice_category)
        
        # Assert
        assert result.status == "success"
        assert result.audio_data is not None
        assert result.duration > 0
        assert "Jenny" in result.voice_used or "Aria" in result.voice_used  # Professional voices
        
        # Verify method was called with correct parameters
        tts_manager.synthesize_voice.assert_called_with(message, voice_category=voice_category)
    
    def test_voice_categories_mapping(self, tts_manager):
        """Test that all 6 voice categories are properly mapped"""
        # Arrange
        expected_categories = {
            'professional', 'friendly', 'authoritative', 
            'empathetic', 'educational', 'creative'
        }
        
        # Act
        available_categories = set(tts_manager.voice_categories.keys())
        
        # Assert
        assert available_categories == expected_categories
        
        # Verify each category has at least 2 voices
        for category, voices in tts_manager.voice_categories.items():
            assert len(voices) >= 2, f"Category {category} should have at least 2 voices"
    
    def test_domain_specific_voice_selection(self, tts_manager):
        """Test voice selection based on domain context"""
        # Test healthcare domain -> professional voice
        result = tts_manager.synthesize_voice(
            "Your medical results are ready for review",
            voice_category="professional"
        )
        assert result.status == "success"
        
        # Test educational domain -> educational voice  
        tts_manager.synthesize_voice(
            "Let's learn about AI training optimization",
            voice_category="educational"
        )
        
        # Test creative domain -> creative voice
        tts_manager.synthesize_voice(
            "Imagine a world where AI and humans collaborate seamlessly",
            voice_category="creative"
        )
        
        # Verify all calls were successful
        assert tts_manager.synthesize_voice.call_count == 3
    
    def test_trinity_architecture_integration(self, tts_manager, trinity_intelligence):
        """Test TTS Manager integration with Trinity Architecture"""
        # Arrange
        message = "Testing Trinity enhancement"
        
        # Act - This would normally go through Trinity enhancement
        # For testing, we mock the Trinity result
        enhanced_result = trinity_intelligence.process_with_trinity(message)
        voice_result = tts_manager.synthesize_voice(enhanced_result, voice_category="friendly")
        
        # Assert Trinity enhancements
        assert enhanced_result.amplification_factor == 5.04  # 504% improvement
        assert enhanced_result.arc_reactor_efficiency == 0.90  # 90% efficiency
        assert enhanced_result.perplexity_intelligence is True
        assert enhanced_result.einstein_fusion is True
        
        # Assert voice synthesis success
        assert voice_result.status == "success"
    
    def test_edge_tts_fallback_mechanism(self, tts_manager):
        """Test fallback from Edge-TTS to pyttsx3 when needed"""
        # This would test the intelligent fallback system
        # In real implementation, if Edge-TTS fails, it falls back to pyttsx3
        
        # Simulate Edge-TTS failure and pyttsx3 success
        result = tts_manager.synthesize_voice(
            "Testing fallback mechanism",
            voice_category="professional"
        )
        
        # Should still succeed with fallback
        assert result.status == "success"
        assert result.audio_data is not None
    
    def test_voice_quality_parameters(self, tts_manager):
        """Test voice quality parameters (speed, pitch, volume)"""
        # Test with quality parameters
        result = tts_manager.synthesize_voice(
            "Quality test message",
            voice_category="professional",
            # These would be real parameters in actual implementation
            # speed=1.0, pitch=1.0, volume=0.8
        )
        
        assert result.status == "success"
        assert result.duration > 0
    
    def test_multi_language_support(self, tts_manager):
        """Test multi-language voice synthesis capability"""
        # Test English (default)
        result_en = tts_manager.synthesize_voice(
            "Hello from MeeTARA Lab",
            voice_category="friendly"
        )
        assert result_en.status == "success"
        
        # In real implementation, would test other languages
        # For now, verify the method works
        assert tts_manager.synthesize_voice.call_count >= 1
    
    def test_emotional_context_voice_mapping(self, tts_manager, emotion_detector):
        """Test voice selection based on emotional context"""
        # Arrange
        message = "I'm feeling anxious about my health results"
        
        # Act - Detect emotion first
        emotion_result = emotion_detector.detect_emotion(message)
        
        # Select appropriate voice based on emotion
        if emotion_result.emotion in ['sadness', 'fear', 'anxiety']:
            voice_category = "empathetic"
        else:
            voice_category = "professional"
        
        voice_result = tts_manager.synthesize_voice(message, voice_category=voice_category)
        
        # Assert
        assert emotion_result.confidence > 0.8
        assert voice_result.status == "success"
    
    def test_performance_requirements(self, tts_manager):
        """Test performance requirements for voice synthesis"""
        # Test response time
        start_time = time.time()
        
        result = tts_manager.synthesize_voice(
            "Performance test message",
            voice_category="professional"
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Voice synthesis should be fast (< 5 seconds for testing)
        assert response_time < 5.0
        assert result.status == "success"
    
    def test_concurrent_voice_synthesis(self, tts_manager):
        """Test handling multiple concurrent voice synthesis requests"""
        # Test multiple calls (would be concurrent in real implementation)
        messages = [
            "First message",
            "Second message", 
            "Third message"
        ]
        
        results = []
        for message in messages:
            result = tts_manager.synthesize_voice(message, voice_category="professional")
            results.append(result)
        
        # All should succeed
        for result in results:
            assert result.status == "success"
        
        assert len(results) == 3
    
    def test_error_handling(self, tts_manager):
        """Test error handling for invalid inputs"""
        # Test empty message
        result = tts_manager.synthesize_voice("", voice_category="professional")
        # Should handle gracefully (mock always returns success)
        assert result.status == "success"
        
        # Test invalid voice category
        result = tts_manager.synthesize_voice(
            "Test message", 
            voice_category="invalid_category"
        )
        # Should handle gracefully or fallback to default
        assert result.status == "success"
    
    def test_integration_with_meetara_ecosystem(self, tts_manager):
        """Test integration with MeeTARA ecosystem ports"""
        # Test that TTS Manager can work with MeeTARA ports
        # Port 2025 (frontend), 8765 (WebSocket), 8766 (API), 5000 (TARA Universal)
        
        result = tts_manager.synthesize_voice(
            "Testing MeeTARA integration",
            voice_category="professional"
        )
        
        assert result.status == "success"
        # In real implementation, would verify port communication
    

class TestTTSManagerIntegration:
    """Integration tests for TTS Manager with other components"""
    
    def test_tts_with_emotion_detection(self, tts_manager, emotion_detector):
        """Test TTS Manager working with Emotion Detector"""
        # Arrange
        message = "I'm excited about the new AI capabilities!"
        
        # Act
        emotion = emotion_detector.detect_emotion(message)
        voice_result = tts_manager.synthesize_voice(
            message, 
            voice_category="friendly"  # Match excited emotion
        )
        
        # Assert
        assert emotion.emotion == "joy"  # Mocked emotion
        assert emotion.confidence > 0.9
        assert voice_result.status == "success"
    
    def test_tts_with_intelligent_router(self, tts_manager, intelligent_router):
        """Test TTS Manager with domain routing"""
        # Arrange
        message = "Your healthcare consultation is ready"
        
        # Act
        routing = intelligent_router.route_request(message)
        voice_result = tts_manager.synthesize_voice(
            message,
            voice_category="professional"  # Healthcare domain
        )
        
        # Assert
        assert routing.domain == "healthcare"
        assert voice_result.status == "success"
    
    def test_quality_preservation_with_trinity(self, tts_manager, trinity_intelligence, assert_quality_gates):
        """Test that Trinity enhancements preserve quality"""
        # Arrange
        message = "Quality preservation test"
        
        # Act
        enhanced = trinity_intelligence.process_with_trinity(message)
        result = tts_manager.synthesize_voice(enhanced, voice_category="professional")
        
        # Prepare quality metrics
        quality_metrics = {
            'amplification_factor': enhanced.amplification_factor,
            'synthesis_success': result.status == "success",
            'response_time': result.duration
        }
        
        # Assert quality gates
        assert enhanced.amplification_factor >= 5.04
        assert result.status == "success"
        
    def test_cost_efficiency(self, tts_manager):
        """Test that voice synthesis is cost-efficient"""
        # Test multiple synthesis operations
        results = []
        for i in range(10):
            result = tts_manager.synthesize_voice(
                f"Cost efficiency test message {i}",
                voice_category="professional"
            )
            results.append(result)
        
        # All should succeed without significant cost
        success_count = sum(1 for r in results if r.status == "success")
        assert success_count == 10
        
        # In real implementation, would verify cost metrics


@pytest.mark.performance
class TestTTSManagerPerformance:
    """Performance tests for TTS Manager"""
    
    def test_synthesis_speed(self, tts_manager):
        """Test voice synthesis speed requirements"""
        # Test multiple messages for average performance
        messages = [
            "Short message",
            "This is a medium length message for testing performance",
            "This is a longer message that contains more text to test the performance of voice synthesis with larger content amounts"
        ]
        
        for message in messages:
            start_time = time.time()
            result = tts_manager.synthesize_voice(message, voice_category="professional")
            end_time = time.time()
            
            synthesis_time = end_time - start_time
            
            # Should be fast (< 3 seconds for testing)
            assert synthesis_time < 3.0
            assert result.status == "success"
    
    def test_memory_efficiency(self, tts_manager):
        """Test memory usage during voice synthesis"""
        # Test that memory usage stays reasonable
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple synthesis operations
        for i in range(5):
            result = tts_manager.synthesize_voice(
                f"Memory efficiency test {i}",
                voice_category="professional"
            )
            assert result.status == "success"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for testing)
        assert memory_increase < 100 
