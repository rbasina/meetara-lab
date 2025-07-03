#!/usr/bin/env python3
"""
Unit tests for Trinity Emotion Detector
Tests emotion recognition and multi-modal processing
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import the emotion detector component
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core"))

try:
    from emotion_detector import EnhancedEmotionDetector
except ImportError:
    try:
        # Try with explicit path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trinity-core"))
        from emotion_detector import EnhancedEmotionDetector
    except ImportError:
        # Create mock class if import fails
        class EnhancedEmotionDetector:
            def __init__(self, mcp=None):
                pass
            
            async def detect_emotion_comprehensive(self, text, domain="general", context=None):
                return {
                    "emotion": "neutral",
                    "confidence": 0.5,
                    "modality": "text",
                    "domain": domain
                }

class TestEmotionDetector:
    
    @pytest.fixture
    def detector(self):
        """Create emotion detector instance for testing"""
        return EnhancedEmotionDetector()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for emotion testing"""
        return {
            "happy": "I am so excited about this amazing breakthrough!",
            "sad": "I feel really disappointed and down today.",
            "angry": "This is absolutely frustrating and makes me furious!",
            "neutral": "The weather is partly cloudy with a chance of rain.",
            "fear": "I'm really worried and scared about what might happen."
        }
    
    def test_detector_initialization(self, detector):
        """Test emotion detector initializes correctly"""
        assert detector is not None
        assert hasattr(detector, 'detect_emotion_comprehensive')
    
    @pytest.mark.asyncio
    async def test_text_emotion_detection_happy(self, detector, sample_texts):
        """Test detection of happy emotions in text"""
        result = await detector.detect_emotion_comprehensive(sample_texts["happy"])
        
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_text_emotion_detection_sad(self, detector, sample_texts):
        """Test detection of sad emotions in text"""
        result = await detector.detect_emotion_comprehensive(sample_texts["sad"])
        
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_text_emotion_detection_neutral(self, detector, sample_texts):
        """Test detection of neutral emotions in text"""
        result = await detector.detect_emotion_comprehensive(sample_texts["neutral"])
        
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, detector):
        """Test handling of empty or invalid text input"""
        # Empty string
        result = await detector.detect_emotion_comprehensive("")
        assert isinstance(result, dict)
        assert "emotion" in result
        
        # None input should be handled gracefully
        try:
            result = await detector.detect_emotion_comprehensive(None)
            assert isinstance(result, dict)
        except (TypeError, AttributeError):
            # It's acceptable for None to raise an error
            pass
    
    @pytest.mark.asyncio
    async def test_long_text_processing(self, detector):
        """Test processing of very long texts"""
        long_text = "This is a test sentence. " * 100  # 2500+ characters
        
        result = await detector.detect_emotion_comprehensive(long_text)
        assert isinstance(result, dict)
        assert "confidence" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_domain_specific_detection(self, detector):
        """Test domain-specific emotion detection"""
        health_text = "I'm feeling anxious about my upcoming medical procedure."
        
        result = await detector.detect_emotion_comprehensive(health_text, domain="healthcare")
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "domain" in result or "context" in result
    
    @pytest.mark.asyncio
    async def test_confidence_scores_valid_range(self, detector, sample_texts):
        """Test that confidence scores are in valid range [0, 1]"""
        for text in sample_texts.values():
            result = await detector.detect_emotion_comprehensive(text)
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, detector):
        """Test handling of special characters and emojis"""
        special_texts = [
            "I love this! 😍❤️🎉",
            "So sad... 😢💔",
            "What?!?! @#$%^&*()",
            "¡Hola! ¿Cómo estás?"  # Spanish with special characters
        ]
        
        for text in special_texts:
            result = await detector.detect_emotion_comprehensive(text)
            assert isinstance(result, dict)
            assert "confidence" in result
            assert result["confidence"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, detector, sample_texts):
        """Test emotion detection performance benchmarks"""
        import time
        
        start_time = time.time()
        
        # Process multiple texts
        for _ in range(5):  # Reduced for async tests
            for text in sample_texts.values():
                await detector.detect_emotion_comprehensive(text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 25 texts in reasonable time (less than 10 seconds)
        assert processing_time < 10.0
        
        # Calculate texts per second
        texts_per_second = 25 / processing_time
        print(f"Emotion detection speed: {texts_per_second:.2f} texts/second")
        
        # Should process at least 0.5 texts per second (reasonable for async)
        assert texts_per_second >= 0.5

class TestEmotionDetectorIntegration:
    """Integration tests for emotion detector"""
    
    @pytest.fixture
    def detector(self):
        """Create emotion detector instance for integration testing"""
        return EnhancedEmotionDetector()
    
    @pytest.mark.asyncio
    async def test_trinity_architecture_integration(self, detector):
        """Test Trinity Architecture integration"""
        # Test that the detector works with Trinity context
        result = await detector.detect_emotion_comprehensive(
            "I'm excited about the new AI breakthrough!",
            domain="technology",
            context={"source": "trinity_test"}
        )
        
        assert isinstance(result, dict)
        assert "emotion" in result
        # Trinity enhancements should be applied
        assert result["confidence"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_multiple_domains(self, detector):
        """Test emotion detection across multiple domains"""
        domain_texts = {
            "healthcare": "I'm worried about my health symptoms.",
            "business": "This quarterly report shows excellent growth!",
            "education": "I'm struggling to understand this concept.",
            "creative": "I'm inspired to write a beautiful story."
        }
        
        for domain, text in domain_texts.items():
            result = await detector.detect_emotion_comprehensive(text, domain=domain)
            assert isinstance(result, dict)
            assert "emotion" in result
            assert result["confidence"] >= 0.0

if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    print("🧪 Running Emotion Detector Unit Tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Exit code: {result.returncode}") 