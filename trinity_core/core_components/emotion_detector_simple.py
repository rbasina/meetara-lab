#!/usr/bin/env python3
"""
Simplified Emotion Detector for Colab
Avoids transformers to prevent CUDA version conflicts
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SimpleEmotionDetector:
    """Simplified emotion detector that doesn't use transformers"""
    
    def __init__(self):
        self.emotion_patterns = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'cheerful', 'elated'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'sorrowful', 'unhappy'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad', 'frustrated'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened'],
            'neutral': ['neutral', 'calm', 'composed', 'balanced', 'stable']
        }
        
        self.domain_patterns = {
            'healthcare': {
                'concern': ['worried', 'concerned', 'anxious', 'stressed'],
                'relief': ['better', 'improved', 'relieved', 'recovered'],
                'pain': ['hurt', 'pain', 'ache', 'sore', 'uncomfortable']
            },
            'business': {
                'frustration': ['frustrated', 'annoyed', 'irritated'],
                'satisfaction': ['pleased', 'satisfied', 'happy'],
                'stress': ['stressed', 'overwhelmed', 'pressured']
            }
        }
        
        self.stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'fallback_usage': 0,
            'domain_specific': 0
        }
    
    async def detect_emotion(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Detect emotions using simple pattern matching"""
        self.stats['total_requests'] += 1
        
        try:
            # Convert to lowercase for pattern matching
            text_lower = text.lower()
            
            # Initialize emotion scores
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_patterns.keys()}
            
            # Count emotion words
            for emotion, patterns in self.emotion_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        emotion_scores[emotion] += 1.0
            
            # Normalize scores
            total_matches = sum(emotion_scores.values())
            if total_matches > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_matches
            else:
                # Default to neutral if no emotions detected
                emotion_scores['neutral'] = 1.0
            
            # Get primary emotion
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            # Add domain-specific analysis
            domain_analysis = await self._analyze_domain_context(text_lower, domain)
            
            result = {
                'success': True,
                'primary_emotion': primary_emotion[0],
                'confidence': primary_emotion[1],
                'emotion_scores': emotion_scores,
                'domain_analysis': domain_analysis,
                'timestamp': datetime.now().isoformat(),
                'method': 'simple_pattern_matching'
            }
            
            self.stats['successful_detections'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            self.stats['fallback_usage'] += 1
            return {
                'success': False,
                'error': str(e),
                'fallback': True,
                'primary_emotion': 'neutral',
                'confidence': 0.0,
                'emotion_scores': {'neutral': 1.0},
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_domain_context(self, text: str, domain: str) -> Dict[str, Any]:
        """Analyze domain-specific context"""
        if domain not in self.domain_patterns:
            return {'domain_specific': False}
        
        domain_patterns = self.domain_patterns[domain]
        domain_emotions = {}
        
        for emotion_type, patterns in domain_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text)
            if count > 0:
                domain_emotions[emotion_type] = count
        
        return {
            'domain_specific': True,
            'domain': domain,
            'domain_emotions': domain_emotions,
            'has_domain_context': len(domain_emotions) > 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'total_requests': self.stats['total_requests'],
            'successful_detections': self.stats['successful_detections'],
            'fallback_usage': self.stats['fallback_usage'],
            'success_rate': self.stats['successful_detections'] / max(self.stats['total_requests'], 1),
            'method': 'simple_pattern_matching'
        } 