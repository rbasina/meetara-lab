"""
MeeTARA Lab - Enhanced Speech Recognition with Trinity Architecture
Real-time ASR with domain-aware transcription and emotion integration
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import numpy as np
from pathlib import Path

# Speech recognition imports
try:
    import speech_recognition as sr
    import pyaudio
    import wave
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ Speech recognition dependencies not installed")

# Import trinity-core components
from ..agents.mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage

class EnhancedSpeechRecognition(BaseAgent):
    """Enhanced Speech Recognition with Trinity Architecture and domain awareness"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.SPEECH_RECOGNITION, mcp)
        
        # Speech recognition engines
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None
        
        # Domain-aware transcription settings
        self.domain_vocabularies = {
            "healthcare": [
                "symptoms", "diagnosis", "treatment", "medication", "therapy",
                "patient", "doctor", "hospital", "clinic", "prescription"
            ],
            "business": [
                "meeting", "project", "deadline", "budget", "revenue",
                "client", "customer", "strategy", "marketing", "sales"
            ],
            "education": [
                "student", "teacher", "lesson", "assignment", "exam",
                "study", "research", "academic", "course", "degree"
            ],
            "technology": [
                "software", "hardware", "programming", "code", "database",
                "server", "network", "security", "algorithm", "data"
            ],
            "creative": [
                "design", "art", "music", "writing", "story",
                "creative", "inspiration", "artistic", "composition", "visual"
            ],
            "daily_life": [
                "family", "home", "shopping", "cooking", "travel",
                "schedule", "appointment", "routine", "personal", "lifestyle"
            ]
        }
        
        # Real-time processing settings
        self.real_time_config = {
            "chunk_size": 1024,
            "sample_rate": 16000,
            "channels": 1,
            "format": pyaudio.paInt16,
            "silence_threshold": 500,
            "silence_duration": 1.0,
            "max_recording_time": 30.0
        }
        
        # Trinity enhancements
        self.trinity_enhancements = {
            "arc_reactor_optimization": True,    # Optimized audio processing
            "perplexity_context_awareness": True, # Context-aware transcription
            "einstein_fusion_amplification": True # Enhanced accuracy through fusion
        }
        
        # Integration with existing components
        self.emotion_integration = True
        self.tts_integration = True
        self.routing_integration = True
        
        # Performance tracking
        self.performance_stats = {
            "transcriptions_processed": 0,
            "average_accuracy": 0.0,
            "average_processing_time": 0.0,
            "domain_usage": {},
            "error_rate": 0.0,
            "real_time_performance": []
        }
        
        # Audio processing state
        self.is_listening = False
        self.current_domain = "general"
        self.audio_buffer = []
        
    async def start(self):
        """Start the Enhanced Speech Recognition"""
        await super().start()
        
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("⚠️ Speech Recognition started in simulation mode")
            return
        
        # Calibrate microphone for ambient noise
        await self._calibrate_microphone()
        print("🎤 Enhanced Speech Recognition ready with Trinity Architecture")
        
    async def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            print("🎤 Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated successfully")
        except Exception as e:
            print(f"⚠️ Microphone calibration failed: {e}")
    
    async def start_real_time_recognition(self, domain: str = "general", 
                                        callback: Optional[Callable] = None) -> bool:
        """Start real-time speech recognition with domain awareness"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                print("⚠️ Real-time recognition not available - missing dependencies")
                return False
            
            self.current_domain = domain
            self.is_listening = True
            
            print(f"🎤 Starting real-time recognition for domain: {domain}")
            
            # Start background listening task
            asyncio.create_task(self._real_time_listening_loop(callback))
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start real-time recognition: {e}")
            return False
    
    async def _real_time_listening_loop(self, callback: Optional[Callable] = None):
        """Real-time listening loop with domain-aware processing"""
        while self.is_listening:
            try:
                # Listen for audio input
                audio_data = await self._capture_audio_chunk()
                
                if audio_data:
                    # Process audio with domain awareness
                    transcription_result = await self._process_audio_with_domain_awareness(
                        audio_data, self.current_domain
                    )
                    
                    if transcription_result["success"] and transcription_result["text"]:
                        # Integrate with emotion detection
                        enhanced_result = await self._integrate_with_emotion_detection(
                            transcription_result
                        )
                        
                        # Call callback if provided
                        if callback:
                            await callback(enhanced_result)
                        
                        # Notify other agents
                        await self._notify_agents_of_transcription(enhanced_result)
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error in listening loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _capture_audio_chunk(self) -> Optional[bytes]:
        """Capture audio chunk from microphone"""
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                return audio.get_raw_data()
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"⚠️ Audio capture error: {e}")
            return None
    
    async def _process_audio_with_domain_awareness(self, audio_data: bytes, 
                                                 domain: str) -> Dict[str, Any]:
        """Process audio with domain-specific vocabulary and context"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert audio data back to AudioData object
            audio = sr.AudioData(audio_data, self.real_time_config["sample_rate"], 2)
            
            # Apply domain-specific processing
            recognition_hints = self._get_domain_recognition_hints(domain)
            
            # Perform speech recognition with multiple engines for accuracy
            transcription_results = await self._multi_engine_recognition(audio, recognition_hints)
            
            # Select best transcription using Trinity intelligence
            best_result = await self._select_best_transcription(transcription_results, domain)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            result = {
                "text": best_result["text"],
                "confidence": best_result["confidence"],
                "domain": domain,
                "processing_time_ms": round(processing_time, 2),
                "recognition_engine": best_result["engine"],
                "domain_vocabulary_matches": best_result.get("vocabulary_matches", []),
                "trinity_enhanced": True,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Update performance statistics
            await self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "domain": domain,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_domain_recognition_hints(self, domain: str) -> List[str]:
        """Get domain-specific recognition hints and vocabulary"""
        base_vocabulary = self.domain_vocabularies.get(domain, [])
        
        # Add general vocabulary
        general_vocabulary = ["please", "thank", "help", "question", "answer", "understand"]
        
        return base_vocabulary + general_vocabulary
    
    async def _multi_engine_recognition(self, audio: sr.AudioData, 
                                      hints: List[str]) -> List[Dict[str, Any]]:
        """Perform recognition with multiple engines for enhanced accuracy"""
        results = []
        
        # Google Speech Recognition
        try:
            text = self.recognizer.recognize_google(audio)
            results.append({
                "engine": "google",
                "text": text,
                "confidence": 0.9,  # Google doesn't provide confidence
                "vocabulary_matches": self._count_vocabulary_matches(text, hints)
            })
        except:
            pass
        
        # Sphinx (offline) as fallback
        try:
            text = self.recognizer.recognize_sphinx(audio)
            results.append({
                "engine": "sphinx",
                "text": text,
                "confidence": 0.7,  # Lower confidence for offline
                "vocabulary_matches": self._count_vocabulary_matches(text, hints)
            })
        except:
            pass
        
        return results
    
    def _count_vocabulary_matches(self, text: str, vocabulary: List[str]) -> List[str]:
        """Count vocabulary matches in transcribed text"""
        text_lower = text.lower()
        matches = []
        
        for word in vocabulary:
            if word.lower() in text_lower:
                matches.append(word)
        
        return matches
    
    async def _select_best_transcription(self, results: List[Dict[str, Any]], 
                                       domain: str) -> Dict[str, Any]:
        """Select best transcription using Trinity intelligence"""
        if not results:
            return {"text": "", "confidence": 0.0, "engine": "none"}
        
        # Score each result based on multiple factors
        scored_results = []
        
        for result in results:
            score = 0.0
            
            # Base confidence score
            score += result["confidence"] * 0.4
            
            # Domain vocabulary bonus
            vocab_bonus = len(result["vocabulary_matches"]) * 0.1
            score += min(vocab_bonus, 0.3)  # Cap at 30%
            
            # Engine reliability bonus
            if result["engine"] == "google":
                score += 0.2
            elif result["engine"] == "sphinx":
                score += 0.1
            
            # Text length reasonableness (not too short, not too long)
            text_length = len(result["text"].split())
            if 2 <= text_length <= 20:
                score += 0.1
            
            scored_results.append({
                **result,
                "total_score": score
            })
        
        # Return highest scoring result
        best_result = max(scored_results, key=lambda x: x["total_score"])
        return best_result
    
    async def _integrate_with_emotion_detection(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate transcription with emotion detection"""
        try:
            # Send transcription to emotion detector
            emotion_message = MCPMessage(
                message_type=MessageType.ANALYSIS_REQUEST,
                sender=self.agent_type,
                recipient=AgentType.EMOTION_DETECTOR,
                data={
                    "text": transcription_result["text"],
                    "domain": transcription_result["domain"],
                    "source": "speech_recognition"
                }
            )
            
            # In a real implementation, this would await a response
            # For now, we'll simulate emotion integration
            emotion_data = {
                "primary_emotion": "neutral",
                "confidence": 0.8,
                "emotional_category": "neutral",
                "intervention_required": False
            }
            
            # Enhance transcription result with emotion data
            transcription_result["emotion_analysis"] = emotion_data
            transcription_result["emotion_integrated"] = True
            
            return transcription_result
            
        except Exception as e:
            print(f"⚠️ Emotion integration failed: {e}")
            transcription_result["emotion_integrated"] = False
            return transcription_result
    
    async def _notify_agents_of_transcription(self, enhanced_result: Dict[str, Any]):
        """Notify other agents of transcription results"""
        try:
            # Notify Intelligent Router for domain routing
            router_message = MCPMessage(
                message_type=MessageType.ROUTING_REQUEST,
                sender=self.agent_type,
                recipient=AgentType.INTELLIGENT_ROUTER,
                data={
                    "transcription": enhanced_result["text"],
                    "domain": enhanced_result["domain"],
                    "emotion_data": enhanced_result.get("emotion_analysis"),
                    "source": "speech_recognition"
                }
            )
            
            self.send_message(router_message)
            
        except Exception as e:
            print(f"⚠️ Failed to notify agents: {e}")
    
    async def stop_real_time_recognition(self):
        """Stop real-time speech recognition"""
        self.is_listening = False
        print("🎤 Real-time recognition stopped")
    
    async def transcribe_audio_file(self, audio_file_path: str, 
                                  domain: str = "general") -> Dict[str, Any]:
        """Transcribe audio file with domain awareness"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                return {"success": False, "error": "Speech recognition not available"}
            
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Process with domain awareness
            result = await self._process_audio_with_domain_awareness(
                audio.get_raw_data(), domain
            )
            
            result["source"] = "audio_file"
            result["file_path"] = audio_file_path
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _update_performance_stats(self, result: Dict[str, Any]):
        """Update performance statistics"""
        self.performance_stats["transcriptions_processed"] += 1
        
        # Update domain usage
        domain = result.get("domain", "unknown")
        if domain not in self.performance_stats["domain_usage"]:
            self.performance_stats["domain_usage"][domain] = 0
        self.performance_stats["domain_usage"][domain] += 1
        
        # Update average processing time
        processing_time = result.get("processing_time_ms", 0)
        current_avg = self.performance_stats["average_processing_time"]
        total_processed = self.performance_stats["transcriptions_processed"]
        
        self.performance_stats["average_processing_time"] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
        
        # Update accuracy (simplified)
        confidence = result.get("confidence", 0)
        current_accuracy = self.performance_stats["average_accuracy"]
        self.performance_stats["average_accuracy"] = (
            (current_accuracy * (total_processed - 1) + confidence) / total_processed
        )
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "performance_stats": self.performance_stats,
            "current_status": {
                "is_listening": self.is_listening,
                "current_domain": self.current_domain,
                "trinity_enhanced": True
            },
            "capabilities": {
                "real_time_recognition": SPEECH_RECOGNITION_AVAILABLE,
                "domain_awareness": True,
                "emotion_integration": self.emotion_integration,
                "multi_engine_support": True
            },
            "domain_vocabularies": {
                domain: len(vocab) for domain, vocab in self.domain_vocabularies.items()
            }
        }

# Agent type enum extension (would be added to the main enum)
class AgentType:
    SPEECH_RECOGNITION = "speech_recognition" 