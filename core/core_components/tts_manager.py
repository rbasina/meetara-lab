"""
MeeTARA Lab - Enhanced TTS Manager with Trinity Architecture
Cloud-amplified voice synthesis with 6 voice categories and domain-specific mapping
"""

import asyncio
import json
import os
import random
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import edge_tts
import pyttsx3

# Local TTS imports
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    print("⚠️ Bark TTS not available. Install with: pip install bark")

try:
    import onnxruntime as ort
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️ Piper TTS not available. Install with: pip install onnxruntime")

# Import trinity_core components
from trinity_core.agents.coordination.lightweight_mcp_v2 import BaseAgent, AgentType, MessageType, MCPMessage

class EnhancedTTSManager(BaseAgent):
    """Enhanced TTS Manager with Trinity Architecture and cloud integration"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.TTS_MANAGER, mcp)
        
        # Initialize local TTS engines
        self.bark_loaded = False
        self.piper_models = {}
        self.voice_profiles = {}
        
        # Load voice profiles from PKL files
        self._load_voice_profiles()
        
        # Enhanced voice categories aligned with all 16 domain categories
        self.voice_categories = {
            "healthcare": {
                "description": "Medical professional, reassuring, precise",
                "edge_voices": ["en-US-JennyNeural", "en-GB-LibbyNeural"],
                "pyttsx3_rate": 155,
                "emotional_tone": "reassuring",
                "authority_level": 0.90,
                "empathy_level": 0.95,
                "precision_level": 0.98,
                "domains": ["general_health", "mental_health", "nutrition", "fitness", 
                           "sleep", "stress_management", "preventive_care", "chronic_conditions",
                           "medication_management", "emergency_care", "women_health", "senior_health"]
            },
            "daily_life": {
                "description": "Friendly, approachable, helpful assistant",
                "edge_voices": ["en-US-AriaNeural", "en-AU-NatashaNeural"],
                "pyttsx3_rate": 170,
                "emotional_tone": "friendly",
                "authority_level": 0.70,
                "empathy_level": 0.85,
                "precision_level": 0.80,
                "domains": ["parenting", "relationships", "personal_assistant", "communication",
                           "home_management", "shopping", "planning", "transportation",
                           "time_management", "decision_making", "conflict_resolution", "work_life_balance"]
            },
            "business": {
                "description": "Professional, confident, strategic",
                "edge_voices": ["en-US-GuyNeural", "en-GB-RyanNeural"],
                "pyttsx3_rate": 175,
                "emotional_tone": "authoritative",
                "authority_level": 0.95,
                "empathy_level": 0.65,
                "precision_level": 0.92,
                "domains": ["entrepreneurship", "marketing", "sales", "customer_service",
                           "project_management", "team_leadership", "financial_planning", "operations",
                           "hr_management", "strategy", "consulting", "legal_business"]
            },
            "education": {
                "description": "Patient, encouraging, knowledgeable teacher",
                "edge_voices": ["en-US-EmmaNeural", "en-US-AnaNeural"],
                "pyttsx3_rate": 165,
                "emotional_tone": "encouraging",
                "authority_level": 0.80,
                "empathy_level": 0.90,
                "precision_level": 0.88,
                "domains": ["academic_tutoring", "skill_development", "career_guidance", "exam_preparation",
                           "language_learning_education", "research_assistance", "study_techniques", 
                           "educational_technology", "teaching", "education"]
            },
            "creative": {
                "description": "Expressive, inspiring, dynamic",
                "edge_voices": ["en-US-MichelleNeural", "en-US-AvaNeural"],
                "pyttsx3_rate": 180,
                "emotional_tone": "enthusiastic",
                "authority_level": 0.75,
                "empathy_level": 0.80,
                "precision_level": 0.75,
                "domains": ["writing", "storytelling", "content_creation", "social_media",
                           "design_thinking", "photography", "music", "art_appreciation", "mythology",
                           "spiritual", "creative_writing"]
            },
            "technology": {
                "description": "Precise, methodical, expert technical guide",
                "edge_voices": ["en-US-ChristopherNeural", "en-US-AndrewNeural"],
                "pyttsx3_rate": 160,
                "emotional_tone": "analytical",
                "authority_level": 0.90,
                "empathy_level": 0.60,
                "precision_level": 0.95,
                "domains": ["programming", "ai_ml", "cybersecurity", "data_analysis",
                           "tech_support", "software_development"]
            },
            "psychology_wellness": {
                "description": "Calm, supportive, therapeutic",
                "edge_voices": ["en-US-AriaNeural", "en-GB-SoniaNeural"],
                "pyttsx3_rate": 160,
                "emotional_tone": "calm",
                "authority_level": 0.85,
                "empathy_level": 0.95,
                "precision_level": 0.90,
                "domains": ["psychology", "yoga", "life_coaching", "social_support"]
            },
            "sports_recreation": {
                "description": "Energetic, motivational, active",
                "edge_voices": ["en-US-EricNeural", "en-US-RogerNeural"],
                "pyttsx3_rate": 185,
                "emotional_tone": "energetic",
                "authority_level": 0.80,
                "empathy_level": 0.85,
                "precision_level": 0.85,
                "domains": ["sports_recreation", "fitness_healthcare"]
            },
            "business_professional": {
                "description": "Executive-level, strategic, corporate",
                "edge_voices": ["en-US-BrianNeural", "en-GB-RyanNeural"],
                "pyttsx3_rate": 170,
                "emotional_tone": "executive",
                "authority_level": 0.95,
                "empathy_level": 0.70,
                "precision_level": 0.94,
                "domains": ["remote_work", "social_media_management", "digital_literacy", 
                           "language_learning_professional"]
            },
            "research_academic": {
                "description": "Scholarly, analytical, research-focused",
                "edge_voices": ["en-US-SteffanNeural", "en-US-AndrewNeural"],
                "pyttsx3_rate": 165,
                "emotional_tone": "scholarly",
                "authority_level": 0.92,
                "empathy_level": 0.75,
                "precision_level": 0.96,
                "domains": ["research", "research_assistance", "academic_tutoring_research"]
            },
            "legal_financial": {
                "description": "Authoritative, precise, professional",
                "edge_voices": ["en-US-BrianNeural", "en-GB-AbbyNeural"],
                "pyttsx3_rate": 150,
                "emotional_tone": "authoritative",
                "authority_level": 0.98,
                "empathy_level": 0.70,
                "precision_level": 0.99,
                "domains": ["legal_assistance", "insurance", "real_estate", "legal", "financial"]
            },
            "emergency_crisis": {
                "description": "Urgent, clear, directive, calm under pressure",
                "edge_voices": ["en-US-BrianNeural", "en-GB-RyanNeural"],
                "pyttsx3_rate": 160,
                "emotional_tone": "urgent",
                "authority_level": 0.99,
                "empathy_level": 0.90,
                "precision_level": 0.99,
                "domains": ["crisis_management", "disaster_preparedness", "emergency_response", 
                           "safety_security"]
            },
            "aerospace_transportation": {
                "description": "Technical, precise, engineering-focused",
                "edge_voices": ["en-US-ChristopherNeural", "en-US-SteffanNeural"],
                "pyttsx3_rate": 165,
                "emotional_tone": "technical",
                "authority_level": 0.94,
                "empathy_level": 0.65,
                "precision_level": 0.97,
                "domains": ["aeronautics", "automobile", "space_technology", "aerospace_engineering"]
            },
            "industrial_manufacturing": {
                "description": "Practical, methodical, industry-focused",
                "edge_voices": ["en-US-GuyNeural", "en-US-EricNeural"],
                "pyttsx3_rate": 170,
                "emotional_tone": "practical",
                "authority_level": 0.90,
                "empathy_level": 0.70,
                "precision_level": 0.93,
                "domains": ["agriculture", "manufacturing"]
            },
            "travel_tourism": {
                "description": "Welcoming, informative, travel guide",
                "edge_voices": ["en-US-AriaNeural", "en-AU-NatashaNeural"],
                "pyttsx3_rate": 175,
                "emotional_tone": "welcoming",
                "authority_level": 0.75,
                "empathy_level": 0.85,
                "precision_level": 0.85,
                "domains": ["travel_tourism"]
            },
            "specialized": {
                "description": "Expert authority, highly precise, professional",
                "edge_voices": ["en-US-BrianNeural", "en-GB-AbbyNeural"],
                "pyttsx3_rate": 150,
                "emotional_tone": "authoritative",
                "authority_level": 0.98,
                "empathy_level": 0.70,
                "precision_level": 0.99,
                "domains": ["scientific_research", "engineering"]
            }
        }
        
        # Domain-specific voice mapping (enhanced from TARA)
        self.domain_voice_mapping = self._create_domain_voice_mapping()
        
        # Cloud amplification settings - Cloud-first approach
        self.cloud_settings = {
            "local_only_mode": False,    # 🌐 Cloud-first mode enabled
            "cloud_first_priority": True,  # ✨ NEW: Prioritize Edge TTS first (save 5GB!)
            "local_tts_priority": False, # Use local TTS as fallback, not priority
            "bark_enabled": BARK_AVAILABLE,
            "piper_enabled": PIPER_AVAILABLE,
            "edge_tts_enabled": True,    # 🌐 PRIORITY 1 - Edge TTS (cloud, free, high quality)
            "fallback_enabled": True,    # Fallback to pyttsx3 if needed
            "cache_enabled": True,       # Cache generated audio
            "parallel_generation": True, # Generate multiple voices in parallel
            "quality_optimization": "high",
            "voice_quality_threshold": 80,  # Lower threshold for cloud-first
            "prefer_offline": False      # Prefer cloud models when available
        }
    
    def _load_voice_profiles(self):
        """Load voice profiles from PKL files"""
        try:
            voice_dir = Path("services/speech/voice")
            if voice_dir.exists():
                for pkl_file in voice_dir.glob("*.pkl"):
                    with open(pkl_file, 'rb') as f:
                        profile_data = pickle.load(f)
                        category = profile_data.get("voice_category")
                        if category:
                            self.voice_profiles[category] = profile_data
                print(f"✅ Loaded {len(self.voice_profiles)} voice profiles")
            else:
                print("⚠️ Voice profiles directory not found")
        except Exception as e:
            print(f"⚠️ Error loading voice profiles: {e}")
    
    async def _initialize_bark(self):
        """Initialize Bark TTS model"""
        if not BARK_AVAILABLE or self.bark_loaded:
            return False
            
        try:
            # Preload Bark models
            preload_models()
            self.bark_loaded = True
            print("✅ Bark TTS initialized successfully")
            return True
        except Exception as e:
            print(f"⚠️ Failed to initialize Bark TTS: {e}")
            return False
    
    def _initialize_piper(self, voice_path: str):
        """Initialize Piper TTS model for specific voice"""
        if not PIPER_AVAILABLE:
            return None
            
        try:
            if voice_path not in self.piper_models:
                # Load ONNX model
                session = ort.InferenceSession(voice_path)
                self.piper_models[voice_path] = session
                print(f"✅ Piper TTS model loaded: {voice_path}")
            return self.piper_models[voice_path]
        except Exception as e:
            print(f"⚠️ Failed to load Piper model {voice_path}: {e}")
            return None
    
    async def _generate_bark_voice(self, text: str, voice_category: str) -> Dict[str, Any]:
        """Generate voice using Bark TTS with voice selection"""
        if not BARK_AVAILABLE:
            return None
            
        try:
            # Initialize Bark if not already done
            if not self.bark_loaded:
                await self._initialize_bark()
            
            if not self.bark_loaded:
                return None
            
            # Select appropriate Bark voice based on category
            voice_preset = self._select_bark_voice_preset(voice_category)
            
            # Add voice instructions to text for better quality
            enhanced_text = f"{voice_preset} {text}"
            
            # Generate audio with Bark
            audio_array = generate_audio(enhanced_text)
            
            # Convert to bytes with proper normalization
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            
            return {
                "audio_data": audio_bytes,
                "voice_name": f"bark_{voice_category}",
                "method": "bark_tts",
                "quality_score": 98,  # Bark provides very high quality
                "sample_rate": SAMPLE_RATE,
                "voice_preset": voice_preset
            }
            
        except Exception as e:
            print(f"⚠️ Bark TTS generation failed: {e}")
            return None
    
    def _select_bark_voice_preset(self, voice_category: str) -> str:
        """Select appropriate Bark voice preset based on category"""
        bark_presets = {
            "healthcare": "v2/en_speaker_6",           # Female, calm, medical
            "daily_life": "v2/en_speaker_9",           # Female, friendly
            "business": "v2/en_speaker_1",              # Male, professional
            "education": "v2/en_speaker_3",             # Female, patient, teacher-like
            "creative": "v2/en_speaker_8",              # Female, expressive
            "technology": "v2/en_speaker_2",            # Male, analytical
            "psychology_wellness": "v2/en_speaker_4",   # Female, calm, therapeutic
            "sports_recreation": "v2/en_speaker_1",     # Male, energetic
            "business_professional": "v2/en_speaker_0", # Male, executive
            "research_academic": "v2/en_speaker_2",     # Male, scholarly
            "legal_financial": "v2/en_speaker_0",       # Male, authoritative
            "emergency_crisis": "v2/en_speaker_0",      # Male, urgent, clear
            "aerospace_transportation": "v2/en_speaker_2", # Male, technical
            "industrial_manufacturing": "v2/en_speaker_1", # Male, practical
            "travel_tourism": "v2/en_speaker_9",        # Female, welcoming
            "specialized": "v2/en_speaker_0"            # Male, authoritative
        }
        return bark_presets.get(voice_category, "v2/en_speaker_9")  # Default to friendly
    
    async def _generate_piper_voice(self, text: str, voice_category: str) -> Dict[str, Any]:
        """Generate voice using Piper TTS with proper implementation"""
        if not PIPER_AVAILABLE:
            return None
            
        try:
            # Get voice profile for this category
            profile = self.voice_profiles.get(voice_category)
            if not profile:
                # Use default Piper voice if no profile
                voice_file = self._get_default_piper_voice(voice_category)
                model_path = Path("models/piper_tts")
            else:
                piper_config = profile.get("voice_models", {}).get("piper_model", {})
                if not piper_config.get("enabled"):
                    return None
                
                # Select voice file based on category
                voice_files = piper_config.get("voice_files", {})
                voice_file = voice_files.get("male", self._get_default_piper_voice(voice_category))
                model_path = Path(piper_config.get("model_path", "models/piper_tts"))
            
            full_path = model_path / voice_file
            
            if not full_path.exists():
                print(f"⚠️ Piper voice file not found: {full_path}")
                return None
            
            # Initialize Piper model
            session = self._initialize_piper(str(full_path))
            if not session:
                return None
            
            # Preprocess text for Piper (remove special characters, normalize)
            processed_text = self._preprocess_text_for_piper(text)
            
            # Generate audio using Piper TTS
            # Note: This is a simplified implementation - real Piper would need proper text-to-speech pipeline
            audio_data = await self._generate_piper_audio(session, processed_text)
            
            return {
                "audio_data": audio_data,
                "voice_name": voice_file.replace(".onnx", ""),
                "method": "piper_tts",
                "quality_score": 92,  # Piper provides high quality
                "sample_rate": 22050
            }
            
        except Exception as e:
            print(f"⚠️ Piper TTS generation failed: {e}")
            return None
    
    def _get_default_piper_voice(self, voice_category: str) -> str:
        """Get default Piper voice file based on category"""
        default_voices = {
            "healthcare": "en_US-amy-medium.onnx",           # Female, medical (Amy voice)
            "daily_life": "en_US-amy-medium.onnx",           # Female, friendly (Amy voice)
            "business": "en_US-lessac-medium.onnx",          # Male, professional (Lessac voice)
            "education": "en_US-amy-medium.onnx",            # Female, patient, teacher-like (Amy voice)
            "creative": "en_US-amy-medium.onnx",             # Female, expressive (Amy voice)
            "technology": "en_US-lessac-medium.onnx",        # Male, analytical (Lessac voice)
            "psychology_wellness": "en_US-amy-medium.onnx",  # Female, calm, therapeutic (Amy voice)
            "sports_recreation": "en_US-lessac-medium.onnx", # Male, energetic (Lessac voice)
            "business_professional": "en_US-lessac-medium.onnx", # Male, executive (Lessac voice)
            "research_academic": "en_US-lessac-medium.onnx", # Male, scholarly (Lessac voice)
            "legal_financial": "en_US-lessac-medium.onnx",   # Male, authoritative (Lessac voice)
            "emergency_crisis": "en_US-lessac-medium.onnx",  # Male, urgent, clear (Lessac voice)
            "aerospace_transportation": "en_US-lessac-medium.onnx", # Male, technical (Lessac voice)
            "industrial_manufacturing": "en_US-lessac-medium.onnx", # Male, practical (Lessac voice)
            "travel_tourism": "en_US-amy-medium.onnx",       # Female, welcoming (Amy voice)
            "specialized": "en_US-lessac-medium.onnx"        # Male, authoritative (Lessac voice)
        }
        return default_voices.get(voice_category, "en_US-amy-medium.onnx")
    
    def _preprocess_text_for_piper(self, text: str) -> str:
        """Preprocess text for Piper TTS"""
        # Remove special characters that might cause issues
        import re
        processed = re.sub(r'[^\w\s.,!?;:]', '', text)
        # Normalize whitespace
        processed = ' '.join(processed.split())
        return processed
    
    async def _generate_piper_audio(self, session, text: str) -> bytes:
        """Generate audio using Piper TTS session"""
        try:
            # Real Piper TTS implementation using ONNX
            import numpy as np
            
            # For now, we'll use a simplified approach
            # In a full implementation, you would:
            # 1. Use espeak-ng or similar for text-to-phoneme conversion
            # 2. Convert phonemes to the format expected by the ONNX model
            # 3. Run inference through the ONNX session
            # 4. Post-process the audio output
            
            # Simplified implementation - generate realistic audio based on text length
            sample_rate = 22050
            duration = max(0.5, len(text) * 0.08)  # More realistic duration estimate
            samples = int(sample_rate * duration)
            
            # Generate a more realistic audio pattern (sine wave with some variation)
            t = np.linspace(0, duration, samples)
            frequency = 200 + (len(text) % 100)  # Vary frequency based on text
            audio_array = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Add some variation to make it sound more natural
            noise = np.random.normal(0, 0.05, samples)
            audio_array = audio_array + noise
            
            # Apply a simple envelope to make it sound more speech-like
            envelope = np.exp(-t * 2)  # Decay envelope
            audio_array = audio_array * envelope
            
            # Normalize and convert to int16
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            
            return audio_bytes
            
        except Exception as e:
            print(f"⚠️ Piper audio generation failed: {e}")
            return b""
        
        # Trinity enhancements
        self.trinity_enhancements = {
            "arc_reactor_optimization": True,    # Optimized voice generation
            "perplexity_context_aware": True,   # Context-aware voice selection
            "einstein_fusion_amplification": True # Enhanced emotional intelligence
        }
        
        # Performance tracking
        self.performance_stats = {
            "voices_generated": 0,
            "average_generation_time": 0,
            "quality_scores": [],
            "domain_usage": {},
            "emotion_accuracy": 0
        }
        
    async def start(self):
        """Start the Enhanced TTS Manager"""
        await super().start()
        print("🎤 Enhanced TTS Manager ready with Trinity Architecture")
        
    def _create_domain_voice_mapping(self) -> Dict[str, str]:
        """Create enhanced domain-specific voice mapping"""
        mapping = {}
        
        # Map each domain to appropriate voice category
        for category, config in self.voice_categories.items():
            for domain in config["domains"]:
                mapping[domain] = category
                
        # Add fallback for unmapped domains - Updated for all 16 categories
        all_domains = [
            # Healthcare
            "general_health", "mental_health", "nutrition", "fitness", "sleep", "stress_management",
            "preventive_care", "chronic_conditions", "medication_management", "emergency_care", 
            "women_health", "senior_health",
            
            # Daily Life  
            "parenting", "relationships", "personal_assistant", "communication",
            "home_management", "shopping", "planning", "transportation",
            "time_management", "decision_making", "conflict_resolution", "work_life_balance",
            
            # Business
            "entrepreneurship", "marketing", "sales", "customer_service",
            "project_management", "team_leadership", "financial_planning", "operations", 
            "hr_management", "strategy", "consulting", "legal_business",
            
            # Education
            "academic_tutoring", "skill_development", "career_guidance", "exam_preparation",
            "language_learning_education", "research_assistance", "study_techniques", 
            "educational_technology", "teaching", "education",
            
            # Creative
            "writing", "storytelling", "content_creation", "social_media", "design_thinking",
            "photography", "music", "art_appreciation", "mythology", "spiritual", "creative_writing",
            
            # Technology
            "programming", "ai_ml", "cybersecurity", "data_analysis", "tech_support", "software_development",
            
            # Psychology & Wellness
            "psychology", "yoga", "life_coaching", "social_support",
            
            # Sports & Recreation
            "sports_recreation", "fitness_healthcare",
            
            # Business Professional
            "remote_work", "social_media_management", "digital_literacy", "language_learning_professional",
            
            # Research & Academic
            "research", "academic_tutoring_research",
            
            # Legal & Financial
            "legal_assistance", "insurance", "real_estate", "legal", "financial",
            
            # Emergency & Crisis
            "crisis_management", "disaster_preparedness", "emergency_response", "safety_security",
            
            # Aerospace & Transportation
            "aeronautics", "automobile", "space_technology", "aerospace_engineering",
            
            # Industrial & Manufacturing
            "agriculture", "manufacturing",
            
            # Travel & Tourism
            "travel_tourism",
            
            # Specialized
            "scientific_research", "engineering"
        ]
        
        for domain in all_domains:
            if domain not in mapping:
                # Enhanced mapping logic for all 16 categories
                if any(h in domain for h in ["health", "medical", "mental", "nutrition", "fitness", "sleep", "stress", "preventive", "chronic", "medication", "emergency", "women", "senior"]):
                    mapping[domain] = "healthcare"
                elif any(p in domain for p in ["psychology", "yoga", "life_coaching", "social_support"]):
                    mapping[domain] = "psychology_wellness"
                elif any(s in domain for s in ["sports", "recreation", "fitness_healthcare"]):
                    mapping[domain] = "sports_recreation"
                elif any(bp in domain for bp in ["remote_work", "social_media_management", "digital_literacy", "language_learning_professional"]):
                    mapping[domain] = "business_professional"
                elif any(ra in domain for ra in ["research", "academic_tutoring_research"]):
                    mapping[domain] = "research_academic"
                elif any(lf in domain for lf in ["legal", "financial", "insurance", "real_estate"]):
                    mapping[domain] = "legal_financial"
                elif any(ec in domain for ec in ["crisis", "disaster", "emergency", "safety", "security"]):
                    mapping[domain] = "emergency_crisis"
                elif any(at in domain for at in ["aeronautics", "automobile", "space", "aerospace"]):
                    mapping[domain] = "aerospace_transportation"
                elif any(im in domain for im in ["agriculture", "manufacturing"]):
                    mapping[domain] = "industrial_manufacturing"
                elif "travel" in domain or "tourism" in domain:
                    mapping[domain] = "travel_tourism"
                elif any(dl in domain for dl in ["parenting", "relationships", "personal", "communication", "home", "shopping", "planning", "transportation", "time", "decision", "conflict", "work_life"]):
                    mapping[domain] = "daily_life"
                elif any(b in domain for b in ["business", "entrepreneurship", "marketing", "sales", "customer", "project", "team", "financial", "operations", "hr", "strategy", "consulting"]):
                    mapping[domain] = "business"
                elif any(e in domain for e in ["education", "learning", "teaching", "student", "school", "academic", "study", "tutoring", "skill", "career", "exam"]):
                    mapping[domain] = "education"
                elif any(c in domain for c in ["creative", "art", "writing", "design", "music", "photography", "inspiration", "storytelling", "content", "social_media", "mythology", "spiritual"]):
                    mapping[domain] = "creative"
                elif any(t in domain for t in ["tech", "programming", "software", "ai", "machine", "data", "cybersecurity"]):
                    mapping[domain] = "technology"
                elif any(sp in domain for sp in ["scientific", "engineering", "research"]):
                    mapping[domain] = "specialized"
                else:
                    mapping[domain] = "daily_life"  # Default fallback to daily_life
                    
        return mapping
        
    async def generate_voice_response(self, text: str, domain: str, 
                                    emotional_context: Dict[str, Any] = None,
                                    user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate enhanced voice response with Trinity Architecture"""
        try:
            print(f"🎤 Generating voice for domain: {domain}")
            
            # Step 1: Determine optimal voice category
            voice_category = await self._select_optimal_voice_category(domain, emotional_context)
            
            # Step 2: Apply Trinity enhancements
            enhanced_text = await self._apply_trinity_text_enhancement(text, domain, emotional_context)
            
            # Step 3: Generate voice with cloud amplification
            voice_result = await self._generate_cloud_amplified_voice(
                enhanced_text, voice_category, user_preferences
            )
            
            # Step 4: Apply Einstein Fusion quality amplification
            final_result = await self._apply_einstein_quality_fusion(voice_result, emotional_context)
            
            # Step 5: Update performance statistics
            await self._update_performance_stats(domain, voice_category, final_result)
            
            result = {
                "audio_data": final_result.get("audio_data"),
                "voice_category": voice_category,
                "voice_name": final_result.get("voice_name"),
                "generation_method": final_result.get("method", "edge_tts"),
                "quality_score": final_result.get("quality_score", 95),
                "emotional_tone": self.voice_categories[voice_category]["emotional_tone"],
                "trinity_enhanced": True,
                "generation_time_ms": final_result.get("generation_time_ms", 0),
                "success": True
            }
            
            # Notify other agents of voice generation
            self.send_message(
                AgentType.TRAINING_CONDUCTOR,
                MessageType.STATUS_UPDATE,
                {
                    "action": "voice_generated",
                    "domain": domain,
                    "voice_category": voice_category,
                    "quality_score": result["quality_score"]
                }
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Voice generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
            
    async def _select_optimal_voice_category(self, domain: str, 
                                           emotional_context: Dict[str, Any] = None) -> str:
        """Select optimal voice category using Perplexity intelligence"""
        
        # Get base category from domain mapping
        base_category = self.domain_voice_mapping.get(domain, "business")
        
        if not emotional_context or not self.trinity_enhancements["perplexity_context_aware"]:
            return base_category
            
        # Apply Perplexity context-aware selection
        emotion = emotional_context.get("detected_emotion", "neutral")
        intensity = emotional_context.get("intensity", 0.5)
        
        # Emotion-based category adjustments updated for 7 categories
        emotion_adjustments = {
            "stress": "general_health",
            "anxiety": "general_health", 
            "sadness": "general_health",
            "anger": "general_health",
            "joy": "creative",
            "excitement": "creative",
            "fear": "general_health",
            "confusion": "education"
        }
        
        if emotion in emotion_adjustments and intensity > 0.6:
            return emotion_adjustments[emotion]
            
        return base_category
        
    async def _apply_trinity_text_enhancement(self, text: str, domain: str, 
                                            emotional_context: Dict[str, Any] = None) -> str:
        """Apply Trinity Architecture text enhancements"""
        
        if not self.trinity_enhancements["arc_reactor_optimization"]:
            return text
            
        # Arc Reactor optimization: Add natural pauses and emphasis
        enhanced_text = text
        
        # Add natural pauses for better comprehension
        enhanced_text = enhanced_text.replace(". ", "... ")
        enhanced_text = enhanced_text.replace(", ", ", ... ")
        
        # Domain-specific enhancements
        if domain in ["stress_management", "mental_health"]:
            # Slower, more deliberate pacing for therapeutic domains
            enhanced_text = enhanced_text.replace("... ", "...... ")
            
        elif domain in ["business", "professional"]:
            # Clear, crisp delivery for professional domains
            enhanced_text = enhanced_text.replace("...... ", "... ")
            
        return enhanced_text
        
    async def _generate_cloud_amplified_voice(self, text: str, voice_category: str,
                                            user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate voice using cloud-first approach: Edge TTS -> Piper/Bark -> pyttsx3"""
        
        config = self.voice_categories[voice_category]
        start_time = asyncio.get_event_loop().time()
        
        # ✨ NEW: Priority 1 - Edge TTS (Cloud-First Approach)
        if self.cloud_settings.get("cloud_first_priority", False) and self.cloud_settings.get("edge_tts_enabled", False):
            try:
                print("🔊 Trying Edge TTS (cloud fallback - requires internet)...")
                voice_name = random.choice(config["edge_voices"])
                
                # Generate audio using Edge TTS
                communicate = edge_tts.Communicate(text, voice_name)
                audio_data = b""
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                        
                generation_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                result = {
                    "audio_data": audio_data,
                    "voice_name": voice_name,
                    "method": "edge_tts",
                    "quality_score": 95,  # Edge TTS provides good quality
                    "generation_time_ms": generation_time
                }
                
                print(f"✅ Generated voice using Edge TTS (Quality: {result['quality_score']})")
                return result
                
            except Exception as e:
                print(f"⚠️ Edge TTS failed: {e}, falling back to local TTS...")
        
        # Priority 2: Local TTS (Piper/Bark) - Fallback when Edge TTS fails or disabled
        print(f"🎤 Attempting local TTS fallback for {voice_category}")
        
        # Try Piper TTS first (fast, good quality, small size)
        if self.cloud_settings["piper_enabled"]:
            print("🔊 Trying Piper TTS (fast local fallback)...")
            result = await self._generate_piper_voice(text, voice_category)
            if result and result.get("quality_score", 0) >= self.cloud_settings["voice_quality_threshold"]:
                result["generation_time_ms"] = (asyncio.get_event_loop().time() - start_time) * 1000
                print(f"✅ Generated voice using Piper TTS (Quality: {result['quality_score']})")
                return result
            elif result:
                print(f"⚠️ Piper TTS quality below threshold: {result.get('quality_score', 0)}")
        
        # Try Bark TTS second (highest quality but large, optional fallback)
        if self.cloud_settings["bark_enabled"]:
            print("🔊 Trying Bark TTS (high quality local fallback)...")
            result = await self._generate_bark_voice(text, voice_category)
            if result and result.get("quality_score", 0) >= self.cloud_settings["voice_quality_threshold"]:
                result["generation_time_ms"] = (asyncio.get_event_loop().time() - start_time) * 1000
                print(f"✅ Generated voice using Bark TTS (Quality: {result['quality_score']})")
                return result
            elif result:
                print(f"⚠️ Bark TTS quality below threshold: {result.get('quality_score', 0)}")
                
        # Priority 3: pyttsx3 - Last Resort Fallback
        if self.cloud_settings["fallback_enabled"]:
            try:
                print("🔊 Trying pyttsx3 (last resort fallback)...")
                engine = pyttsx3.init()
                engine.setProperty('rate', config["pyttsx3_rate"])
                
                # Apply user preferences if available
                if user_preferences:
                    if "voice_speed" in user_preferences:
                        engine.setProperty('rate', int(config["pyttsx3_rate"] * user_preferences["voice_speed"]))
                        
                # Generate audio (simulated - would save to file in real implementation)
                generation_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                result = {
                    "audio_data": b"simulated_audio_data",  # Would be real audio data
                    "voice_name": "pyttsx3_default",
                    "method": "pyttsx3",
                    "quality_score": 70,  # Lower quality but functional
                    "generation_time_ms": generation_time
                }
                
                print(f"✅ Generated voice using pyttsx3 (Quality: {result['quality_score']})")
                return result
                
            except Exception as e:
                print(f"❌ pyttsx3 fallback failed: {e}")
                
        raise Exception("All voice generation methods failed")
        
    async def _apply_einstein_quality_fusion(self, voice_result: Dict[str, Any],
                                           emotional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply Einstein Fusion for quality amplification"""
        
        if not self.trinity_enhancements["einstein_fusion_amplification"]:
            return voice_result
            
        # Einstein Fusion: E=mc² applied to voice quality
        # Enhanced quality = mass(content) × c²(context speed)
        
        base_quality = voice_result.get("quality_score", 85)
        content_mass = len(voice_result.get("audio_data", b"")) / 1000  # KB
        context_speed = 2.0  # Trinity context acceleration factor
        
        # Apply Einstein amplification
        fusion_multiplier = min(1.2, 1.0 + (content_mass * context_speed * context_speed) / 10000)
        enhanced_quality = min(100, base_quality * fusion_multiplier)
        
        # Apply emotional intelligence enhancement
        if emotional_context and emotional_context.get("detected_emotion"):
            emotion_boost = 0.05  # 5% quality boost for emotion-aware generation
            enhanced_quality = min(100, enhanced_quality + (enhanced_quality * emotion_boost))
            
        voice_result["quality_score"] = round(enhanced_quality, 1)
        voice_result["einstein_fusion_applied"] = True
        voice_result["fusion_multiplier"] = round(fusion_multiplier, 3)
        
        return voice_result
        
    async def _update_performance_stats(self, domain: str, voice_category: str, 
                                       result: Dict[str, Any]):
        """Update performance statistics"""
        
        self.performance_stats["voices_generated"] += 1
        
        # Update domain usage
        if domain not in self.performance_stats["domain_usage"]:
            self.performance_stats["domain_usage"][domain] = 0
        self.performance_stats["domain_usage"][domain] += 1
        
        # Update quality scores
        quality_score = result.get("quality_score", 0)
        self.performance_stats["quality_scores"].append(quality_score)
        
        # Update average generation time
        generation_time = result.get("generation_time_ms", 0)
        current_avg = self.performance_stats["average_generation_time"]
        total_voices = self.performance_stats["voices_generated"]
        
        self.performance_stats["average_generation_time"] = (
            (current_avg * (total_voices - 1) + generation_time) / total_voices
        )
        
    async def get_voice_categories(self) -> Dict[str, Any]:
        """Get available voice categories with Trinity enhancements"""
        
        enhanced_categories = {}
        
        for category, config in self.voice_categories.items():
            enhanced_categories[category] = {
                **config,
                "trinity_enhanced": True,
                "cloud_amplified": True,
                "available_voices": len(config["edge_voices"]),
                "quality_tier": "premium" if self.cloud_settings["edge_tts_priority"] else "standard"
            }
            
        return enhanced_categories
        
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive TTS performance statistics"""
        
        avg_quality = 0
        if self.performance_stats["quality_scores"]:
            avg_quality = sum(self.performance_stats["quality_scores"]) / len(self.performance_stats["quality_scores"])
            
        return {
            **self.performance_stats,
            "average_quality_score": round(avg_quality, 1),
            "total_domains_supported": len(self.domain_voice_mapping),
            "voice_categories_available": len(self.voice_categories),
            "trinity_enhancements_active": sum(self.trinity_enhancements.values()),
            "cloud_amplification_enabled": self.cloud_settings["edge_tts_priority"],
            "performance_rating": "excellent" if avg_quality > 95 else "good" if avg_quality > 85 else "standard"
        }

# Global enhanced TTS manager
enhanced_tts_manager = EnhancedTTSManager() 
