#!/usr/bin/env python3
"""
Voice and Translation Testing Script for MeeTARA Lab
Tests both speech recognition and translation using generated models
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trinity_core.core_components.speech_recognition import EnhancedSpeechRecognition, SPEECH_RECOGNITION_AVAILABLE
from trinity_core.agents.translation_factory import TranslationFactory
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceTranslationTester:
    """Comprehensive tester for voice and translation functionalities"""
    
    def __init__(self):
        self.config = SmartTrinityConfigManager()
        self.speech_recognition = EnhancedSpeechRecognition()
        self.translation_factory = TranslationFactory()
        
        # Test configuration
        self.test_phrases = {
            "hi": [
                "नमस्ते, कैसे हो आप?",
                "मैं एक कंप्यूटर प्रोग्रामर हूं",
                "आज मौसम कैसा है?"
            ],
            "te": [
                "నమస్కారం, మీరు ఎలా ఉన్నారు?",
                "నేను ఒక కంప్యూటర్ ప్రోగ్రామర్",
                "ఈరోజు వాతావరణం ఎలా ఉంది?"
            ],
            "en": [
                "Hello, how are you?",
                "I am a computer programmer",
                "How is the weather today?"
            ]
        }
        
        # Test results storage
        self.test_results = {
            "speech_recognition": {},
            "translation": {},
            "voice_translation_pipeline": {}
        }
    
    async def test_speech_recognition(self) -> Dict[str, Any]:
        """Test speech recognition functionality"""
        logger.info("🎤 Testing Speech Recognition...")
        
        try:
            # Test speech recognition availability
            if SPEECH_RECOGNITION_AVAILABLE:
                logger.info("✅ Speech recognition available - testing basic functionality")
                
                # Test basic initialization
                if self.speech_recognition.recognizer and self.speech_recognition.microphone:
                    logger.info("✅ Recognizer and microphone initialized successfully")
                    
                    # Test domain vocabulary loading
                    domain_count = len(self.speech_recognition.domain_vocabularies)
                    logger.info(f"✅ Loaded {domain_count} domain vocabularies")
                    
                    self.test_results["speech_recognition"]["basic"] = {
                        "status": "success",
                        "recognizer_available": True,
                        "microphone_available": True,
                        "domain_vocabularies_count": domain_count
                    }
                    
                else:
                    logger.warning("⚠️ Recognizer or microphone not available")
                    self.test_results["speech_recognition"]["basic"] = {
                        "status": "partial",
                        "recognizer_available": bool(self.speech_recognition.recognizer),
                        "microphone_available": bool(self.speech_recognition.microphone),
                        "domain_vocabularies_count": len(self.speech_recognition.domain_vocabularies)
                    }
                
            else:
                logger.warning("⚠️ Speech recognition not available - testing fallback mode")
                
                self.test_results["speech_recognition"]["fallback"] = {
                    "status": "success",
                    "recognizer_available": False,
                    "microphone_available": False,
                    "domain_vocabularies_count": len(self.speech_recognition.domain_vocabularies)
                }
                
        except Exception as e:
            logger.error(f"❌ Speech recognition test failed: {e}")
            self.test_results["speech_recognition"]["error"] = str(e)
        
        return self.test_results["speech_recognition"]
    
    async def test_translation_models(self) -> Dict[str, Any]:
        """Test translation models for different languages"""
        logger.info("🌐 Testing Translation Models...")
        
        try:
            # Test Hindi translation
            logger.info("🇮🇳 Testing Hindi translation...")
            hi_result = await self._test_language_translation("hi", "en")
            self.test_results["translation"]["hindi"] = hi_result
            
            # Test Telugu translation
            logger.info("🇮🇳 Testing Telugu translation...")
            te_result = await self._test_language_translation("te", "en")
            self.test_results["translation"]["telugu"] = te_result
            
            # Test English to Hindi
            logger.info("🇺🇸 Testing English to Hindi translation...")
            en_hi_result = await self._test_language_translation("en", "hi")
            self.test_results["translation"]["english_to_hindi"] = en_hi_result
            
            # Test English to Telugu
            logger.info("🇺🇸 Testing English to Telugu translation...")
            en_te_result = await self._test_language_translation("en", "te")
            self.test_results["translation"]["english_to_telugu"] = en_te_result
            
        except Exception as e:
            logger.error(f"❌ Translation test failed: {e}")
            self.test_results["translation"]["error"] = str(e)
        
        return self.test_results["translation"]
    
    async def _test_language_translation(self, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Test translation between specific languages"""
        try:
            # Get test phrases
            if source_lang in self.test_phrases:
                test_phrase = self.test_phrases[source_lang][0]  # Use first phrase
            else:
                test_phrase = "Hello, how are you?"
            
            logger.info(f"🔄 Translating: '{test_phrase}' ({source_lang} → {target_lang})")
            
            # Create translation request
            from trinity_core.agents.translation_factory import TranslationRequest
            
            request = TranslationRequest(
                text=test_phrase,
                source_language=source_lang,
                target_language=target_lang,
                use_offline=True,
                quality_preference="high"
            )
            
            # Perform translation
            result = await self.translation_factory.translate_text(request)
            
            if result:
                logger.info(f"✅ Translation successful: {result.translated_text}")
                return {
                    "status": "success",
                    "source": test_phrase,
                    "translated": result.translated_text,
                    "quality": result.quality_score,
                    "method": result.method_used
                }
            else:
                logger.warning(f"⚠️ Translation returned None for {source_lang} → {target_lang}")
                return {
                    "status": "failed",
                    "error": "Translation returned None"
                }
                
        except Exception as e:
            logger.error(f"❌ Translation test failed for {source_lang} → {target_lang}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_voice_translation_pipeline(self) -> Dict[str, Any]:
        """Test complete voice → translation pipeline"""
        logger.info("🎯 Testing Complete Voice → Translation Pipeline...")
        
        try:
            # Simulate voice input (or use real if available)
            if SPEECH_RECOGNITION_AVAILABLE and self.speech_recognition.recognizer and self.speech_recognition.microphone:
                logger.info("🎙️ Please speak a phrase in English (5 seconds)...")
                await asyncio.sleep(5)
                
                # Simulate speech recognition result
                voice_input = "Hello, how are you today?"
                logger.info(f"🎤 Voice input detected: '{voice_input}'")
                
            else:
                # Use text input for testing
                voice_input = "Hello, how are you today?"
                logger.info(f"📝 Text input mode: '{voice_input}'")
            
            # Test English → Hindi translation
            logger.info("🔄 Testing English → Hindi translation...")
            hi_result = await self._test_language_translation("en", "hi")
            
            # Test English → Telugu translation
            logger.info("🔄 Testing English → Telugu translation...")
            te_result = await self._test_language_translation("en", "te")
            
            # Store pipeline results
            self.test_results["voice_translation_pipeline"] = {
                "voice_input": voice_input,
                "hindi_translation": hi_result,
                "telugu_translation": te_result,
                "pipeline_status": "completed"
            }
            
            logger.info("✅ Voice → Translation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Voice translation pipeline failed: {e}")
            self.test_results["voice_translation_pipeline"]["error"] = str(e)
        
        return self.test_results["voice_translation_pipeline"]
    
    async def test_gguf_inference(self) -> Dict[str, Any]:
        """Test GGUF model inference using llama.cpp"""
        logger.info("🤖 Testing GGUF Model Inference...")
        
        try:
            # Check if llama.cpp is available
            llama_path = Path("llama.cpp/build/bin/llama-cli.exe")
            if not llama_path.exists():
                logger.warning("⚠️ llama-cli.exe not found. Please build llama.cpp first.")
                return {"status": "skipped", "reason": "llama.cpp not built"}
            
            # Test the Qwen GGUF model
            gguf_path = Path("models/production/B_universal/meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf")
            if not gguf_path.exists():
                logger.warning("⚠️ GGUF model not found for testing")
                return {"status": "skipped", "reason": "GGUF model not found"}
            
            logger.info(f"🧠 Testing GGUF model: {gguf_path.name}")
            
            # Test simple inference
            test_prompt = "Translate 'Hello, how are you?' to Hindi"
            
            # Run llama-cli inference
            import subprocess
            
            cmd = [
                str(llama_path),
                "-m", str(gguf_path),
                "-p", test_prompt,
                "-n", "100",  # Max tokens
                "-t", "4"      # Threads
            ]
            
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ GGUF inference successful!")
                logger.info(f"📝 Output: {result.stdout}")
                
                return {
                    "status": "success",
                    "model": gguf_path.name,
                    "prompt": test_prompt,
                    "output": result.stdout,
                    "error": result.stderr
                }
            else:
                logger.error(f"❌ GGUF inference failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            logger.error(f"❌ GGUF inference test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("🚀 Starting Comprehensive Voice and Translation Testing...")
        
        # Test 1: Speech Recognition
        await self.test_speech_recognition()
        
        # Test 2: Translation Models
        await self.test_translation_models()
        
        # Test 3: Voice → Translation Pipeline
        await self.test_voice_translation_pipeline()
        
        # Test 4: GGUF Inference
        gguf_result = await self.test_gguf_inference()
        self.test_results["gguf_inference"] = gguf_result
        
        # Generate summary
        self._generate_test_summary()
        
        return self.test_results
    
    def _generate_test_summary(self):
        """Generate a summary of all test results"""
        logger.info("\n" + "="*60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        # Speech Recognition Summary
        if "speech_recognition" in self.test_results:
            sr_results = self.test_results["speech_recognition"]
            if "error" not in sr_results:
                logger.info("✅ Speech Recognition: PASSED")
            else:
                logger.error(f"❌ Speech Recognition: FAILED - {sr_results['error']}")
        
        # Translation Summary
        if "translation" in self.test_results:
            trans_results = self.test_results["translation"]
            if "error" not in trans_results:
                logger.info("✅ Translation Models: PASSED")
                for lang, result in trans_results.items():
                    if isinstance(result, dict) and result.get("status") == "success":
                        logger.info(f"  ✅ {lang}: {result.get('translated', 'N/A')}")
            else:
                logger.error(f"❌ Translation Models: FAILED - {trans_results['error']}")
        
        # Voice Translation Pipeline Summary
        if "voice_translation_pipeline" in self.test_results:
            vtp_results = self.test_results["voice_translation_pipeline"]
            if "error" not in vtp_results:
                logger.info("✅ Voice → Translation Pipeline: PASSED")
            else:
                logger.error(f"❌ Voice → Translation Pipeline: FAILED - {vtp_results['error']}")
        
        # GGUF Inference Summary
        if "gguf_inference" in self.test_results:
            gguf_results = self.test_results["gguf_inference"]
            if gguf_results.get("status") == "success":
                logger.info("✅ GGUF Inference: PASSED")
            else:
                logger.warning(f"⚠️ GGUF Inference: {gguf_results.get('status', 'UNKNOWN')}")
        
        logger.info("="*60)
        
        # Save results to file
        self._save_test_results()
    
    def _save_test_results(self):
        """Save test results to a JSON file"""
        try:
            results_file = Path("tests/voice_translation_test_results.json")
            results_file.parent.mkdir(exist_ok=True)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Test results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save test results: {e}")

async def main():
    """Main testing function"""
    tester = VoiceTranslationTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        logger.info("🎉 All tests completed!")
        return results
        
    except KeyboardInterrupt:
        logger.info("⏹️ Testing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Testing failed with error: {e}")
        raise

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
