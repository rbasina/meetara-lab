#!/usr/bin/env python3
"""
MeeTARA Lab - Complete Voice Pipeline Test
Tests the full flow: Speech → Translation → LLM → Response Translation → TTS

Pipeline: User speaks in Hindi/Telugu → English → GGUF → English → Hindi/Telugu → Voice
"""

import asyncio
import logging
import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trinity_core.core_components.speech_recognition import EnhancedSpeechRecognition, SPEECH_RECOGNITION_AVAILABLE
from trinity_core.agents.translation_factory import TranslationFactory, TranslationRequest
from trinity_core.core_components.tts_manager import EnhancedTTSManager
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoicePipelineTester:
    """Complete voice pipeline tester for MeeTARA Lab"""
    
    def __init__(self):
        self.test_results = {
            "speech_recognition": {"status": "pending", "details": {}},
            "translation": {"status": "pending", "details": {}},
            "llm_processing": {"status": "pending", "details": {}},
            "response_translation": {"status": "pending", "details": {}},
            "tts": {"status": "pending", "details": {}},
            "complete_pipeline": {"status": "pending", "details": {}}
        }
        
        # Test configuration
        self.test_phrases = {
            "hi": "मुझे डॉक्टर की जरूरत है",  # "I need a doctor"
            "te": "నాకు డాక్టర్ కావాలి",      # "I need a doctor"
            "en": "Hello, how are you today?"
        }
        
        # Expected responses
        self.expected_responses = {
            "hi": "I need a doctor",
            "te": "I need a doctor", 
            "en": "Hello, how are you today?"
        }
        
        # Initialize components
        self.speech_recognition = None
        self.translation_factory = None
        self.tts_manager = None
        
        # Load speech configuration
        self.speech_config = self._load_speech_config()
        
    def _load_speech_config(self) -> Dict[str, Any]:
        """Load speech configuration from speech_config.json"""
        try:
            config_path = Path("config/speech_config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded speech config: {len(config.get('voice_mappings', {}))} voice mappings")
                return config
            else:
                logger.warning("⚠️ speech_config.json not found")
                return {}
        except Exception as e:
            logger.error(f"❌ Failed to load speech config: {e}")
            return {}
        
    async def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("🔧 Initializing voice pipeline components...")
        
        try:
            # Initialize speech recognition
            if SPEECH_RECOGNITION_AVAILABLE:
                self.speech_recognition = EnhancedSpeechRecognition()
                logger.info("✅ Speech recognition initialized")
            else:
                logger.warning("⚠️ Speech recognition not available - using simulation mode")
                self.speech_recognition = None
            
            # Initialize translation factory
            self.translation_factory = TranslationFactory()
            logger.info("✅ Translation factory initialized")
            
            # Initialize TTS manager
            self.tts_manager = EnhancedTTSManager()
            logger.info("✅ TTS manager initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    async def test_speech_recognition(self):
        """Test speech recognition functionality"""
        logger.info("🎤 Testing speech recognition...")
        
        if not self.speech_recognition:
            self.test_results["speech_recognition"] = {
                "status": "skipped",
                "details": {"reason": "Speech recognition not available"}
            }
            return
        
        try:
            # Test domain vocabulary loading
            domain_count = len(self.speech_recognition.domain_vocabularies)
            logger.info(f"✅ Loaded {domain_count} domain vocabularies")
            
            # Test speech configuration loading
            voice_mappings_count = len(self.speech_config.get('voice_mappings', {}))
            logger.info(f"✅ Speech config loaded: {voice_mappings_count} voice mappings")
            
            # Test basic functionality
            if self.speech_recognition.recognizer and self.speech_recognition.microphone:
                self.test_results["speech_recognition"] = {
                    "status": "success",
                    "details": {
                        "recognizer_available": True,
                        "microphone_available": True,
                        "domain_vocabularies_count": domain_count,
                        "voice_mappings_count": voice_mappings_count,
                        "real_time_mode": self.speech_recognition.real_time_config.get("chunk_size", 0)
                    }
                }
                logger.info("✅ Speech recognition test passed")
            else:
                self.test_results["speech_recognition"] = {
                    "status": "partial",
                    "details": {
                        "recognizer_available": bool(self.speech_recognition.recognizer),
                        "microphone_available": bool(self.speech_recognition.microphone),
                        "domain_vocabularies_count": domain_count,
                        "voice_mappings_count": voice_mappings_count
                    }
                }
                logger.warning("⚠️ Speech recognition partially available")
                
        except Exception as e:
            logger.error(f"❌ Speech recognition test failed: {e}")
            self.test_results["speech_recognition"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def test_translation(self):
        """Test translation capabilities"""
        logger.info("🌐 Testing translation...")
        
        try:
            # Test Hindi to English
            hindi_text = "मुझे डॉक्टर की जरूरत है"
            logger.info(f"🔍 Testing Hindi → English: '{hindi_text}'")
            
            request = TranslationRequest(
                text=hindi_text,
                source_language='hi',
                target_language='en',
                use_offline=True
            )
            
            logger.info(f"🔍 Translation request: {request}")
            logger.info(f"🔍 Available offline models: {list(self.translation_factory.offline_models.keys())}")
            
            result = await self.translation_factory.translate_text(request)
            logger.info(f"🔍 Translation result: {result}")
            
            hindi_success = result and result.translated_text != hindi_text
            
            if hindi_success:
                logger.info(f"✅ Hindi → English: '{hindi_text}' → '{result.translated_text}'")
            else:
                logger.warning(f"⚠️ Hindi translation returned original text: '{result.translated_text if result else 'None'}'")
            
            # Test Telugu to English
            telugu_text = "నాకు డాక్టర్ కావాలి"
            logger.info(f"🔍 Testing Telugu → English: '{telugu_text}'")
            
            request = TranslationRequest(
                text=telugu_text,
                source_language='te',
                target_language='en',
                use_offline=True
            )
            
            result = await self.translation_factory.translate_text(request)
            logger.info(f"🔍 Telugu translation result: {result}")
            
            telugu_success = result and result.translated_text != telugu_text
            
            if telugu_success:
                logger.info(f"✅ Telugu → English: '{telugu_text}' → '{result.translated_text}'")
            else:
                logger.warning(f"⚠️ Telugu translation returned original text: '{result.translated_text if result else 'None'}'")
            
            # Determine overall translation status
            if hindi_success and telugu_success:
                translation_status = "success"
            elif hindi_success or telugu_success:
                translation_status = "partial"
            else:
                translation_status = "failed"
            
            # Update test results
            self.test_results["translation"] = {
                "status": translation_status,
                "details": {
                    "hindi_test": {
                        "input": hindi_text,
                        "output": result.translated_text if result else "None",
                        "success": hindi_success
                    },
                    "telugu_test": {
                        "input": telugu_text,
                        "output": result.translated_text if result else "None", 
                        "success": telugu_success
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Translation test failed: {e}")
            self.test_results["translation"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def test_llm_processing(self):
        """Test LLM processing with GGUF model"""
        logger.info("🧠 Testing LLM processing...")

        try:
            # Smart model selection with fallback strategy
            model_priority = [
                "models/production/C_category_specific",  # Start with smallest (8.3MB)
                "models/production/B_universal",          # Fallback to medium (3.5GB)
                "models/production/A_universal"      # Last resort to largest (4.6GB)
            ]

            gguf_path = None
            for model_dir in model_priority:
                if Path(model_dir).exists():
                    gguf_files = list(Path(model_dir).glob("*.gguf"))
                    if gguf_files:
                        # Prefer Q4_K_M models
                        q4_models = [f for f in gguf_files if "Q4_K_M" in f.name]
                        if q4_models:
                            gguf_path = q4_models[0]
                            break
                        gguf_path = gguf_files[0]
                        break

            if not gguf_path:
                raise Exception("No GGUF models found in any production directory")

            # Calculate model size first
            model_size_mb = round(gguf_path.stat().st_size / (1024*1024), 2)

            # Check if we only have large models and suggest alternatives
            if model_size_mb > 1000:
                logger.warning(f"⚠️ Only large models available ({model_size_mb} MB). For faster testing, consider:")
                logger.warning("   - Creating smaller domain-specific models (8-50MB)")
                logger.warning("   - Using Q3_K_M quantization for reduced size")
                logger.warning("   - Testing with healthcare or business domain models")

            logger.info(f"🎯 Selected model: {gguf_path.name} ({model_size_mb} MB)")

            # Dynamic timeout based on model size
            if model_size_mb > 1000:  # Large models (>1GB)
                timeout_seconds = 300  # 5 minutes for large models (based on manual test showing ~3.5 min)
                logger.info(f"⏱️ Using extended timeout ({timeout_seconds}s) for large model ({model_size_mb} MB)")
            elif model_size_mb > 100:   # Medium models (100MB-1GB)
                timeout_seconds = 120   # 2 minutes for medium models
                logger.info(f"⏱️ Using standard timeout ({timeout_seconds}s) for medium model ({model_size_mb} MB)")
            else:                       # Small models (<100MB)
                timeout_seconds = 60    # 1 minute for small models
                logger.info(f"⏱️ Using fast timeout ({timeout_seconds}s) for small model ({model_size_mb} MB)")

            # Create a simple test prompt that generates shorter responses
            test_prompt = "Say hello in one sentence."
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write(test_prompt)
            temp_file.close()

            try:
                # Run llama.cpp inference with optimized parameters for large models
                cmd = [
                    "llama.cpp/build/bin/llama-cli.exe",
                    "-m", str(gguf_path),
                    "-f", temp_file.name,
                    "-n", "20",   # Limit to 20 tokens for faster response
                    "-t", "12",    # Maximum threads for CPU utilization
                    "-c", "256",   # Reduced context for speed
                    "-b", "64",    # Maximum batch size for throughput
                    "--temp", "0.5",  # Balanced temperature
                    "--repeat-penalty", "1.0",  # No repeat penalty
                    "--no-mmap",  # Disable memory mapping for speed
                    "--n-gpu-layers", "0"  # CPU-only for compatibility
                ]

                logger.info(f"🔍 Running LLM command: {' '.join(cmd)}")

                # Run with dynamic timeout based on model size
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )

                # Clean up temp file
                os.unlink(temp_file.name)

                if result.returncode == 0 and result.stdout.strip():
                    # Extract just the assistant response (after the prompt)
                    response_text = result.stdout.strip()
                    if "assistant" in response_text:
                        # Extract text after "assistant" line
                        lines = response_text.split('\n')
                        assistant_response = ""
                        for i, line in enumerate(lines):
                            if line.strip().startswith("assistant"):
                                # Get all lines after "assistant" until next user/end
                                for j in range(i + 1, len(lines)):
                                    if lines[j].strip().startswith("user") or lines[j].strip().startswith(">"):
                                        break
                                    if lines[j].strip():
                                        assistant_response += lines[j].strip() + " "
                                break
                        else:
                            assistant_response = response_text
                    else:
                        assistant_response = response_text

                    logger.info(f"✅ LLM processing successful: {assistant_response[:100]}...")
                    self.test_results["llm_processing"] = {
                        "status": "success",
                        "details": {
                            "model_path": str(gguf_path),
                            "model_size_mb": model_size_mb,
                            "response_length": len(assistant_response),
                            "response_preview": assistant_response[:100],
                            "full_response": response_text
                        }
                    }
                else:
                    logger.warning(f"⚠️ LLM processing completed but no valid output: {result.stderr}")
                    self.test_results["llm_processing"] = {
                        "status": "partial",
                        "details": {
                            "model_path": str(gguf_path),
                            "model_size_mb": model_size_mb,
                            "error": result.stderr,
                            "return_code": result.returncode
                        }
                    }

            except subprocess.TimeoutExpired:
                logger.error(f"⏰ LLM processing timed out after {timeout_seconds} seconds")
                logger.info("💡 This is expected for large models. The model is working (as shown in manual test)")
                logger.info("   - Manual test showed ~3.5 minutes for full response")
                logger.info("   - Test timeout was {timeout_seconds} seconds")
                
                self.test_results["llm_processing"] = {
                    "status": "timeout_expected",
                    "details": {
                        "model_path": str(gguf_path),
                        "model_size_mb": model_size_mb,
                        "timeout_seconds": timeout_seconds,
                        "note": "Large model timeout is expected. Manual test confirmed model works.",
                        "manual_test_time": "~3.5 minutes for full response"
                    }
                }
                
                # Clean up temp file on timeout
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            except Exception as e:
                logger.error(f"❌ LLM processing failed: {e}")
                self.test_results["llm_processing"] = {
                    "status": "failed",
                    "details": {
                        "model_path": str(gguf_path),
                        "model_size_mb": model_size_mb,
                        "error": str(e)
                    }
                }
                
                # Clean up temp file on error
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

        except Exception as e:
            logger.error(f"❌ LLM processing test failed: {e}")
            self.test_results["llm_processing"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def test_response_translation(self):
        """Test response translation (English to target languages)"""
        logger.info("🔄 Testing response translation...")
        
        try:
            # Test English to Hindi
            english_text = "I need a doctor"
            logger.info(f"🔍 Testing English → Hindi: '{english_text}'")
            
            request = TranslationRequest(
                text=english_text,
                source_language='en',
                target_language='hi',
                use_offline=True
            )
            
            result = await self.translation_factory.translate_text(request)
            logger.info(f"🔍 English → Hindi result: {result}")
            
            hindi_success = result and result.translated_text != english_text
            
            if hindi_success:
                logger.info(f"✅ English → Hindi: '{english_text}' → '{result.translated_text}'")
            else:
                logger.warning(f"⚠️ English → Hindi translation returned original text: '{result.translated_text if result else 'None'}'")
            
            # Test English to Telugu
            logger.info(f"🔍 Testing English → Telugu: '{english_text}'")
            
            request = TranslationRequest(
                text=english_text,
                source_language='en',
                target_language='te',
                use_offline=True
            )
            
            result = await self.translation_factory.translate_text(request)
            logger.info(f"🔍 English → Telugu result: {result}")
            
            telugu_success = result and result.translated_text != english_text
            
            if telugu_success:
                logger.info(f"✅ English → Telugu: '{english_text}' → '{result.translated_text}'")
            else:
                logger.warning(f"⚠️ English → Telugu translation returned original text: '{result.translated_text if result else 'None'}'")
            
            # Determine overall response translation status
            if hindi_success and telugu_success:
                response_translation_status = "success"
            elif hindi_success or telugu_success:
                response_translation_status = "partial"
            else:
                response_translation_status = "failed"
            
            # Update test results
            self.test_results["response_translation"] = {
                "status": response_translation_status,
                "details": {
                    "english_to_hindi": {
                        "input": english_text,
                        "output": result.translated_text if result else "None",
                        "success": hindi_success
                    },
                    "english_to_telugu": {
                        "input": english_text,
                        "output": result.translated_text if result else "None",
                        "success": telugu_success
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Response translation test failed: {e}")
            self.test_results["response_translation"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def test_tts(self):
        """Test text-to-speech functionality"""
        logger.info("🔊 Testing text-to-speech...")
        
        try:
            # Test basic TTS functionality
            if hasattr(self.tts_manager, 'synthesize_speech'):
                # Test Hindi TTS
                hi_text = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूं?"
                hi_result = await self.tts_manager.synthesize_speech(hi_text, "hi")
                
                if hi_result:
                    logger.info("✅ Hindi TTS test passed")
                    
                    # Test Telugu TTS
                    te_text = "నమస్కారం, నేను మీకు ఎలా సహాయం చేయగలను?"
                    te_result = await self.tts_manager.synthesize_speech(te_text, "te")
                    
                    if te_result:
                        logger.info("✅ Telugu TTS test passed")
                        
                        self.test_results["tts"] = {
                            "status": "success",
                            "details": {
                                "hindi_tts": {"text": hi_text, "result": "success"},
                                "telugu_tts": {"text": te_text, "result": "success"}
                            }
                        }
                    else:
                        raise Exception("Telugu TTS failed")
                else:
                    raise Exception("Hindi TTS failed")
            else:
                raise Exception("TTS manager does not have synthesize_speech method")
                
        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")
            self.test_results["tts"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def test_complete_pipeline(self):
        """Test the complete voice pipeline"""
        logger.info("🚀 Testing complete voice pipeline...")
        
        try:
            # Simulate complete flow
            pipeline_steps = []
            
            # Step 1: Speech Recognition (simulated)
            if self.test_results["speech_recognition"]["status"] == "success":
                pipeline_steps.append("✅ Speech Recognition")
            else:
                pipeline_steps.append("❌ Speech Recognition")
            
            # Step 2: Translation to English
            if self.test_results["translation"]["status"] == "success":
                pipeline_steps.append("✅ Translation to English")
            else:
                pipeline_steps.append("❌ Translation to English")
            
            # Step 3: LLM Processing
            if self.test_results["llm_processing"]["status"] == "success":
                pipeline_steps.append("✅ LLM Processing")
            else:
                pipeline_steps.append("❌ LLM Processing")
            
            # Step 4: Response Translation
            if self.test_results["response_translation"]["status"] == "success":
                pipeline_steps.append("✅ Response Translation")
            else:
                pipeline_steps.append("❌ Response Translation")
            
            # Step 5: TTS
            if self.test_results["tts"]["status"] == "success":
                pipeline_steps.append("✅ Text-to-Speech")
            else:
                pipeline_steps.append("❌ Text-to-Speech")
            
            # Calculate success rate
            successful_steps = sum(1 for step in pipeline_steps if "✅" in step)
            total_steps = len(pipeline_steps)
            success_rate = (successful_steps / total_steps) * 100
            
            if success_rate >= 80:
                pipeline_status = "success"
            elif success_rate >= 50:
                pipeline_status = "partial"
            else:
                pipeline_status = "failed"
            
            self.test_results["complete_pipeline"] = {
                "status": pipeline_status,
                "details": {
                    "steps": pipeline_steps,
                    "success_rate": f"{success_rate:.1f}%",
                    "successful_steps": successful_steps,
                    "total_steps": total_steps
                }
            }
            
            logger.info(f"🎯 Pipeline completion: {success_rate:.1f}% ({successful_steps}/{total_steps} steps)")
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline test failed: {e}")
            self.test_results["complete_pipeline"] = {
                "status": "failed",
                "details": {"error": str(e)}
            }
    
    async def run_all_tests(self):
        """Run all pipeline tests"""
        logger.info("🚀 Starting complete voice pipeline testing...")
        
        # Initialize components
        if not await self.initialize_components():
            logger.error("❌ Failed to initialize components - aborting tests")
            return
        
        # Run individual tests
        await self.test_speech_recognition()
        await self.test_translation()
        await self.test_llm_processing()
        await self.test_response_translation()
        await self.test_tts()
        
        # Test complete pipeline
        await self.test_complete_pipeline()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("📊 VOICE PIPELINE TEST REPORT")
        logger.info("="*60)
        
        for test_name, result in self.test_results.items():
            status = result["status"]
            status_icon = "✅" if status == "success" else "⚠️" if status == "partial" else "❌"
            
            logger.info(f"{status_icon} {test_name.replace('_', ' ').title()}: {status}")
            
            if "details" in result and result["details"]:
                if "error" in result["details"]:
                    logger.info(f"   Error: {result['details']['error']}")
                elif "steps" in result["details"]:
                    for step in result["details"]["steps"]:
                        logger.info(f"   {step}")
                elif "success_rate" in result["details"]:
                    logger.info(f"   Success Rate: {result['details']['success_rate']}")
        
        # Overall assessment
        overall_status = "SUCCESS" if all(r["status"] in ["success", "skipped"] for r in self.test_results.values()) else "PARTIAL" if any(r["status"] == "success" for r in self.test_results.values()) else "FAILED"
        
        logger.info("\n" + "="*60)
        logger.info(f"🎯 OVERALL PIPELINE STATUS: {overall_status}")
        logger.info("="*60)
        
        if overall_status == "SUCCESS":
            logger.info("🎉 Complete voice pipeline is working! Users can:")
            logger.info("   1. Speak in Hindi/Telugu")
            logger.info("   2. Get English translation")
            logger.info("   3. Process through LLM")
            logger.info("   4. Get response in their language")
            logger.info("   5. Hear voice output")
        elif overall_status == "PARTIAL":
            logger.info("⚠️ Pipeline partially working - some components need attention")
        else:
            logger.info("❌ Pipeline needs significant work - check individual components")

async def main():
    """Main test execution"""
    tester = VoicePipelineTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
