#!/usr/bin/env python3
"""
MeeTARA Lab - Services Only Factory (Real Trinity Core Agents)
=============================================================

🎯 PURPOSE: Create real speech and translation models using Trinity Core agents
🚀 ARCHITECTURE: Uses ONLY real agents, no fallback logic
🎤 CREATES: Real emotion detection, voice synthesis, routing, and translation models

REAL TRINITY CORE AGENTS:
✅ SpeechModelsFactory - Real emotion detection, voice synthesis, routing
✅ TranslationFactory - Real translation models for Hindi/Telugu
✅ No fallback - Only real AI models, no configuration files
"""

import asyncio
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path for imports
sys.path.append(str(Path.cwd()))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeeTARAServicesFactory:
    """Real Trinity Core Services Factory - No Fallback Logic"""
    
    def __init__(self):
        """Initialize with real Trinity Core agents only"""
        self.services_dir = Path("services")
        self.speech_factory = None
        self.translation_factory = None
        
        # Initialize real agents
        self._initialize_real_agents()
    
    def _initialize_real_agents(self):
        """Initialize real Trinity Core agents - no fallback"""
        try:
            # Import real Trinity Core agents
            from core.agents.speech_models_factory import SpeechModelsFactory
            from core.agents.translation_factory import TranslationFactory
            
            # Initialize real factories
            self.speech_factory = SpeechModelsFactory()
            self.translation_factory = TranslationFactory()
            
            logger.info("✅ Real Trinity Core agents initialized successfully")
            logger.info(f"   → Speech Factory: {type(self.speech_factory).__name__}")
            logger.info(f"   → Translation Factory: {type(self.translation_factory).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real Trinity Core agents: {e}")
            raise RuntimeError(f"Cannot proceed without real Trinity Core agents: {e}")
    
    async def create_all_services(self) -> Dict[str, Any]:
        """Create all services using real Trinity Core agents"""
        logger.info("🚀 Starting MeeTARA Services Creation with Real Trinity Core...")
        
        start_time = time.time()
        results = {
            "speech_models": {"success": False, "files": [], "count": 0},
            "translation": {"success": False, "files": [], "count": 0}
        }
        
        try:
            # Step 1: Create speech models using real factory
            logger.info("🎤 STEP 1: Creating Real Speech Models...")
            speech_result = await self._create_real_speech_models()
            results["speech_models"] = speech_result
            
            # Step 2: Create translation models using real factory
            logger.info("🌐 STEP 2: Creating Real Translation Models...")
            translation_result = await self._create_real_translation_models()
            results["translation"] = translation_result
            
            # Calculate final results
            total_time = time.time() - start_time
            total_files = sum(r["count"] for r in results.values())
            success_count = sum(1 for r in results.values() if r["success"])
            
            final_result = {
                "success": success_count == len(results),
                "execution_time": round(total_time, 2),
                "services_created": success_count,
                "total_files_created": total_files,
                "success_rate": round((success_count / len(results)) * 100, 1),
                "results": results
            }
            
            logger.info("🎉 MeeTARA Services Creation Complete!")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Services creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "results": results
            }
    
    async def _create_real_speech_models(self) -> Dict[str, Any]:
        """Create real speech models using SpeechModelsFactory"""
        try:
            # Create request for speech models with ALL categories from Trinity config
            request = {
                "emotion_models": True,
                "voice_profiles": True,
                "routing_models": True,
                "categories": [
                    # Main Categories from trinity_config.yaml
                    "healthcare", "daily_life", "business", "education", "creative", 
                    "technology", "specialized", "psychology_wellness", "sports_recreation",
                    "business_professional", "research_academic", "legal_financial", 
                    "emergency_crisis", "aerospace_transportation", "industrial_manufacturing",
                    "travel_tourism"
                ],
                "output_dir": str(self.services_dir)
            }
            
            logger.info("🎤 Creating real speech models using SpeechModelsFactory...")
            
            # Use real factory to create models
            result = await self.speech_factory.create_speech_models(request)
            
            if result.get("success"):
                logger.info(f"✅ Real speech models created successfully")
                logger.info(f"   → Files: {result.get('files_created', 0)}")
                logger.info(f"   → Method: {result.get('method', 'real_factory')}")
                
                return {
                    "success": True,
                    "files": result.get("files_created", []),
                    "count": result.get("files_created", 0),
                    "method": "real_trinity_core"
                }
            else:
                logger.error(f"❌ Speech models creation failed: {result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "files": [],
                    "count": 0,
                    "error": result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Real speech models creation failed: {e}")
            return {
                "success": False,
                "files": [],
                "count": 0,
                "error": str(e)
            }
    
    async def _create_real_translation_models(self) -> Dict[str, Any]:
        """Create real translation models using TranslationFactory"""
        try:
            # Create request for translation models with multiple languages
            languages = ["hi", "te", "es", "fr", "de", "ja", "ko", "zh", "ar", "pt"]  # Hindi, Telugu, Spanish, French, German, Japanese, Korean, Chinese, Arabic, Portuguese
            quantization_type = "Q4_K_M"
            
            logger.info("🌐 Creating real translation models using TranslationFactory...")
            
            # Use real factory to create models
            result = self.translation_factory.create_translation_bundle(languages, quantization_type)
            
            if result.get("success"):
                logger.info(f"✅ Real translation models created successfully")
                logger.info(f"   → Languages: {languages}")
                logger.info(f"   → Files: {result.get('files_created', 0)}")
                
                return {
                    "success": True,
                    "files": result.get("files_created", []),
                    "count": result.get("files_created", 0),
                    "method": "real_trinity_core"
                }
            else:
                logger.error(f"❌ Translation models creation failed: {result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "files": [],
                    "count": 0,
                    "error": result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Real translation models creation failed: {e}")
            return {
                "success": False,
                "files": [],
                "count": 0,
                "error": str(e)
            }
    
    def _create_summary(self, result: Dict[str, Any]) -> str:
        """Create summary of results"""
        if not result["success"]:
            return f"❌ Services creation failed: {result.get('error', 'Unknown error')}"
        
        results = result["results"]
        total_files = sum(r["count"] for r in results.values())
        
        summary = f"""
📊 SERVICES CREATION RESULTS:
   Execution time: {result['execution_time']} seconds
   Services created: {result['services_created']}/{len(results)}
   Total files created: {total_files}
   Success rate: {result['success_rate']}%

📋 SERVICE DETAILS:"""
        
        for service_name, service_result in results.items():
            status = "✅" if service_result["success"] else "❌"
            method = service_result.get("method", "unknown")
            summary += f"\n   {status} {service_name}: {service_result['count']} files ({method})"
        
        summary += f"""

🎉 ALL SERVICES CREATED SUCCESSFULLY!
   📁 Services directory: {self.services_dir.absolute()}
   🚀 Ready for MeeTARA integration!

📋 Detailed results saved: services_creation_results.json"""
        
        return summary

async def main():
    """Main execution function"""
    print("🎤 MeeTARA Services Factory (Real Trinity Core)")
    print("=" * 50)
    print("Creates: Real Speech Models + Real Translation Models")
    print()
    
    try:
        # Initialize factory
        factory = MeeTARAServicesFactory()
        
        # Create all services
        result = await factory.create_all_services()
        
        # Save results
        with open("services_creation_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print(factory._create_summary(result))
        
        return result["success"]
        
    except Exception as e:
        logger.error(f"❌ Factory execution failed: {e}")
        print(f"❌ Factory execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)