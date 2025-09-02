#!/usr/bin/env python3
"""
GGUF Real-World Scenarios Test
Tests the accuracy and performance of GGUF models across different use cases
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GGUFRealWorldTester:
    def __init__(self):
        self.model_path = None
        self.test_results = {}
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> List[Dict[str, Any]]:
        """Define different real-world test scenarios"""
        return [
            {
                "name": "Simple Conversation",
                "prompt": "Hello, how are you today?",
                "expected_length": "short",
                "category": "basic_chat",
                "max_tokens": 20
            },
            {
                "name": "Technical Question",
                "prompt": "What is the difference between CPU and GPU?",
                "expected_length": "medium",
                "category": "technical",
                "max_tokens": 50
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short story about a robot learning to paint.",
                "expected_length": "long",
                "category": "creative",
                "max_tokens": 100
            },
            {
                "name": "Code Generation",
                "prompt": "Write a Python function to calculate fibonacci numbers.",
                "expected_length": "medium",
                "category": "coding",
                "max_tokens": 60
            },
            {
                "name": "Problem Solving",
                "prompt": "If a train travels 120 km in 2 hours, what is its speed?",
                "expected_length": "short",
                "category": "math",
                "max_tokens": 30
            },
            {
                "name": "Language Translation",
                "prompt": "Translate 'Good morning, how are you?' to Spanish.",
                "expected_length": "short",
                "category": "translation",
                "max_tokens": 20
            },
            {
                "name": "Medical Advice",
                "prompt": "What are the symptoms of dehydration?",
                "expected_length": "medium",
                "category": "healthcare",
                "max_tokens": 50
            },
            {
                "name": "Business Analysis",
                "prompt": "What are the key factors for successful project management?",
                "expected_length": "medium",
                "category": "business",
                "max_tokens": 60
            }
        ]
    
    def find_gguf_model(self) -> str:
        """Find the best available GGUF model for testing"""
        model_priority = [
            "models/production/C_category_specific",  # Smallest (8-50MB)
            "models/production/B_universal",          # Medium (3-4GB)
            "models/production/A_universal_full"      # Largest (4-5GB)
        ]
        
        for model_dir in model_priority:
            if Path(model_dir).exists():
                gguf_files = list(Path(model_dir).glob("*.gguf"))
                if gguf_files:
                    # Prefer Q4_K_M models
                    q4_models = [f for f in gguf_files if "Q4_K_M" in f.name]
                    if q4_models:
                        return str(q4_models[0])
                    return str(gguf_files[0])
        
        raise Exception("No GGUF models found in any production directory")
    
    async def test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single scenario with the GGUF model"""
        logger.info(f"🧪 Testing: {scenario['name']}")
        
        # Create temporary file with prompt
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(scenario['prompt'])
        temp_file.close()
        
        start_time = time.time()
        
        try:
            # Run llama.cpp with scenario-specific parameters
            cmd = [
                "llama.cpp/build/bin/llama-cli.exe",
                "-m", self.model_path,
                "-f", temp_file.name,
                "-n", str(scenario['max_tokens']),
                "-t", "12",  # Use all available threads
                "-c", "512",  # Reduced context for speed
                "-b", "64",   # Batch size
                "--temp", "0.7",  # Balanced creativity
                "--repeat-penalty", "1.1",  # Slight repeat penalty
                "--no-mmap",  # Disable memory mapping for speed
                "--n-gpu-layers", "0"  # CPU-only
            ]
            
            logger.info(f"🔍 Running: {scenario['name']}")
            logger.info(f"📝 Prompt: {scenario['prompt'][:50]}...")
            
            # Run with timeout based on expected length
            timeout_map = {"short": 120, "medium": 180, "long": 300}
            timeout = timeout_map[scenario['expected_length']]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if result.returncode == 0 and result.stdout.strip():
                # Extract assistant response
                response_text = result.stdout.strip()
                assistant_response = self._extract_assistant_response(response_text)
                
                # Analyze response quality
                quality_score = self._analyze_response_quality(
                    scenario, assistant_response, processing_time
                )
                
                logger.info(f"✅ {scenario['name']} completed in {processing_time:.2f}s")
                logger.info(f"📊 Quality Score: {quality_score:.2f}/10")
                
                return {
                    "status": "success",
                    "scenario": scenario['name'],
                    "prompt": scenario['prompt'],
                    "response": assistant_response,
                    "processing_time": processing_time,
                    "quality_score": quality_score,
                    "category": scenario['category'],
                    "expected_length": scenario['expected_length'],
                    "max_tokens": scenario['max_tokens'],
                    "full_output": response_text
                }
                
            else:
                logger.warning(f"⚠️ {scenario['name']} failed: {result.stderr}")
                return {
                    "status": "failed",
                    "scenario": scenario['name'],
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {scenario['name']} timed out after {timeout}s")
            return {
                "status": "timeout",
                "scenario": scenario['name'],
                "timeout_seconds": timeout
            }
            
        except Exception as e:
            logger.error(f"❌ {scenario['name']} failed: {e}")
            return {
                "status": "error",
                "scenario": scenario['name'],
                "error": str(e)
            }
    
    def _extract_assistant_response(self, output: str) -> str:
        """Extract the assistant response from llama.cpp output"""
        if "assistant" in output:
            lines = output.split('\n')
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
            return assistant_response.strip()
        return output.strip()
    
    def _analyze_response_quality(self, scenario: Dict, response: str, processing_time: float) -> float:
        """Analyze the quality of the response (0-10 scale)"""
        score = 5.0  # Base score
        
        # Length appropriateness
        expected_length = scenario['expected_length']
        response_length = len(response.split())
        
        if expected_length == "short" and 5 <= response_length <= 25:
            score += 1.0
        elif expected_length == "medium" and 20 <= response_length <= 60:
            score += 1.0
        elif expected_length == "long" and 50 <= response_length <= 150:
            score += 1.0
        
        # Processing time efficiency
        if processing_time < 60:  # Under 1 minute
            score += 1.0
        elif processing_time < 120:  # Under 2 minutes
            score += 0.5
        
        # Response coherence (basic check)
        if len(response) > 10 and not response.startswith("I don't know"):
            score += 1.0
        
        # Category-specific scoring
        if scenario['category'] == 'coding' and ('def ' in response or 'function' in response):
            score += 1.0
        elif scenario['category'] == 'math' and any(char.isdigit() for char in response):
            score += 1.0
        elif scenario['category'] == 'translation' and len(response.split()) >= 3:
            score += 1.0
        
        return min(10.0, score)
    
    async def run_all_scenarios(self):
        """Run all test scenarios"""
        logger.info("🚀 Starting GGUF Real-World Scenarios Test")
        
        # Find the best available model
        try:
            self.model_path = self.find_gguf_model()
            model_size_mb = round(Path(self.model_path).stat().st_size / (1024*1024), 2)
            logger.info(f"🎯 Using model: {Path(self.model_path).name} ({model_size_mb} MB)")
        except Exception as e:
            logger.error(f"❌ Failed to find GGUF model: {e}")
            return
        
        # Test each scenario
        for scenario in self.scenarios:
            result = await self.test_scenario(scenario)
            self.test_results[scenario['name']] = result
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        # Generate comprehensive report
        self.generate_report()
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("📊 GGUF REAL-WORLD SCENARIOS TEST REPORT")
        logger.info("="*80)
        
        # Calculate statistics
        total_scenarios = len(self.scenarios)
        successful = sum(1 for r in self.test_results.values() if r['status'] == 'success')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        timeouts = sum(1 for r in self.test_results.values() if r['status'] == 'timeout')
        
        # Overall performance
        if successful > 0:
            avg_quality = sum(r['quality_score'] for r in self.test_results.values() if r['status'] == 'success') / successful
            avg_time = sum(r['processing_time'] for r in self.test_results.values() if r['status'] == 'success') / successful
        else:
            avg_quality = 0
            avg_time = 0
        
        logger.info(f"📈 OVERALL PERFORMANCE:")
        logger.info(f"   Total Scenarios: {total_scenarios}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Timeouts: {timeouts}")
        logger.info(f"   Success Rate: {(successful/total_scenarios)*100:.1f}%")
        logger.info(f"   Average Quality Score: {avg_quality:.2f}/10")
        logger.info(f"   Average Processing Time: {avg_time:.2f}s")
        
        # Category performance
        logger.info(f"\n🏷️  PERFORMANCE BY CATEGORY:")
        categories = {}
        for result in self.test_results.values():
            if result['status'] == 'success':
                cat = result['category']
                if cat not in categories:
                    categories[cat] = {'count': 0, 'total_score': 0, 'total_time': 0}
                categories[cat]['count'] += 1
                categories[cat]['total_score'] += result['quality_score']
                categories[cat]['total_time'] += result['processing_time']
        
        for cat, stats in categories.items():
            avg_score = stats['total_score'] / stats['count']
            avg_time = stats['total_time'] / stats['count']
            logger.info(f"   {cat.upper()}: {stats['count']} tests, Avg Score: {avg_score:.2f}/10, Avg Time: {avg_time:.2f}s")
        
        # Detailed results
        logger.info(f"\n📋 DETAILED RESULTS:")
        for scenario_name, result in self.test_results.items():
            status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "error": "💥"}
            emoji = status_emoji.get(result['status'], "❓")
            
            if result['status'] == 'success':
                logger.info(f"   {emoji} {scenario_name}: {result['quality_score']:.2f}/10 in {result['processing_time']:.2f}s")
                logger.info(f"      Response: {result['response'][:100]}...")
            else:
                logger.info(f"   {emoji} {scenario_name}: {result['status']}")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        if avg_quality < 7.0:
            logger.info("   - Consider fine-tuning the model for better accuracy")
        if avg_time > 120:
            logger.info("   - Model is slow for real-time applications")
        if timeouts > 0:
            logger.info("   - Increase timeout or use smaller models for testing")
        
        logger.info("="*80)
        
        # Save detailed results to file
        self.save_results()
    
    def save_results(self):
        """Save detailed results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gguf_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Detailed results saved to: {filename}")

async def main():
    """Main function"""
    tester = GGUFRealWorldTester()
    await tester.run_all_scenarios()

if __name__ == "__main__":
    asyncio.run(main())
