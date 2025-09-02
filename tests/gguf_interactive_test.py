#!/usr/bin/env python3
"""
GGUF Interactive Test
Interactive testing of GGUF models with custom prompts
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GGUFInteractiveTester:
    def __init__(self):
        self.model_path = None
        self.test_history = []
        
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
    
    def test_prompt(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> dict:
        """Test a single prompt with the GGUF model"""
        logger.info(f"🧪 Testing prompt: {prompt[:50]}...")
        
        # Create temporary file with prompt
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(prompt)
        temp_file.close()
        
        start_time = time.time()
        
        try:
            # Run llama.cpp with optimized parameters
            cmd = [
                "llama.cpp/build/bin/llama-cli.exe",
                "-m", self.model_path,
                "-f", temp_file.name,
                "-n", str(max_tokens),
                "-t", "12",  # Use all available threads
                "-c", "512",  # Reduced context for speed
                "-b", "64",   # Batch size
                "--temp", str(temperature),
                "--repeat-penalty", "1.1",
                "--no-mmap",  # Disable memory mapping for speed
                "--n-gpu-layers", "0"  # CPU-only
            ]
            
            logger.info(f"🔍 Running command: {' '.join(cmd)}")
            
            # Run with timeout
            timeout = 300  # 5 minutes
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
                
                logger.info(f"✅ Response generated in {processing_time:.2f}s")
                
                test_result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt,
                    "response": assistant_response,
                    "processing_time": processing_time,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "full_output": response_text,
                    "status": "success"
                }
                
                self.test_history.append(test_result)
                return test_result
                
            else:
                logger.warning(f"⚠️ Failed: {result.stderr}")
                return {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt,
                    "error": result.stderr,
                    "return_code": result.returncode,
                    "status": "failed"
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timed out after {timeout}s")
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "error": f"Timeout after {timeout}s",
                "status": "timeout"
            }
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "error": str(e),
                "status": "error"
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
    
    def show_test_history(self):
        """Display test history"""
        if not self.test_history:
            print("📝 No tests run yet.")
            return
        
        print("\n" + "="*80)
        print("📋 TEST HISTORY")
        print("="*80)
        
        for i, test in enumerate(self.test_history, 1):
            status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "error": "💥"}
            emoji = status_emoji.get(test['status'], "❓")
            
            print(f"\n{i}. {emoji} {test['timestamp']}")
            print(f"   Prompt: {test['prompt'][:60]}...")
            
            if test['status'] == 'success':
                print(f"   Response: {test['response'][:80]}...")
                print(f"   Time: {test['processing_time']:.2f}s | Tokens: {test['max_tokens']} | Temp: {test['temperature']}")
            else:
                print(f"   Error: {test['error']}")
        
        print("="*80)
    
    def save_history(self):
        """Save test history to file"""
        if not self.test_history:
            print("📝 No tests to save.")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gguf_interactive_test_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Test history saved to: {filename}")
    
    def run_interactive(self):
        """Run interactive testing session"""
        print("🚀 GGUF Interactive Test Session")
        print("="*50)
        
        # Find model
        try:
            self.model_path = self.find_gguf_model()
            model_size_mb = round(Path(self.model_path).stat().st_size / (1024*1024), 2)
            print(f"🎯 Using model: {Path(self.model_path).name} ({model_size_mb} MB)")
        except Exception as e:
            print(f"❌ Failed to find GGUF model: {e}")
            return
        
        print("\n💡 Available commands:")
        print("   /help     - Show this help")
        print("   /history  - Show test history")
        print("   /save     - Save test history")
        print("   /quit     - Exit testing session")
        print("   /stats    - Show performance statistics")
        print("\n📝 Just type your prompt to test it!")
        
        while True:
            try:
                user_input = input("\n🤖 Enter prompt (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/help':
                        print("\n💡 Available commands:")
                        print("   /help     - Show this help")
                        print("   /history  - Show test history")
                        print("   /save     - Save test history")
                        print("   /quit     - Exit testing session")
                        print("   /stats    - Show performance statistics")
                        print("   /prompt   - Show prompt template")
                    elif user_input == '/history':
                        self.show_test_history()
                    elif user_input == '/save':
                        self.save_history()
                    elif user_input == '/quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input == '/stats':
                        self.show_statistics()
                    elif user_input == '/prompt':
                        print("\n📝 Prompt Template Examples:")
                        print("   Simple: Hello, how are you?")
                        print("   Technical: Explain quantum computing")
                        print("   Creative: Write a haiku about AI")
                        print("   Code: Python function to sort a list")
                        print("   Math: Solve 2x + 5 = 13")
                    else:
                        print(f"❓ Unknown command: {user_input}")
                    continue
                
                # Get parameters
                try:
                    max_tokens = int(input("📊 Max tokens (default 50): ") or "50")
                except ValueError:
                    max_tokens = 50
                
                try:
                    temperature = float(input("🌡️ Temperature (default 0.7): ") or "0.7")
                except ValueError:
                    temperature = 0.7
                
                # Run test
                print(f"\n🧪 Testing with {max_tokens} tokens, temp {temperature}...")
                result = self.test_prompt(user_input, max_tokens, temperature)
                
                if result['status'] == 'success':
                    print(f"\n✅ Response ({result['processing_time']:.2f}s):")
                    print(f"📝 {result['response']}")
                else:
                    print(f"\n❌ Test failed: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_statistics(self):
        """Show performance statistics"""
        if not self.test_history:
            print("📊 No tests run yet.")
            return
        
        successful = [t for t in self.test_history if t['status'] == 'success']
        failed = [t for t in self.test_history if t['status'] != 'success']
        
        if successful:
            avg_time = sum(t['processing_time'] for t in successful) / len(successful)
            avg_tokens = sum(t['max_tokens'] for t in successful) / len(successful)
            avg_temp = sum(t['temperature'] for t in successful) / len(successful)
            
            print(f"\n📊 PERFORMANCE STATISTICS")
            print(f"="*40)
            print(f"Total Tests: {len(self.test_history)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Success Rate: {(len(successful)/len(self.test_history))*100:.1f}%")
            print(f"Average Time: {avg_time:.2f}s")
            print(f"Average Tokens: {avg_tokens:.1f}")
            print(f"Average Temperature: {avg_temp:.2f}")
        else:
            print("📊 No successful tests to analyze.")

def main():
    """Main function"""
    tester = GGUFInteractiveTester()
    tester.run_interactive()

if __name__ == "__main__":
    main()
