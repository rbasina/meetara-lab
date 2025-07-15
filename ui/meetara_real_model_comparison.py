#!/usr/bin/env python3
"""
🚀 MeeTARA Lab - Real Model Comparison Backend
Lightweight UI backend that loads actual GGUF models for real comparison
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RealModelLoader:
    """Loads and manages real GGUF models"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.llama_cpp = None
        self.loaded_models = {}
        
        logger.info("🏭 Real Model Loader initialized")
        logger.info(f"   📁 Models directory: {self.models_dir}")
        
        # Try to import llama-cpp-python
        self._setup_llama_cpp()
    
    def _setup_llama_cpp(self):
        """Setup llama-cpp-python for real model loading"""
        try:
            from llama_cpp import Llama
            self.llama_cpp = Llama
            logger.info("✅ llama-cpp-python available for real model loading")
        except ImportError:
            logger.warning("⚠️ llama-cpp-python not available, using simulation mode")
            self.llama_cpp = None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available GGUF models, including all D_domain_specific models automatically."""
        models = {
            "A_universal_full": {
                "path": self.models_dir / "production" / "A_universal_full" / "meetara_a_universal_full.gguf",
                "size": "3.5GB",
                "type": "Maximum Intelligence",
                "description": "Qwen 2.5-14B with all 62 domains"
            },
            "B_universal_lite": {
                "path": self.models_dir / "production" / "B_universal_lite" / "meetara_b_universal_lite.gguf",
                "size": "800MB",
                "type": "Universal Speed",
                "description": "Qwen 2.5-7B with all 62 domains"
            },
            "C_category_specific": {
                "path": self.models_dir / "production" / "C_category_specific" / "meetara_healthcare_specialist_v1_Q4.gguf",
                "size": "8.3MB",
                "type": "Healthcare Specialist",
                "description": "Healthcare category specialist"
            }
        }
        # --- Auto-discover D_domain_specific models (only Q4_K_M) ---
        d_root = self.models_dir / "production" / "D_domain_specific"
        if d_root.exists():
            for category_dir in d_root.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    for domain_dir in category_dir.iterdir():
                        if domain_dir.is_dir():
                            domain = domain_dir.name
                            # Only add Q4_K_M quantization
                            for gguf_file in domain_dir.glob(f"{domain}_*Q4_K_M.gguf"):
                                label = f"D_{domain}"
                                models[label] = {
                                    "path": gguf_file,
                                    "size": f"{gguf_file.stat().st_size // (1024*1024)}MB",
                                    "type": "Domain Specialist",
                                    "description": f"{category}/{domain} (Q4_K_M)",
                                    "category": category
                                }
        # --- End auto-discovery ---
        # Check which models actually exist
        available_models = {}
        for name, info in models.items():
            if info["path"].exists():
                info["available"] = True
                info["file_size"] = f"{info['path'].stat().st_size / (1024**2):.1f}MB"
            else:
                info["available"] = False
                info["file_size"] = "Not found"
            available_models[name] = info
        return available_models
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        if model_name in self.loaded_models:
            logger.info(f"✅ Model {model_name} already loaded")
            return True
        
        models = self.get_available_models()
        if model_name not in models or not models[model_name]["available"]:
            logger.error(f"❌ Model {model_name} not available")
            return False
        
        model_path = models[model_name]["path"]
        
        if self.llama_cpp:
            try:
                logger.info(f"📥 Loading real model: {model_name}")
                
                # Configure model parameters based on size
                if "full" in model_name.lower():
                    n_ctx = 4096
                    n_threads = 8
                elif "lite" in model_name.lower():
                    n_ctx = 2048
                    n_threads = 4
                else:
                    n_ctx = 1024
                    n_threads = 2
                
                model = self.llama_cpp(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=False
                )
                
                self.loaded_models[model_name] = {
                    "model": model,
                    "info": models[model_name],
                    "loaded_at": time.time()
                }
                
                logger.info(f"✅ Real model {model_name} loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load real model {model_name}: {e}")
                logger.info(f"🔄 Falling back to simulation mode for {model_name}")
                # Fallback to simulation mode
                self.loaded_models[model_name] = {
                    "model": "simulated",
                    "info": models[model_name],
                    "loaded_at": time.time()
                }
                return True
        else:
            # Simulation mode
            logger.info(f"🎭 Simulating model load: {model_name}")
            self.loaded_models[model_name] = {
                "model": "simulated",
                "info": models[model_name],
                "loaded_at": time.time()
            }
            return True
    
    def generate_response(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Generate response from loaded model"""
        if model_name not in self.loaded_models:
            return {
                "error": f"Model {model_name} not loaded",
                "response": "",
                "metrics": {}
            }
        
        start_time = time.time()
        
        if self.llama_cpp and self.loaded_models[model_name]["model"] != "simulated":
            try:
                # Real model inference
                model = self.loaded_models[model_name]["model"]
                
                response = model(
                    prompt,
                    max_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False
                )
                
                generated_text = response["choices"][0]["text"].strip()
                
                end_time = time.time()
                
                return {
                    "response": generated_text,
                    "model": model_name,
                    "metrics": {
                        "response_time": f"{end_time - start_time:.2f}s",
                        "tokens_generated": len(generated_text.split()),
                        "model_type": "real",
                        "model_size": self.loaded_models[model_name]["info"]["size"]
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Real inference failed for {model_name}: {e}")
                return {
                    "error": f"Inference failed: {e}",
                    "response": "",
                    "metrics": {}
                }
        else:
            # Simulation mode with realistic responses
            end_time = time.time()
            
            # Simulate different response characteristics based on model type
            if "full" in model_name.lower():
                simulated_response = f"[A_universal_full - Maximum Intelligence] Comprehensive analysis: {prompt[:50]}... This requires deep reasoning across multiple domains. Based on the Qwen 2.5-14B base model with 62 specialized domains, I can provide a thorough analysis that considers healthcare implications, business context, technological factors, and human psychology. The optimal approach involves..."
                response_time = 0.8  # Slower but comprehensive
            elif "lite" in model_name.lower():
                simulated_response = f"[B_universal_lite - Universal Speed] Quick response: {prompt[:30]}... Based on the Qwen 2.5-7B base model with universal domain coverage, here's an efficient answer that covers key points across relevant areas. The solution involves..."
                response_time = 0.2  # Fast universal
            else:
                simulated_response = f"[C_universal_category - Specialist] Category-focused: {prompt[:40]}... From the combined expertise of all 7 categories (healthcare, business, specialized, technology, creative, daily_life, education), this requires specialized knowledge. The approach involves..."
                response_time = 0.05  # Ultra-fast specialist
            
            return {
                "response": simulated_response,
                "model": model_name,
                "metrics": {
                    "response_time": f"{response_time:.2f}s",
                    "tokens_generated": len(simulated_response.split()),
                    "model_type": "simulated",
                    "model_size": self.loaded_models[model_name]["info"]["size"]
                }
            }

class SmartRouting:
    """Intelligent routing to select optimal model"""
    
    def __init__(self):
        self.routing_rules = {
            "emergency_keywords": ["emergency", "urgent", "critical", "pain", "help", "doctor"],
            "complex_keywords": ["analyze", "explain", "comprehensive", "detailed", "research"],
            "healthcare_keywords": ["health", "medical", "symptom", "treatment", "medicine", "doctor"]
        }
    
    def analyze_query(self, prompt: str) -> Dict[str, Any]:
        """Analyze query and recommend optimal model"""
        prompt_lower = prompt.lower()
        
        # Emergency detection
        if any(keyword in prompt_lower for keyword in self.routing_rules["emergency_keywords"]):
            return {
                "recommended_model": "C_healthcare_specialist",
                "reason": "Emergency/healthcare query detected",
                "complexity": "urgent",
                "confidence": 0.95
            }
        
        # Healthcare specialization
        if any(keyword in prompt_lower for keyword in self.routing_rules["healthcare_keywords"]):
            return {
                "recommended_model": "C_healthcare_specialist", 
                "reason": "Healthcare specialization needed",
                "complexity": "specialized",
                "confidence": 0.85
            }
        
        # Complex reasoning
        if any(keyword in prompt_lower for keyword in self.routing_rules["complex_keywords"]) or len(prompt) > 200:
            return {
                "recommended_model": "A_universal_full",
                "reason": "Complex reasoning required",
                "complexity": "high",
                "confidence": 0.80
            }
        
        # Default to universal lite
        return {
            "recommended_model": "B_universal_lite",
            "reason": "Standard universal query",
            "complexity": "medium",
            "confidence": 0.70
        }

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global instances
model_loader = RealModelLoader()
smart_routing = SmartRouting()

@app.route('/')
def index():
    """Main comparison interface"""
    return render_template('real_model_comparison.html')

@app.route('/api/models')
def get_models():
    models = model_loader.get_available_models()
    # Convert all Path objects to strings for JSON serialization
    for m in models.values():
        if isinstance(m.get("path"), Path):
            m["path"] = str(m["path"])
    return jsonify(models)

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load a specific model"""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    success = model_loader.load_model(model_name)
    
    return jsonify({
        "success": success,
        "message": f"Model {model_name} {'loaded' if success else 'failed to load'}"
    })

@app.route('/api/analyze_query', methods=['POST'])
def analyze_query():
    """Analyze query and recommend model"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    analysis = smart_routing.analyze_query(prompt)
    return jsonify(analysis)

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare responses from multiple models"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    models = data.get('models', ['A_universal_full', 'B_universal_lite', 'C_healthcare_specialist'])
    
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    
    # Get smart routing recommendation
    routing_analysis = smart_routing.analyze_query(prompt)
    
    results = {
        "prompt": prompt,
        "routing_analysis": routing_analysis,
        "responses": {},
        "comparison_metrics": {}
    }
    
    # Generate responses from each model
    total_start_time = time.time()
    
    for model_name in models:
        # Load model if not already loaded
        if model_name not in model_loader.loaded_models:
            model_loader.load_model(model_name)
        
        # Generate response
        response_data = model_loader.generate_response(model_name, prompt)
        results["responses"][model_name] = response_data
    
    total_end_time = time.time()
    
    # Calculate comparison metrics
    response_times = [float(results["responses"][model]["metrics"].get("response_time", "0").replace("s", "")) 
                     for model in models if "metrics" in results["responses"][model]]
    
    results["comparison_metrics"] = {
        "total_time": f"{total_end_time - total_start_time:.2f}s",
        "fastest_model": min(models, key=lambda m: float(results["responses"][m]["metrics"].get("response_time", "999").replace("s", ""))) if response_times else "unknown",
        "recommended_model": routing_analysis["recommended_model"],
        "models_compared": len(models)
    }
    
    return jsonify(results)

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        "loaded_models": list(model_loader.loaded_models.keys()),
        "llama_cpp_available": model_loader.llama_cpp is not None,
        "total_models_available": len([m for m in model_loader.get_available_models().values() if m["available"]]),
        "system_ready": True
    })

if __name__ == '__main__':
    logger.info("🚀 Starting MeeTARA Real Model Comparison Server...")
    logger.info("🌐 Access at: http://localhost:5001")
    logger.info("🤖 Real GGUF model comparison ready!")
    
    app.run(host='0.0.0.0', port=5001, debug=True) 