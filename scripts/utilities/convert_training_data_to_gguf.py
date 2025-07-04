#!/usr/bin/env python3
"""
Convert Training Data JSON to GGUF Models
Converts the generated training data JSON files into actual GGUF model files
"""

import os
import json
import sys
from pathlib import Path

# Add model-factory to path
sys.path.append(str(Path(__file__).parent / "model-factory" / "02_gguf_creation"))

from gguf_factory import TrinityGGUFFactory

def convert_training_data_to_gguf():
    """Convert all training data JSON files to GGUF models"""
    
    print("🚀 Converting Training Data to GGUF Models")
    print("=" * 50)
    
    # Initialize Trinity GGUF Factory
    factory = TrinityGGUFFactory()
    
    # Find all training data JSON files
    training_data_dir = Path("model-factory/real_training_data")
    
    if not training_data_dir.exists():
        print("❌ Training data directory not found!")
        return
    
    # Process each category
    results = {}
    total_models = 0
    
    for category_dir in training_data_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        print(f"\n📂 Processing category: {category_name}")
        
        # Process each domain in category
        for json_file in category_dir.glob("*_training_data.json"):
            domain = json_file.stem.replace("_training_data", "")
            
            print(f"   🔄 Converting {domain}...")
            
            # Load training data
            with open(json_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # Create GGUF model
            try:
                model_result = factory.create_gguf_model(domain, training_data)
                results[domain] = model_result
                total_models += 1
                print(f"   ✅ {domain} → {model_result.get('estimated_size', '8.3MB')} GGUF")
                
            except Exception as e:
                print(f"   ❌ Error creating {domain}: {e}")
                results[domain] = {"error": str(e)}
    
    print(f"\n🎉 Conversion Complete!")
    print(f"   → Total models created: {total_models}")
    print(f"   → Output directory: model-factory/trinity_gguf_models/")
    
    # Show final structure
    print(f"\n📁 Final GGUF Model Structure:")
    gguf_dir = Path("model-factory/trinity_gguf_models")
    
    if gguf_dir.exists():
        for item in sorted(gguf_dir.rglob("*.gguf")):
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name} ({size_mb:.1f}MB)")
    
    return results

if __name__ == "__main__":
    results = convert_training_data_to_gguf() 