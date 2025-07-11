#!/usr/bin/env python3
"""
Create a minimal test GGUF file to verify UI functionality
"""

import os
import struct
from pathlib import Path

def create_minimal_gguf(output_path: str, model_name: str = "test_model"):
    """Create a minimal valid GGUF file for testing"""
    
    # GGUF file format starts with magic bytes "GGUF"
    magic = b"GGUF"
    version = struct.pack("<I", 3)  # Version 3
    
    # Minimal metadata
    metadata_count = struct.pack("<Q", 1)  # 1 metadata entry
    
    # Model name metadata
    key = "general.name"
    key_length = struct.pack("<Q", len(key))
    key_bytes = key.encode('utf-8')
    
    value_type = struct.pack("<I", 8)  # String type
    value = model_name
    value_length = struct.pack("<Q", len(value))
    value_bytes = value.encode('utf-8')
    
    # Tensor count (0 for minimal test)
    tensor_count = struct.pack("<Q", 0)
    
    # Write the file
    with open(output_path, 'wb') as f:
        f.write(magic)
        f.write(version)
        f.write(metadata_count)
        f.write(key_length)
        f.write(key_bytes)
        f.write(value_type)
        f.write(value_length)
        f.write(value_bytes)
        f.write(tensor_count)
    
    print(f"✅ Created minimal GGUF file: {output_path}")
    print(f"   Size: {os.path.getsize(output_path)} bytes")

def main():
    """Create test GGUF files"""
    models_dir = Path("../models")
    
    # Create test files
    test_models = [
        ("A_universal_full", "MeeTARA A Universal Full"),
        ("B_universal_lite", "MeeTARA B Universal Lite"),
        ("C_category_specific", "MeeTARA C Category Specific")
    ]
    
    for model_dir, model_name in test_models:
        model_path = models_dir / model_dir
        model_path.mkdir(exist_ok=True)
        
        gguf_file = model_path / f"meetara_{model_dir.lower()}.gguf"
        
        # Only create if doesn't exist or is very large (likely corrupted)
        if not gguf_file.exists() or gguf_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
            create_minimal_gguf(str(gguf_file), model_name)
        else:
            print(f"✅ {gguf_file} already exists and is reasonable size")

if __name__ == "__main__":
    main() 