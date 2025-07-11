#!/usr/bin/env python3
"""
MeeTARA Lab - Quick GGUF Model Check
Fast validation of all GGUF models in A, B, C, D directories
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import json

def quick_gguf_check():
    """Quick check of all GGUF models"""
    
    print("🧪 MeeTARA Lab - Quick GGUF Model Check")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Model variant requirements
    requirements = {
        "A_universal_full": {"target_mb": 4600, "tolerance": 0.5},
        "B_universal_lite": {"target_mb": 1200, "tolerance": 0.3},
        "C_category_specific": {"target_mb": 150, "tolerance": 0.4},
        "D_domain_specific": {"target_mb": 8.3, "tolerance": 0.2}
    }
    
    results = {}
    total_files = 0
    total_passed = 0
    
    for variant_name, req in requirements.items():
        print(f"\n📋 Checking {variant_name}")
        
        variant_path = models_dir / variant_name
        
        if not variant_path.exists():
            print(f"   ⚠️ Directory not found")
            results[variant_name] = {"status": "not_found", "files": 0, "passed": 0}
            continue
        
        # Find GGUF files
        gguf_files = []
        if variant_name in ["C_category_specific", "D_domain_specific"]:
            # Check subdirectories
            for subdir in variant_path.iterdir():
                if subdir.is_dir():
                    gguf_files.extend(list(subdir.glob("*.gguf")))
        
        # Also check root directory
        gguf_files.extend(list(variant_path.glob("*.gguf")))
        
        if not gguf_files:
            print(f"   ⚠️ No GGUF files found")
            results[variant_name] = {"status": "no_files", "files": 0, "passed": 0}
            continue
        
        print(f"   🔍 Found {len(gguf_files)} GGUF files")
        
        passed = 0
        for gguf_file in gguf_files:
            # Check file size
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            target = req["target_mb"]
            tolerance = req["tolerance"]
            
            min_size = target * (1 - tolerance)
            max_size = target * (1 + tolerance)
            
            if min_size <= size_mb <= max_size:
                passed += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"      {status} {gguf_file.name}: {size_mb:.1f}MB (target: {target}MB)")
        
        pass_rate = (passed / len(gguf_files)) * 100
        print(f"   📊 Pass rate: {pass_rate:.1f}% ({passed}/{len(gguf_files)})")
        
        results[variant_name] = {
            "status": "checked",
            "files": len(gguf_files),
            "passed": passed,
            "pass_rate": pass_rate
        }
        
        total_files += len(gguf_files)
        total_passed += passed
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY")
    print("=" * 30)
    print(f"Total Files: {total_files}")
    print(f"Total Passed: {total_passed}")
    
    if total_files > 0:
        overall_rate = (total_passed / total_files) * 100
        print(f"Overall Pass Rate: {overall_rate:.1f}%")
        
        if overall_rate >= 95:
            print("🎉 EXCELLENT: All models meet requirements!")
        elif overall_rate >= 80:
            print("✅ GOOD: Most models meet requirements")
        else:
            print("⚠️ NEEDS WORK: Some models need optimization")
    else:
        print("⚠️ No GGUF files found in any variant")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_files == 0:
        print("1. Run enhanced factory: python scripts/gguf_factory/working_enhanced_factory.py")
    elif overall_rate < 80:
        print("1. Review model generation parameters")
        print("2. Check quantization settings")
        print("3. Validate training data quality")
    else:
        print("1. Models look good - ready for production testing!")
        print("2. Consider real inference testing with llama.cpp")
    
    return results

if __name__ == "__main__":
    quick_gguf_check() 