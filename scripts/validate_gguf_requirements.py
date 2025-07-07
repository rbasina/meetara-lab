#!/usr/bin/env python3
"""
MeeTARA Lab - GGUF Requirements Validation Script
Simple script to validate all GGUF files against requirements
"""

import asyncio
import sys
from pathlib import Path

# Add tests directory to path
sys.path.append(str(Path(__file__).parent.parent / "tests"))

from integration.test_gguf_quality_validation import ComprehensiveGGUFValidator

async def main():
    """Run GGUF requirements validation"""
    
    print("🧪 MeeTARA Lab - GGUF Requirements Validation")
    print("=" * 50)
    print("Validating all GGUF files in A, B, C, D model variants...")
    print()
    
    validator = ComprehensiveGGUFValidator()
    results = await validator.validate_all_gguf_files()
    
    # Show quick summary
    summary = results["summary"]
    print(f"\n🎯 VALIDATION COMPLETE")
    print(f"📊 Pass Rate: {summary['overall_pass_rate']:.1f}%")
    print(f"✅ Passed: {summary['total_passed']}/{summary['total_files']} files")
    
    if summary["overall_pass_rate"] >= 95:
        print("🎉 EXCELLENT: All models meet requirements!")
    elif summary["overall_pass_rate"] >= 80:
        print("✅ GOOD: Most models meet requirements")
    else:
        print("⚠️ NEEDS WORK: Some models need optimization")
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 