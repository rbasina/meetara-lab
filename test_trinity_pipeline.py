#!/usr/bin/env python3
"""
Test Trinity Pipeline with Correct Data Directory
"""

import sys
from pathlib import Path

# Add paths
sys.path.append('scripts/training')
sys.path.append('model-factory')
sys.path.append('trinity-core')
sys.path.append('scripts/gguf_factory')

from complete_trinity_training_pipeline import CompleteTrinityPipeline

def test_trinity_pipeline():
    """Test the Trinity pipeline with the correct data directory"""
    
    print("🧪 TESTING TRINITY PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = CompleteTrinityPipeline()
    
    # Run with correct data directory (where Trinity generated the data)
    results = pipeline.run_complete_pipeline('data/real')
    
    # Show results
    if results.get('successful_domains', 0) > 0:
        print(f"\n🎉 SUCCESS! Pipeline completed with {results['successful_domains']} successful domains!")
        print(f"   → Success rate: {results['success_rate']:.1f}%")
        print(f"   → Data quality: {results['average_data_quality']:.1f}%")
    else:
        print(f"\n❌ No domains processed successfully")
        print(f"   → Error: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_trinity_pipeline() 