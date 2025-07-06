#!/usr/bin/env python3
"""
Quick Training Results Analysis
"""

import json
import os
import glob
from collections import defaultdict

def analyze_training_results():
    print("🚀 MeeTARA Lab Training Results - Quick Analysis")
    print("="*60)
    
    # Load session summaries
    pattern = os.path.join("logs", "session_summary_*.json")
    files = glob.glob(pattern)
    
    print(f"📁 Found {len(files)} session summary files")
    
    results = []
    categories = defaultdict(list)
    tiers = defaultdict(list)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
                
                # Categorize
                category = data.get('domain_validation', {}).get('category', 'unknown')
                categories[category].append(data)
                
                tier = data.get('model_selection', {}).get('model_tier', 'unknown')
                tiers[tier].append(data)
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"✅ Loaded {len(results)} training sessions")
    
    # Quality Analysis
    print("\n📊 QUALITY METRICS:")
    quality_scores = [r.get('quality_metrics', {}).get('quality_score', 0) for r in results]
    passed_count = sum(1 for r in results if r.get('quality_metrics', {}).get('passed', False))
    
    print(f"   Total Domains: {len(results)}")
    print(f"   Passed: {passed_count} ({passed_count/len(results)*100:.1f}%)")
    print(f"   Avg Quality: {sum(quality_scores)/len(quality_scores):.2f}%")
    print(f"   Range: {min(quality_scores):.2f}% - {max(quality_scores):.2f}%")
    
    # Category Analysis
    print("\n📂 CATEGORIES:")
    for category, cat_results in categories.items():
        if cat_results:
            cat_quality = [r.get('quality_metrics', {}).get('quality_score', 0) for r in cat_results]
            avg_quality = sum(cat_quality) / len(cat_quality)
            print(f"   {category}: {len(cat_results)} domains, {avg_quality:.2f}% avg quality")
    
    # Tier Analysis
    print("\n🏗️ MODEL TIERS:")
    for tier, tier_results in tiers.items():
        if tier_results:
            tier_quality = [r.get('quality_metrics', {}).get('quality_score', 0) for r in tier_results]
            avg_quality = sum(tier_quality) / len(tier_quality)
            models = set(r.get('model_selection', {}).get('base_model', '') for r in tier_results)
            print(f"   {tier}: {len(tier_results)} domains, {avg_quality:.2f}% avg quality")
            print(f"      Models: {', '.join(models)}")
    
    # GGUF Analysis
    print("\n🏭 GGUF CREATION:")
    gguf_created = sum(1 for r in results if r.get('gguf_creation', {}).get('gguf_info'))
    gguf_sizes = [r.get('gguf_creation', {}).get('gguf_info', {}).get('size', 0) for r in results if r.get('gguf_creation', {}).get('gguf_info')]
    
    print(f"   Files Created: {gguf_created}")
    if gguf_sizes:
        print(f"   Avg Size: {sum(gguf_sizes)/len(gguf_sizes):.1f} MB")
    
    # Top Performers
    print("\n🌟 TOP 10 PERFORMERS:")
    sorted_results = sorted(results, key=lambda x: x.get('quality_metrics', {}).get('quality_score', 0), reverse=True)
    
    for i, result in enumerate(sorted_results[:10]):
        domain = result.get('domain', 'unknown')
        quality = result.get('quality_metrics', {}).get('quality_score', 0)
        category = result.get('domain_validation', {}).get('category', 'unknown')
        print(f"   {i+1:2d}. {domain:<25} {quality:>6.2f}% ({category})")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   ✅ {passed_count}/{len(results)} domains successfully trained")
    print(f"   📈 Average quality: {sum(quality_scores)/len(quality_scores):.2f}%")
    print(f"   🏭 {gguf_created} GGUF files created")
    
    return results

if __name__ == "__main__":
    analyze_training_results() 