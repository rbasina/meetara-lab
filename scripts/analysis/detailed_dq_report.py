#!/usr/bin/env python3
"""
Detailed Data Quality Report for MeeTARA Lab Training Results
Comprehensive analysis of all 60+ domains with DQ metrics
"""

import json
import os
import glob
from datetime import datetime
from collections import defaultdict

def generate_detailed_dq_report():
    print("📊 MeeTARA Lab - Comprehensive Data Quality Report")
    print("="*80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Analysis: All Domain Training Results from Google Colab")
    
    # Load all session summaries
    pattern = os.path.join("logs", "session_summary_*.json")
    files = glob.glob(pattern)
    
    results = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print(f"📁 Loaded {len(results)} training session results")
    
    # Sort by domain name for consistent reporting
    results.sort(key=lambda x: x.get('domain', ''))
    
    # Generate detailed report
    print("\n" + "="*80)
    print("📋 DETAILED DOMAIN-BY-DOMAIN ANALYSIS")
    print("="*80)
    
    # Header
    print(f"{'#':<3} {'Domain':<25} {'Category':<12} {'Tier':<8} {'Quality':<8} {'Time':<6} {'Samples':<8} {'GGUF':<8} {'Model':<35}")
    print("-" * 120)
    
    total_quality = 0
    total_time = 0
    total_samples = 0
    category_stats = defaultdict(lambda: {'count': 0, 'quality': 0, 'time': 0})
    tier_stats = defaultdict(lambda: {'count': 0, 'quality': 0, 'time': 0})
    
    for i, result in enumerate(results, 1):
        domain = result.get('domain', 'unknown')[:24]
        category = result.get('domain_validation', {}).get('category', 'unknown')[:11]
        tier = result.get('model_selection', {}).get('model_tier', 'unknown')[:7]
        quality = result.get('quality_metrics', {}).get('quality_score', 0)
        duration = result.get('duration_seconds', 0)
        samples = result.get('sample_generation', {}).get('generated_samples', 0)
        gguf_created = "✅" if result.get('gguf_creation', {}).get('gguf_info') else "❌"
        model = result.get('model_selection', {}).get('base_model', 'unknown')[:34]
        
        # Accumulate stats
        total_quality += quality
        total_time += duration
        total_samples += samples
        
        category_stats[category]['count'] += 1
        category_stats[category]['quality'] += quality
        category_stats[category]['time'] += duration
        
        tier_stats[tier]['count'] += 1
        tier_stats[tier]['quality'] += quality
        tier_stats[tier]['time'] += duration
        
        print(f"{i:<3} {domain:<25} {category:<12} {tier:<8} {quality:>6.2f}% {duration:>5.1f}s {samples:>7} {gguf_created:<8} {model:<35}")
    
    # Summary statistics
    avg_quality = total_quality / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0
    avg_samples = total_samples / len(results) if results else 0
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print(f"📈 Total Domains Trained: {len(results)}")
    print(f"📊 Average Quality Score: {avg_quality:.2f}%")
    print(f"⏱️ Average Training Time: {avg_time:.1f}s")
    print(f"📊 Average Samples Generated: {avg_samples:,.0f}")
    print(f"⏱️ Total Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📊 Total Samples Generated: {total_samples:,}")
    
    # Category breakdown
    print("\n" + "="*80)
    print("📂 CATEGORY BREAKDOWN")
    print("="*80)
    
    for category, stats in sorted(category_stats.items()):
        if stats['count'] > 0:
            avg_cat_quality = stats['quality'] / stats['count']
            avg_cat_time = stats['time'] / stats['count']
            print(f"📁 {category.upper()}: {stats['count']} domains")
            print(f"   📈 Average Quality: {avg_cat_quality:.2f}%")
            print(f"   ⏱️ Average Time: {avg_cat_time:.1f}s")
            print(f"   📊 Total Time: {stats['time']:.1f}s")
    
    # Tier breakdown
    print("\n" + "="*80)
    print("🏗️ MODEL TIER BREAKDOWN")
    print("="*80)
    
    for tier, stats in sorted(tier_stats.items()):
        if stats['count'] > 0:
            avg_tier_quality = stats['quality'] / stats['count']
            avg_tier_time = stats['time'] / stats['count']
            print(f"🎯 {tier.upper()}: {stats['count']} domains")
            print(f"   📈 Average Quality: {avg_tier_quality:.2f}%")
            print(f"   ⏱️ Average Time: {avg_tier_time:.1f}s")
            print(f"   📊 Total Time: {stats['time']:.1f}s")
    
    # Data Quality Analysis
    print("\n" + "="*80)
    print("🔍 DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Quality distribution
    excellent = sum(1 for r in results if r.get('quality_metrics', {}).get('quality_score', 0) >= 99.5)
    good = sum(1 for r in results if 99.0 <= r.get('quality_metrics', {}).get('quality_score', 0) < 99.5)
    acceptable = sum(1 for r in results if 95.0 <= r.get('quality_metrics', {}).get('quality_score', 0) < 99.0)
    poor = sum(1 for r in results if r.get('quality_metrics', {}).get('quality_score', 0) < 95.0)
    
    print(f"🌟 Excellent Quality (≥99.5%): {excellent} domains ({excellent/len(results)*100:.1f}%)")
    print(f"✅ Good Quality (99.0-99.5%): {good} domains ({good/len(results)*100:.1f}%)")
    print(f"👍 Acceptable Quality (95.0-99.0%): {acceptable} domains ({acceptable/len(results)*100:.1f}%)")
    print(f"⚠️ Poor Quality (<95.0%): {poor} domains ({poor/len(results)*100:.1f}%)")
    
    # Sample generation efficiency
    print(f"\n📊 Sample Generation Efficiency:")
    generation_times = [r.get('sample_generation', {}).get('generation_time', 0) for r in results]
    avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
    print(f"   Average Generation Time: {avg_gen_time:.2f}s")
    print(f"   Samples per Second: {avg_samples/avg_gen_time:.0f}" if avg_gen_time > 0 else "   Samples per Second: N/A")
    
    # GGUF creation analysis
    gguf_created = sum(1 for r in results if r.get('gguf_creation', {}).get('gguf_info'))
    gguf_sizes = [r.get('gguf_creation', {}).get('gguf_info', {}).get('size', 0) for r in results if r.get('gguf_creation', {}).get('gguf_info')]
    gguf_qualities = [r.get('gguf_creation', {}).get('gguf_info', {}).get('quality', 0) for r in results if r.get('gguf_creation', {}).get('gguf_info')]
    
    print(f"\n🏭 GGUF Creation Analysis:")
    print(f"   Files Created: {gguf_created}/{len(results)} ({gguf_created/len(results)*100:.1f}%)")
    if gguf_sizes:
        print(f"   Average Size: {sum(gguf_sizes)/len(gguf_sizes):.1f} MB")
        print(f"   Size Consistency: {'✅ Consistent' if len(set(gguf_sizes)) <= 1 else '⚠️ Variable'}")
    if gguf_qualities:
        print(f"   Average GGUF Quality: {sum(gguf_qualities)/len(gguf_qualities):.1f}%")
    
    # Performance insights
    print("\n" + "="*80)
    print("🚀 PERFORMANCE INSIGHTS")
    print("="*80)
    
    # Fastest domains
    fastest_domains = sorted(results, key=lambda x: x.get('duration_seconds', 999))[:5]
    print("⚡ Fastest Training Domains:")
    for i, domain in enumerate(fastest_domains, 1):
        name = domain.get('domain', 'unknown')
        time = domain.get('duration_seconds', 0)
        quality = domain.get('quality_metrics', {}).get('quality_score', 0)
        print(f"   {i}. {name}: {time:.1f}s ({quality:.2f}%)")
    
    # Highest quality domains
    highest_quality = sorted(results, key=lambda x: x.get('quality_metrics', {}).get('quality_score', 0), reverse=True)[:5]
    print("\n🏆 Highest Quality Domains:")
    for i, domain in enumerate(highest_quality, 1):
        name = domain.get('domain', 'unknown')
        quality = domain.get('quality_metrics', {}).get('quality_score', 0)
        time = domain.get('duration_seconds', 0)
        print(f"   {i}. {name}: {quality:.2f}% ({time:.1f}s)")
    
    # Trinity Architecture validation
    print("\n" + "="*80)
    print("🔱 TRINITY ARCHITECTURE VALIDATION")
    print("="*80)
    
    # Check for Trinity components
    arc_reactor_efficiency = avg_quality  # Quality represents Arc Reactor efficiency
    perplexity_intelligence = len([r for r in results if r.get('domain_validation', {}).get('valid', False)]) / len(results) * 100
    einstein_fusion = (total_samples / len(results)) / 5000 * 100  # Capability amplification vs baseline
    
    print(f"⚡ Arc Reactor Foundation: {arc_reactor_efficiency:.1f}% efficiency")
    print(f"🧠 Perplexity Intelligence: {perplexity_intelligence:.1f}% validation success")
    print(f"🔬 Einstein Fusion: {einstein_fusion:.1f}% capability amplification")
    
    # Final assessment
    print("\n" + "="*80)
    print("🎯 FINAL ASSESSMENT")
    print("="*80)
    
    success_rate = len([r for r in results if r.get('quality_metrics', {}).get('passed', False)]) / len(results) * 100
    
    print(f"✅ Overall Success Rate: {success_rate:.1f}%")
    print(f"📊 Data Quality Score: {avg_quality:.2f}%")
    print(f"⚡ Training Efficiency: {avg_samples/avg_time:.0f} samples/second")
    print(f"🏭 Production Readiness: {'✅ READY' if success_rate >= 95 and avg_quality >= 99 else '⚠️ NEEDS REVIEW'}")
    
    # Trinity Architecture status
    trinity_status = "✅ OPERATIONAL" if (arc_reactor_efficiency >= 99 and perplexity_intelligence >= 95 and einstein_fusion >= 90) else "⚠️ OPTIMIZATION NEEDED"
    print(f"🔱 Trinity Architecture: {trinity_status}")
    
    return results

if __name__ == "__main__":
    generate_detailed_dq_report() 