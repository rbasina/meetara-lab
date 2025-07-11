#!/usr/bin/env python3
"""
MeeTARA Lab Training Results Analysis
Comprehensive analysis of all 60+ domain training results from Colab execution
"""

import json
import os
import glob
from datetime import datetime
from collections import defaultdict
import pandas as pd

class TrainingResultsAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = logs_dir
        self.results = []
        self.categories = {
            'healthcare': [],
            'business': [],
            'education': [],
            'technology': [],
            'creative': [],
            'daily_life': [],
            'specialized': []
        }
        
    def load_session_summaries(self):
        """Load all session summary JSON files"""
        pattern = os.path.join(self.logs_dir, "session_summary_*.json")
        files = glob.glob(pattern)
        
        print(f"🔍 Found {len(files)} session summary files")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results.append(data)
                    
                    # Categorize by domain category
                    category = data.get('domain_validation', {}).get('category', 'unknown')
                    if category in self.categories:
                        self.categories[category].append(data)
                        
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        print(f"✅ Successfully loaded {len(self.results)} training sessions")
        return self.results
    
    def analyze_quality_metrics(self):
        """Analyze quality metrics across all domains"""
        print("\n" + "="*80)
        print("📊 QUALITY METRICS ANALYSIS")
        print("="*80)
        
        total_domains = len(self.results)
        quality_scores = []
        passed_count = 0
        
        for result in self.results:
            quality_data = result.get('quality_metrics', {})
            quality_score = quality_data.get('quality_score', 0)
            quality_scores.append(quality_score)
            
            if quality_data.get('passed', False):
                passed_count += 1
        
        # Calculate statistics
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        min_quality = min(quality_scores) if quality_scores else 0
        max_quality = max(quality_scores) if quality_scores else 0
        
        print(f"📈 Total Domains Trained: {total_domains}")
        print(f"✅ Domains Passed: {passed_count} ({passed_count/total_domains*100:.1f}%)")
        print(f"📊 Average Quality Score: {avg_quality:.2f}%")
        print(f"📊 Quality Range: {min_quality:.2f}% - {max_quality:.2f}%")
        
        # Quality distribution
        excellent = sum(1 for q in quality_scores if q >= 99.5)
        good = sum(1 for q in quality_scores if 99.0 <= q < 99.5)
        acceptable = sum(1 for q in quality_scores if 95.0 <= q < 99.0)
        
        print(f"🌟 Excellent (≥99.5%): {excellent} domains")
        print(f"✅ Good (99.0-99.5%): {good} domains")
        print(f"👍 Acceptable (95.0-99.0%): {acceptable} domains")
        
        return {
            'total_domains': total_domains,
            'passed_count': passed_count,
            'avg_quality': avg_quality,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'quality_scores': quality_scores
        }
    
    def analyze_model_tiers(self):
        """Analyze model tier distribution and performance"""
        print("\n" + "="*80)
        print("🏗️ MODEL TIER ANALYSIS")
        print("="*80)
        
        tier_stats = defaultdict(lambda: {
            'count': 0,
            'models': set(),
            'avg_quality': 0,
            'avg_time': 0,
            'domains': []
        })
        
        for result in self.results:
            domain = result.get('domain', 'unknown')
            model_data = result.get('model_selection', {})
            tier = model_data.get('model_tier', 'unknown')
            base_model = model_data.get('base_model', 'unknown')
            
            quality_score = result.get('quality_metrics', {}).get('quality_score', 0)
            duration = result.get('duration_seconds', 0)
            
            tier_stats[tier]['count'] += 1
            tier_stats[tier]['models'].add(base_model)
            tier_stats[tier]['avg_quality'] += quality_score
            tier_stats[tier]['avg_time'] += duration
            tier_stats[tier]['domains'].append(domain)
        
        # Calculate averages
        for tier, stats in tier_stats.items():
            if stats['count'] > 0:
                stats['avg_quality'] /= stats['count']
                stats['avg_time'] /= stats['count']
        
        # Display results
        for tier, stats in sorted(tier_stats.items()):
            print(f"\n🎯 {tier.upper()} TIER:")
            print(f"   📊 Domains: {stats['count']}")
            print(f"   🤖 Models: {', '.join(stats['models'])}")
            print(f"   📈 Avg Quality: {stats['avg_quality']:.2f}%")
            print(f"   ⏱️ Avg Time: {stats['avg_time']:.1f}s")
            print(f"   🏷️ Domains: {', '.join(stats['domains'][:5])}{'...' if len(stats['domains']) > 5 else ''}")
        
        return tier_stats
    
    def analyze_categories(self):
        """Analyze performance by domain category"""
        print("\n" + "="*80)
        print("📂 CATEGORY ANALYSIS")
        print("="*80)
        
        category_stats = {}
        
        for category, results in self.categories.items():
            if not results:
                continue
                
            quality_scores = [r.get('quality_metrics', {}).get('quality_score', 0) for r in results]
            durations = [r.get('duration_seconds', 0) for r in results]
            
            category_stats[category] = {
                'count': len(results),
                'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'min_quality': min(quality_scores) if quality_scores else 0,
                'max_quality': max(quality_scores) if quality_scores else 0,
                'avg_time': sum(durations) / len(durations) if durations else 0,
                'domains': [r.get('domain', 'unknown') for r in results]
            }
        
        # Display results
        for category, stats in sorted(category_stats.items()):
            print(f"\n📁 {category.upper()} ({stats['count']} domains):")
            print(f"   📈 Quality: {stats['avg_quality']:.2f}% (Range: {stats['min_quality']:.2f}% - {stats['max_quality']:.2f}%)")
            print(f"   ⏱️ Avg Time: {stats['avg_time']:.1f}s")
            print(f"   🏷️ Domains: {', '.join(stats['domains'][:8])}{'...' if len(stats['domains']) > 8 else ''}")
        
        return category_stats
    
    def analyze_training_parameters(self):
        """Analyze training parameters across domains"""
        print("\n" + "="*80)
        print("⚙️ TRAINING PARAMETERS ANALYSIS")
        print("="*80)
        
        param_stats = defaultdict(list)
        
        for result in self.results:
            params = result.get('parameters', {}).get('parameters', {})
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_stats[key].append(value)
        
        # Display parameter statistics
        for param, values in param_stats.items():
            if values:
                avg_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                unique_vals = len(set(values))
                
                print(f"📊 {param}:")
                print(f"   Average: {avg_val:.3f}")
                print(f"   Range: {min_val} - {max_val}")
                print(f"   Unique values: {unique_vals}")
        
        return param_stats
    
    def analyze_gguf_creation(self):
        """Analyze GGUF creation results"""
        print("\n" + "="*80)
        print("🏭 GGUF CREATION ANALYSIS")
        print("="*80)
        
        gguf_stats = {
            'total_created': 0,
            'avg_size': 0,
            'formats': defaultdict(int),
            'qualities': [],
            'compression_ratios': []
        }
        
        for result in self.results:
            gguf_data = result.get('gguf_creation', {}).get('gguf_info', {})
            if gguf_data:
                gguf_stats['total_created'] += 1
                gguf_stats['avg_size'] += gguf_data.get('size', 0)
                gguf_stats['formats'][gguf_data.get('format', 'unknown')] += 1
                gguf_stats['qualities'].append(gguf_data.get('quality', 0))
                
                # Calculate compression ratio
                original_size = gguf_data.get('model_size_mb', 0)
                compressed_size = gguf_data.get('size', 0)
                if original_size > 0:
                    ratio = original_size / compressed_size
                    gguf_stats['compression_ratios'].append(ratio)
        
        # Calculate averages
        if gguf_stats['total_created'] > 0:
            gguf_stats['avg_size'] /= gguf_stats['total_created']
        
        avg_quality = sum(gguf_stats['qualities']) / len(gguf_stats['qualities']) if gguf_stats['qualities'] else 0
        avg_compression = sum(gguf_stats['compression_ratios']) / len(gguf_stats['compression_ratios']) if gguf_stats['compression_ratios'] else 0
        
        print(f"📊 Total GGUF Files Created: {gguf_stats['total_created']}")
        print(f"📏 Average Size: {gguf_stats['avg_size']:.1f} MB")
        print(f"📈 Average Quality: {avg_quality:.1f}%")
        print(f"🗜️ Average Compression Ratio: {avg_compression:.1f}x")
        
        print(f"\n📋 Format Distribution:")
        for fmt, count in gguf_stats['formats'].items():
            print(f"   {fmt}: {count} files")
        
        return gguf_stats
    
    def generate_detailed_report(self):
        """Generate detailed domain-by-domain report"""
        print("\n" + "="*80)
        print("📝 DETAILED DOMAIN REPORT")
        print("="*80)
        
        # Sort by quality score descending
        sorted_results = sorted(self.results, key=lambda x: x.get('quality_metrics', {}).get('quality_score', 0), reverse=True)
        
        print(f"{'Domain':<25} {'Category':<12} {'Model Tier':<10} {'Quality':<8} {'Time':<6} {'GGUF':<8}")
        print("-" * 80)
        
        for result in sorted_results:
            domain = result.get('domain', 'unknown')[:24]
            category = result.get('domain_validation', {}).get('category', 'unknown')[:11]
            tier = result.get('model_selection', {}).get('model_tier', 'unknown')[:9]
            quality = result.get('quality_metrics', {}).get('quality_score', 0)
            duration = result.get('duration_seconds', 0)
            gguf_created = "✅" if result.get('gguf_creation', {}).get('gguf_info') else "❌"
            
            print(f"{domain:<25} {category:<12} {tier:<10} {quality:>6.1f}% {duration:>5.1f}s {gguf_created:<8}")
        
        return sorted_results
    
    def export_to_csv(self, filename="training_results_analysis.csv"):
        """Export results to CSV for further analysis"""
        data = []
        
        for result in self.results:
            row = {
                'domain': result.get('domain', ''),
                'category': result.get('domain_validation', {}).get('category', ''),
                'model_tier': result.get('model_selection', {}).get('model_tier', ''),
                'base_model': result.get('model_selection', {}).get('base_model', ''),
                'quality_score': result.get('quality_metrics', {}).get('quality_score', 0),
                'quality_passed': result.get('quality_metrics', {}).get('passed', False),
                'duration_seconds': result.get('duration_seconds', 0),
                'samples_generated': result.get('sample_generation', {}).get('generated_samples', 0),
                'generation_time': result.get('sample_generation', {}).get('generation_time', 0),
                'gguf_size_mb': result.get('gguf_creation', {}).get('gguf_info', {}).get('size', 0),
                'gguf_quality': result.get('gguf_creation', {}).get('gguf_info', {}).get('quality', 0),
                'batch_size': result.get('parameters', {}).get('parameters', {}).get('batch_size', 0),
                'max_steps': result.get('parameters', {}).get('parameters', {}).get('max_steps', 0),
                'lora_r': result.get('parameters', {}).get('parameters', {}).get('lora_r', 0),
                'learning_rate': result.get('parameters', {}).get('parameters', {}).get('learning_rate', ''),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n💾 Results exported to {filename}")
        return df
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 MeeTARA Lab Training Results Analysis")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_session_summaries()
        
        # Run all analyses
        quality_stats = self.analyze_quality_metrics()
        tier_stats = self.analyze_model_tiers()
        category_stats = self.analyze_categories()
        param_stats = self.analyze_training_parameters()
        gguf_stats = self.analyze_gguf_creation()
        detailed_report = self.generate_detailed_report()
        
        # Export results
        try:
            df = self.export_to_csv()
            print(f"📊 CSV export successful with {len(df)} rows")
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
        
        # Summary
        print("\n" + "="*80)
        print("🎯 EXECUTIVE SUMMARY")
        print("="*80)
        print(f"✅ Successfully trained {quality_stats['total_domains']} domains")
        print(f"📈 Average quality score: {quality_stats['avg_quality']:.2f}%")
        print(f"🏭 GGUF files created: {gguf_stats['total_created']}")
        print(f"⏱️ Total training time: {sum(r.get('duration_seconds', 0) for r in self.results):.1f}s")
        print(f"🎯 Success rate: {quality_stats['passed_count']}/{quality_stats['total_domains']} ({quality_stats['passed_count']/quality_stats['total_domains']*100:.1f}%)")
        
        return {
            'quality_stats': quality_stats,
            'tier_stats': tier_stats,
            'category_stats': category_stats,
            'param_stats': param_stats,
            'gguf_stats': gguf_stats,
            'detailed_report': detailed_report
        }

if __name__ == "__main__":
    analyzer = TrainingResultsAnalyzer()
    results = analyzer.run_complete_analysis() 