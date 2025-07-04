#!/usr/bin/env python3
"""
MeeTARA Lab - Generate Training Data for All 62 Domains
Uses the existing data_generator_agent.py to create comprehensive training data
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with proper path handling
try:
    from trinity_core.agents.data_generator_agent import DataGeneratorAgent
    from trinity_core.domain_integration import get_all_domains, get_domain_categories
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import path...")
    sys.path.insert(0, str(project_root / "trinity-core"))
    sys.path.insert(0, str(project_root / "trinity-core" / "agents"))
    from data_generator_agent import DataGeneratorAgent
    from domain_integration import get_all_domains, get_domain_categories

async def generate_all_domains_training_data():
    """Generate training data for all 62 domains using DataGeneratorAgent"""
    
    print("🚀 MeeTARA Lab - Generating Training Data for All 62 Domains")
    print("=" * 60)
    
    # Initialize the data generator agent
    data_generator = DataGeneratorAgent()
    await data_generator.start()
    
    # Get all domains from Trinity configuration
    all_domains = get_all_domains()
    domain_categories = get_domain_categories()
    
    print(f"📊 Total domains to process: {len(all_domains)}")
    print(f"📂 Categories: {len(domain_categories)}")
    
    # Display domain breakdown
    print("\n📋 Domain breakdown by category:")
    for category, domain_list in domain_categories.items():
        print(f"   → {category}: {len(domain_list)} domains")
    
    # Generate training data for each domain
    successful_domains = []
    failed_domains = []
    
    for i, domain in enumerate(all_domains, 1):
        try:
            print(f"\n🎯 Processing domain {i}/{len(all_domains)}: {domain}")
            
            # Prepare training data request
            training_request = {
                "domain": domain,
                "samples": 2000,  # TARA proven parameter
                "quality_requirements": {
                    "emotional_intelligence": 0.25,
                    "domain_accuracy": 0.30,
                    "contextual_relevance": 0.25,
                    "crisis_handling": 0.20
                },
                "scenario_distribution": {
                    "realtime_ratio": 0.40,  # 40% real-time scenarios
                    "crisis_ratio": 0.05,    # 5% crisis scenarios
                    "standard_ratio": 0.55   # 55% standard scenarios
                },
                "output_path": f"data/training/{domain}"
            }
            
            # Generate training data using the agent
            await data_generator._prepare_training_data(training_request)
            
            successful_domains.append(domain)
            print(f"✅ Successfully generated training data for {domain}")
            
        except Exception as e:
            failed_domains.append((domain, str(e)))
            print(f"❌ Failed to generate training data for {domain}: {e}")
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 TRAINING DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful domains: {len(successful_domains)}")
    print(f"❌ Failed domains: {len(failed_domains)}")
    print(f"📈 Success rate: {len(successful_domains)/len(all_domains)*100:.1f}%")
    
    if successful_domains:
        print(f"\n✅ Successfully processed domains:")
        for domain in successful_domains:
            print(f"   → {domain}")
    
    if failed_domains:
        print(f"\n❌ Failed domains:")
        for domain, error in failed_domains:
            print(f"   → {domain}: {error}")
    
    # Save generation report
    report = {
        "timestamp": str(asyncio.get_event_loop().time()),
        "total_domains": len(all_domains),
        "successful_domains": successful_domains,
        "failed_domains": [{"domain": d, "error": e} for d, e in failed_domains],
        "success_rate": len(successful_domains)/len(all_domains)*100,
        "training_parameters": {
            "samples_per_domain": 2000,
            "realtime_ratio": 0.40,
            "crisis_ratio": 0.05,
            "quality_threshold": 0.8
        }
    }
    
    report_path = Path("data/training/generation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Generation report saved to: {report_path}")
    print(f"🎯 Ready for next step: Base model training with all {len(successful_domains)} domains!")
    
    return successful_domains, failed_domains

if __name__ == "__main__":
    asyncio.run(generate_all_domains_training_data()) 