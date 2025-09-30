#!/usr/bin/env python3
"""
Training Data Validation Script
Validates JSON training data before training to prevent corrupted data issues
"""

import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any

def find_latest_training_file(domain: str) -> str:
    """Find the latest training file for a domain"""
    
    # Try both patterns: domain-specific directory and root directory
    patterns = [
        f"data/training/{domain}/{domain}_raw_structured_data_*.json", # New structure for raw data
        f"data/training/{domain}/{domain}_train_*.json",  # Existing structure with domain directories
        f"data/training/{domain}_train_*.json"  # Old structure
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def get_domain_keywords(domain: str) -> List[str]:
    """Get domain-specific keywords for validation"""
    
    domain_keywords = {
        "healthcare": [
            "health", "medical", "doctor", "patient", "treatment", "symptoms",
            "medication", "hospital", "emergency", "care", "wellness", "therapy",
            "diagnosis", "prescription", "nurse", "clinic", "appointment",
            "healthcare", "provider", "physician", "specialist", "condition",
            "disease", "illness", "injury", "surgery", "recovery", "prevention"
        ],
        "parenting": [
            "child", "parent", "family", "kids", "children", "baby", "toddler",
            "teenager", "discipline", "education", "behavior", "development",
            "school", "homework", "activities", "safety", "nutrition", "sleep",
            "emotional", "support", "guidance", "rules", "boundaries"
        ],
        "relationships": [
            "relationship", "partner", "marriage", "dating", "communication",
            "love", "trust", "conflict", "understanding", "support", "intimacy",
            "commitment", "family", "friendship", "connection", "emotional"
        ],
        "personal_assistant": [
            "help", "assist", "organize", "schedule", "reminder", "task",
            "planning", "efficiency", "productivity", "management", "support"
        ],
        "communication": [
            "communication", "speak", "listen", "conversation", "express",
            "understand", "message", "dialogue", "interaction", "clarity"
        ],
        "home_management": [
            "home", "house", "cleaning", "organization", "maintenance",
            "chores", "household", "domestic", "living", "space"
        ],
        "shopping": [
            "buy", "purchase", "shop", "store", "product", "price", "deal",
            "discount", "quality", "comparison", "budget", "spending"
        ],
        "planning": [
            "plan", "organize", "schedule", "arrange", "prepare", "coordinate",
            "timeline", "deadline", "goal", "objective", "strategy"
        ],
        "transportation": [
            "travel", "transport", "vehicle", "car", "bus", "train", "flight",
            "commute", "route", "destination", "journey", "mobility"
        ],
        "time_management": [
            "time", "schedule", "prioritize", "efficiency", "productivity",
            "deadline", "timeline", "organization", "planning", "balance"
        ],
        "decision_making": [
            "decision", "choose", "select", "option", "choice", "consider",
            "evaluate", "analyze", "judgment", "conclusion", "outcome"
        ],
        "conflict_resolution": [
            "conflict", "dispute", "resolution", "mediation", "compromise",
            "negotiation", "agreement", "understanding", "peace", "harmony"
        ],
        "work_life_balance": [
            "work", "life", "balance", "career", "personal", "professional",
            "time", "stress", "wellness", "happiness", "fulfillment"
        ]
    }
    
    # Return domain-specific keywords or generic keywords if domain not found
    return domain_keywords.get(domain, [
        "help", "support", "assist", "guide", "advice", "information",
        "solution", "answer", "response", "understanding"
    ])

def validate_healthcare_data(data_path: str) -> Dict[str, Any]:
    """Validate healthcare training data for proper content"""
    return validate_domain_data(data_path, "healthcare")

def validate_domain_data(data_path: str, domain: str) -> Dict[str, Any]:
    """Validate training data for any domain"""
    
    print(f"[DATA] Validating {domain} training data: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[DATA] Total samples: {len(data)}")
        
        # Validation results
        validation = {
            "total_samples": len(data),
            "valid_samples": 0,
            "corrupted_samples": 0,
            "issues": [],
            "sample_responses": [],
            "domain_keywords_found": [],
            "wrong_content_found": []
        }
        
        # Domain-specific keywords that should be present
        domain_keywords = get_domain_keywords(domain)
        
        # Wrong content patterns that indicate corruption
        wrong_patterns = [
            "arcane", "study associate", "inventory", "accord", "research findings",
            "gaming", "rpg", "quest", "character", "level up", "experience points",
            "dungeon", "spell", "magic", "wizard", "warrior", "monster"
        ]
        
        for i, sample in enumerate(data):
            try:
                # Check if sample has required structure
                if "turns" not in sample or len(sample["turns"]) < 2:
                    validation["corrupted_samples"] += 1
                    validation["issues"].append(f"Sample {i}: Missing conversation turns")
                    continue
                
                # Get assistant response (second turn)
                assistant_response = sample["turns"][1]["content"]
                
                # Only add to sample responses for first 5
                if len(validation["sample_responses"]) < 5:
                    validation["sample_responses"].append(assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response)
                
                # Check for domain keywords
                response_lower = assistant_response.lower()
                found_keywords = [kw for kw in domain_keywords if kw in response_lower]
                if found_keywords:
                    validation["domain_keywords_found"].extend(found_keywords)
                
                # Check for wrong content
                found_wrong = [pattern for pattern in wrong_patterns if pattern in response_lower]
                if found_wrong:
                    validation["wrong_content_found"].extend(found_wrong)
                    validation["corrupted_samples"] += 1
                    validation["issues"].append(f"Sample {i}: Contains wrong content: {found_wrong}")
                else:
                    validation["valid_samples"] += 1
                
                # Check more samples for better validation
                if i >= 100:  # Check first 100 samples
                    break
                    
            except Exception as e:
                validation["corrupted_samples"] += 1
                validation["issues"].append(f"Sample {i}: Error processing - {e}")
        
        # Calculate validation score based on samples checked
        samples_checked = min(100, validation["total_samples"])
        validation["validation_score"] = validation["valid_samples"] / samples_checked if samples_checked > 0 else 0
        
        # Remove duplicates from keyword lists
        validation["domain_keywords_found"] = list(set(validation["domain_keywords_found"]))
        validation["wrong_content_found"] = list(set(validation["wrong_content_found"]))
        
        return validation
        
    except Exception as e:
        return {
            "error": f"Failed to validate data: {e}",
            "validation_score": 0
        }

def print_validation_results(validation: Dict[str, Any], domain: str):
    """Print validation results in a clear format"""
    
    print(f"\n{'='*60}")
    print(f"[RESULTS] {domain.upper()} TRAINING DATA VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if "error" in validation:
        print(f"[ERROR] {validation['error']}")
        return
    
    print(f"[DATA] Total Samples: {validation['total_samples']}")
    print(f"[SUCCESS] Valid Samples: {validation['valid_samples']}")
    print(f"[ERROR] Corrupted Samples: {validation['corrupted_samples']}")
    print(f"[SCORE] Validation Score: {validation['validation_score']:.2%}")
    
    if validation['domain_keywords_found']:
        print(f"\n[{domain.upper()}] Keywords Found: {', '.join(validation['domain_keywords_found'])}")
    
    if validation['wrong_content_found']:
        print(f"\n[WARNING] WRONG CONTENT DETECTED: {', '.join(validation['wrong_content_found'])}")
        print("[WARNING] This indicates corrupted training data!")
    
    if validation['sample_responses']:
        print(f"\n[SAMPLES] Sample Responses:")
        for i, response in enumerate(validation['sample_responses'], 1):
            print(f"  {i}. {response}")
    
    if validation['issues']:
        print(f"\n[ISSUES] Issues Found:")
        for issue in validation['issues'][:10]:  # Show first 10 issues
            print(f"  • {issue}")
        if len(validation['issues']) > 10:
            print(f"  ... and {len(validation['issues']) - 10} more issues")
    
    # Final recommendation
    print(f"\n{'='*60}")
    if validation['validation_score'] >= 0.9:
        print("[EXCELLENT] Data is ready for training")
    elif validation['validation_score'] >= 0.7:
        print("[GOOD] Data has minor issues but can be used")
    elif validation['validation_score'] >= 0.5:
        print("[POOR] Data has significant issues, consider regenerating")
    else:
        print("[CRITICAL] Data is corrupted, must regenerate before training")
    
    print(f"{'='*60}")

def main():
    """Main validation function"""
    
    # Find the latest healthcare training file
    healthcare_data_path = find_latest_training_file("healthcare")
    
    if not healthcare_data_path:
        print(f"[ERROR] No healthcare training data found")
        print("[INFO] Run data generation first: python scripts/tara_universal_pipeline.py --domain healthcare --auto_generate")
        return
    
    print(f"[FOUND] Found latest healthcare file: {healthcare_data_path}")
    
    # Validate healthcare data
    validation = validate_healthcare_data(healthcare_data_path)
    print_validation_results(validation, "healthcare")
    
    # Save validation report
    report_path = "training_data_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    print(f"\n[REPORT] Validation report saved to: {report_path}")
    
    # Return validation score for automation
    return validation.get('validation_score', 0)

if __name__ == "__main__":
    main() 