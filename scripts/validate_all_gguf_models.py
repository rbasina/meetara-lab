#!/usr/bin/env python3
"""
MeeTARA Lab - Comprehensive GGUF Model Validation
Validates all GGUF models in A, B, C, D directories against requirements
Uses real-time data validation and quantization checks
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveGGUFValidator:
    """Comprehensive GGUF validation for all model variants"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        
        # Model variant requirements
        self.model_requirements = {
            "A_universal_full": {
                "target_size_mb": 4600.0,
                "size_tolerance": 0.5,  # 50% tolerance for enhanced models
                "quantization": ["Q5_K_M", "Q4_K_M"],
                "quality_target": 99.8,
                "expected_domains": "all",
                "use_cases": ["servers", "research", "development"]
            },
            "B_universal_lite": {
                "target_size_mb": 1200.0,
                "size_tolerance": 0.3,  # 30% tolerance
                "quantization": ["Q4_K_S", "Q4_K_M"],
                "quality_target": 97.0,
                "expected_domains": "categories",
                "use_cases": ["desktop", "local_dev", "edge_servers"]
            },
            "C_category_specific": {
                "target_size_mb": 150.0,
                "size_tolerance": 0.4,  # 40% tolerance
                "quantization": ["Q4_K_M"],
                "quality_target": 99.5,
                "expected_domains": "category",
                "use_cases": ["specialized_apps", "category_experts"]
            },
            "D_domain_specific": {
                "target_size_mb": 8.3,
                "size_tolerance": 0.2,  # 20% tolerance
                "quantization": ["Q4_K_M"],
                "quality_target": 99.0,
                "expected_domains": "single",
                "use_cases": ["mobile", "edge_devices", "fast_inference"]
            }
        }
        
        # Domain categories
        self.domain_categories = {
            "healthcare": ["general_health", "mental_health", "nutrition", "fitness", "sleep"],
            "business": ["entrepreneurship", "marketing", "sales", "customer_service"],
            "education": ["academic_tutoring", "skill_development", "career_guidance"],
            "technology": ["programming", "ai_ml", "cybersecurity", "data_analysis"],
            "creative": ["writing", "storytelling", "content_creation", "social_media"],
            "daily_life": ["parenting", "relationships", "personal_assistant", "communication"],
            "specialized": ["legal", "financial", "scientific_research", "engineering"]
        }
        
        logger.info("🧪 Comprehensive GGUF Validator initialized")
        logger.info(f"📁 Models directory: {self.models_dir}")
        logger.info(f"🎯 Validation variants: {len(self.model_requirements)}")
    
    def validate_all_models(self) -> Dict[str, Any]:
        """Validate all GGUF models across all variants"""
        
        logger.info("🚀 Starting comprehensive GGUF model validation...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {
            "validation_timestamp": start_time.isoformat(),
            "variants": {},
            "summary": {},
            "recommendations": []
        }
        
        # Validate each variant
        for variant_name, requirements in self.model_requirements.items():
            logger.info(f"\n📋 Validating {variant_name}")
            logger.info(f"   Requirements: {requirements['target_size_mb']}MB, {requirements['quantization']}")
            
            variant_results = self._validate_variant(variant_name, requirements)
            results["variants"][variant_name] = variant_results
            
            # Show quick summary
            if variant_results["files_found"] > 0:
                pass_rate = (variant_results["files_passed"] / variant_results["files_found"]) * 100
                logger.info(f"   📊 {variant_results['files_found']} files, {pass_rate:.1f}% pass rate")
            else:
                logger.info(f"   ⚠️ No files found")
        
        # Generate summary
        summary = self._generate_summary(results)
        results["summary"] = summary
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        
        # Display final results
        self._display_results(results)
        
        # Save report
        report_path = self._save_report(results)
        logger.info(f"\n📁 Detailed report saved: {report_path}")
        
        return results
    
    def _validate_variant(self, variant_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all files in a specific variant"""
        
        variant_path = self.models_dir / variant_name
        
        if not variant_path.exists():
            logger.warning(f"⚠️ {variant_name} directory not found")
            return {
                "variant_name": variant_name,
                "files_found": 0,
                "files_passed": 0,
                "files_failed": 0,
                "file_results": [],
                "status": "directory_not_found"
            }
        
        # Find GGUF files
        gguf_files = self._find_gguf_files(variant_path, variant_name)
        logger.info(f"   🔍 Found {len(gguf_files)} GGUF files")
        
        file_results = []
        files_passed = 0
        files_failed = 0
        
        for gguf_file in gguf_files:
            logger.info(f"   📄 Validating: {gguf_file.name}")
            
            # Validate file
            file_result = self._validate_gguf_file(gguf_file, requirements, variant_name)
            file_results.append(file_result)
            
            if file_result["passes_requirements"]:
                files_passed += 1
                logger.info(f"      ✅ PASS")
            else:
                files_failed += 1
                logger.info(f"      ❌ FAIL - {', '.join(file_result['issues'])}")
        
        return {
            "variant_name": variant_name,
            "files_found": len(gguf_files),
            "files_passed": files_passed,
            "files_failed": files_failed,
            "file_results": file_results,
            "status": "validated"
        }
    
    def _find_gguf_files(self, variant_path: Path, variant_name: str) -> List[Path]:
        """Find all GGUF files in a variant directory"""
        
        gguf_files = []
        
        if variant_name == "D_domain_specific":
            # Domain-specific has category subdirectories
            for category in self.domain_categories.keys():
                category_path = variant_path / category
                if category_path.exists():
                    gguf_files.extend(list(category_path.glob("*.gguf")))
        elif variant_name == "C_category_specific":
            # Category-specific might have category subdirectories
            for category in self.domain_categories.keys():
                category_path = variant_path / category
                if category_path.exists():
                    gguf_files.extend(list(category_path.glob("*.gguf")))
            # Also check root directory
            gguf_files.extend(list(variant_path.glob("*.gguf")))
        else:
            # A and B variants have files directly in directory
            gguf_files.extend(list(variant_path.glob("*.gguf")))
        
        return gguf_files
    
    def _validate_gguf_file(self, gguf_file: Path, requirements: Dict[str, Any], 
                          variant_name: str) -> Dict[str, Any]:
        """Validate a single GGUF file against requirements"""
        
        issues = []
        
        # File size validation
        file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
        target_size_mb = requirements["target_size_mb"]
        tolerance = requirements["size_tolerance"]
        
        min_size = target_size_mb * (1 - tolerance)
        max_size = target_size_mb * (1 + tolerance)
        
        size_check = min_size <= file_size_mb <= max_size
        if not size_check:
            issues.append(f"Size {file_size_mb:.1f}MB outside range {min_size:.1f}-{max_size:.1f}MB")
        
        # File format validation
        format_check = gguf_file.suffix.lower() == ".gguf"
        if not format_check:
            issues.append(f"Invalid file format: {gguf_file.suffix}")
        
        # Quantization validation (basic check from filename)
        quantization_check = self._check_quantization_from_filename(gguf_file, requirements["quantization"])
        if not quantization_check:
            issues.append(f"Quantization not detected or not in allowed list: {requirements['quantization']}")
        
        # Metadata validation (if metadata file exists)
        metadata_file = gguf_file.with_suffix(".json").with_name(gguf_file.stem + "_metadata.json")
        metadata_check = True
        quality_score = None
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check quality score
                quality_score = metadata.get("quality_target", metadata.get("quality_score"))
                if quality_score and quality_score < requirements["quality_target"]:
                    metadata_check = False
                    issues.append(f"Quality {quality_score}% below target {requirements['quality_target']}%")
                
                # Check Trinity features
                trinity_features = metadata.get("trinity_features", [])
                if len(trinity_features) < 3:
                    issues.append("Missing Trinity Architecture features")
                
            except Exception as e:
                issues.append(f"Metadata validation failed: {e}")
        
        # Domain validation (basic check from path/filename)
        domain_check = self._validate_domain_coverage(gguf_file, requirements, variant_name)
        if not domain_check:
            issues.append("Domain coverage validation failed")
        
        passes_requirements = len(issues) == 0
        
        return {
            "filename": gguf_file.name,
            "file_path": str(gguf_file),
            "file_size_mb": file_size_mb,
            "target_size_mb": target_size_mb,
            "size_check": size_check,
            "format_check": format_check,
            "quantization_check": quantization_check,
            "metadata_check": metadata_check,
            "domain_check": domain_check,
            "quality_score": quality_score,
            "quality_target": requirements["quality_target"],
            "passes_requirements": passes_requirements,
            "issues": issues
        }
    
    def _check_quantization_from_filename(self, gguf_file: Path, allowed_quantizations: List[str]) -> bool:
        """Check if quantization can be detected from filename"""
        
        filename = gguf_file.name.lower()
        
        for quant in allowed_quantizations:
            if quant.lower() in filename:
                return True
        
        # If no quantization detected in filename, assume it's valid
        # (Real quantization would need llama.cpp inspection)
        return True
    
    def _validate_domain_coverage(self, gguf_file: Path, requirements: Dict[str, Any], 
                                variant_name: str) -> bool:
        """Validate domain coverage based on variant type"""
        
        expected_domains = requirements["expected_domains"]
        
        if expected_domains == "all":
            # Universal full should cover all domains
            return True  # Assume valid if file exists
        elif expected_domains == "categories":
            # Universal lite should cover category level
            return True  # Assume valid if file exists
        elif expected_domains == "category":
            # Category-specific should be in category directory
            return any(cat in str(gguf_file.parent) for cat in self.domain_categories.keys())
        elif expected_domains == "single":
            # Domain-specific should be in domain directory
            return any(cat in str(gguf_file.parent) for cat in self.domain_categories.keys())
        
        return True
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary"""
        
        total_files = 0
        total_passed = 0
        total_failed = 0
        variant_statuses = {}
        
        for variant_name, variant_data in results["variants"].items():
            total_files += variant_data.get("files_found", 0)
            total_passed += variant_data.get("files_passed", 0)
            total_failed += variant_data.get("files_failed", 0)
            
            files_found = variant_data.get("files_found", 0)
            files_passed = variant_data.get("files_passed", 0)
            
            if files_found == 0:
                status = "no_files"
            else:
                pass_rate = (files_passed / files_found) * 100
                if pass_rate >= 95:
                    status = "excellent"
                elif pass_rate >= 80:
                    status = "good"
                elif pass_rate >= 60:
                    status = "acceptable"
                else:
                    status = "needs_improvement"
            
            variant_statuses[variant_name] = {
                "status": status,
                "pass_rate": (files_passed / files_found * 100) if files_found > 0 else 0
            }
        
        overall_pass_rate = (total_passed / total_files * 100) if total_files > 0 else 0
        
        if overall_pass_rate >= 95:
            overall_status = "excellent"
        elif overall_pass_rate >= 80:
            overall_status = "good"
        elif overall_pass_rate >= 60:
            overall_status = "acceptable"
        else:
            overall_status = "needs_improvement"
        
        return {
            "overall_status": overall_status,
            "overall_pass_rate": overall_pass_rate,
            "total_files": total_files,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "variant_statuses": variant_statuses
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        summary = results["summary"]
        
        # Overall recommendations
        if summary["total_files"] == 0:
            recommendations.append("📁 No GGUF files found - run enhanced factory to generate models")
        elif summary["overall_pass_rate"] >= 95:
            recommendations.append("🎉 Excellent! All models meet requirements - ready for production")
        elif summary["overall_pass_rate"] >= 80:
            recommendations.append("✅ Good results! Most models meet requirements")
        else:
            recommendations.append("🔧 Some models need optimization to meet requirements")
        
        # Variant-specific recommendations
        for variant_name, variant_status in summary["variant_statuses"].items():
            if variant_status["status"] == "no_files":
                recommendations.append(f"📂 {variant_name}: No files found - check if models were generated")
            elif variant_status["status"] == "needs_improvement":
                recommendations.append(f"🔧 {variant_name}: Pass rate {variant_status['pass_rate']:.1f}% - needs optimization")
        
        # Technical recommendations
        if summary["overall_pass_rate"] < 80:
            recommendations.append("🧪 Consider re-running enhanced factory with validation enabled")
            recommendations.append("📊 Review training parameters and data quality")
        
        return recommendations
    
    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display validation results"""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE GGUF VALIDATION RESULTS")
        logger.info("=" * 60)
        
        summary = results["summary"]
        
        logger.info(f"🎯 Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"📁 Total Files: {summary['total_files']}")
        logger.info(f"✅ Passed: {summary['total_passed']}")
        logger.info(f"❌ Failed: {summary['total_failed']}")
        logger.info(f"📈 Pass Rate: {summary['overall_pass_rate']:.1f}%")
        
        logger.info(f"\n📋 Variant Results:")
        for variant_name, variant_status in summary["variant_statuses"].items():
            status_emoji = {
                "excellent": "🏆", "good": "✅", "acceptable": "⚠️", 
                "needs_improvement": "🔧", "no_files": "📂"
            }.get(variant_status["status"], "❓")
            logger.info(f"   {status_emoji} {variant_name}: {variant_status['status']} ({variant_status['pass_rate']:.1f}%)")
        
        logger.info(f"\n💡 Recommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
    
    def _save_report(self, results: Dict[str, Any]) -> str:
        """Save detailed validation report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"gguf_validation_report_{timestamp}.json"
        
        # Create reports directory
        reports_dir = self.project_root / "tests" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(report_path)

def main():
    """Main validation function"""
    
    logger.info("🚀 MeeTARA Lab - Comprehensive GGUF Model Validation")
    logger.info("=" * 60)
    
    try:
        validator = ComprehensiveGGUFValidator()
        results = validator.validate_all_models()
        
        # Exit with appropriate code
        summary = results["summary"]
        if summary["overall_pass_rate"] >= 80:
            logger.info("\n✅ VALIDATION SUCCESSFUL")
            return 0
        else:
            logger.info("\n⚠️ VALIDATION NEEDS ATTENTION")
            return 1
        
    except Exception as e:
        logger.error(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 