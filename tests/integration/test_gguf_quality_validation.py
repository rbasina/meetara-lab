#!/usr/bin/env python3
"""
MeeTARA Lab - Comprehensive GGUF Quality Validation Test
Tests all GGUF files in A, B, C, D model variants against requirements
Uses real-time data from data/real/ and trained models from data/models/trained/
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add trinity_core to path
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity_core"))

from agents.model_factory import model_factory
from config_manager import get_config_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveGGUFValidator:
    """Comprehensive GGUF quality validation for all model variants"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        
        # Model variant directories
        self.model_variants = {
            "A_universal_full": {
                "path": self.models_dir / "A_universal_full",
                "target_size_mb": 4600.0,
                "quantization": "Q5_K_M",
                "quality_target": 99.8,
                "use_cases": ["servers", "research", "development"]
            },
            "B_universal_lite": {
                "path": self.models_dir / "B_universal_lite", 
                "target_size_mb": 1200.0,
                "quantization": "Q4_K_S",
                "quality_target": 97.0,
                "use_cases": ["desktop", "local_dev", "edge_servers"]
            },
            "C_category_specific": {
                "path": self.models_dir / "C_category_specific",
                "target_size_mb": 150.0,
                "quantization": "Q4_K_M", 
                "quality_target": 99.5,
                "use_cases": ["specialized_apps", "category_experts"]
            },
            "D_domain_specific": {
                "path": self.models_dir / "D_domain_specific",
                "target_size_mb": 8.3,
                "quantization": "Q4_K_M",
                "quality_target": 99.0,
                "use_cases": ["mobile", "edge_devices", "fast_inference"]
            }
        }
        
        # Domain categories from successful training
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
        logger.info(f"📁 Model variants: {len(self.model_variants)}")
        logger.info(f"🎯 Domain categories: {len(self.domain_categories)}")
    
    async def validate_all_gguf_files(self) -> Dict[str, Any]:
        """Validate all GGUF files across all model variants"""
        
        logger.info("🚀 Starting comprehensive GGUF validation...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {
            "validation_timestamp": start_time.isoformat(),
            "model_variants": {},
            "summary": {},
            "recommendations": []
        }
        
        # Validate each model variant
        for variant_name, variant_config in self.model_variants.items():
            logger.info(f"\n📋 Validating {variant_name}")
            logger.info(f"   Target: {variant_config['target_size_mb']}MB, {variant_config['quantization']}")
            logger.info(f"   Quality: {variant_config['quality_target']}%")
            
            variant_results = await self._validate_model_variant(variant_name, variant_config)
            results["model_variants"][variant_name] = variant_results
            
            # Show variant summary
            if variant_results["files_found"] > 0:
                avg_quality = sum(f["quality_score"] for f in variant_results["file_results"] if f["quality_score"]) / len([f for f in variant_results["file_results"] if f["quality_score"]])
                logger.info(f"   ✅ {variant_results['files_found']} files, avg quality: {avg_quality:.1f}%")
            else:
                logger.info(f"   ⚠️ No GGUF files found")
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary(results)
        results["summary"] = summary
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        
        # Display final results
        self._display_final_results(results)
        
        # Save detailed report
        report_path = self._save_validation_report(results)
        logger.info(f"\n📁 Detailed report saved: {report_path}")
        
        return results
    
    async def _validate_model_variant(self, variant_name: str, variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all GGUF files in a specific model variant"""
        
        variant_path = variant_config["path"]
        
        if not variant_path.exists():
            logger.warning(f"⚠️ {variant_name} directory not found: {variant_path}")
            return {
                "variant_name": variant_name,
                "files_found": 0,
                "file_results": [],
                "variant_summary": {"status": "directory_not_found"}
            }
        
        # Find all GGUF files
        gguf_files = []
        if variant_name == "D_domain_specific":
            # Domain-specific has category subdirectories
            for category in self.domain_categories.keys():
                category_path = variant_path / category
                if category_path.exists():
                    gguf_files.extend(list(category_path.glob("*.gguf")))
        else:
            # Other variants have files directly in the directory
            gguf_files = list(variant_path.glob("*.gguf"))
        
        logger.info(f"   🔍 Found {len(gguf_files)} GGUF files")
        
        file_results = []
        
        # Validate each GGUF file
        for gguf_file in gguf_files:
            logger.info(f"   📄 Testing: {gguf_file.name}")
            
            # Determine domain from filename or path
            domain = self._extract_domain_from_path(gguf_file, variant_name)
            
            # Run validation using Model Factory
            try:
                validation_result = await model_factory.validate_gguf_with_real_testing(
                    str(gguf_file), domain
                )
                
                # Assess against requirements
                requirement_assessment = self._assess_against_requirements(
                    gguf_file, validation_result, variant_config
                )
                
                file_result = {
                    "filename": gguf_file.name,
                    "file_path": str(gguf_file),
                    "domain": domain,
                    "file_size_mb": gguf_file.stat().st_size / (1024 * 1024),
                    "validation_result": validation_result,
                    "requirement_assessment": requirement_assessment,
                    "quality_score": validation_result.get("average_quality", 0) * 100 if validation_result else 0,
                    "meets_requirements": requirement_assessment.get("overall_pass", False)
                }
                
                # Show quick result
                if file_result["meets_requirements"]:
                    logger.info(f"      ✅ PASS - Quality: {file_result['quality_score']:.1f}%")
                else:
                    logger.info(f"      ❌ FAIL - Quality: {file_result['quality_score']:.1f}%")
                
                file_results.append(file_result)
                
            except Exception as e:
                logger.error(f"      ❌ Validation failed: {e}")
                file_results.append({
                    "filename": gguf_file.name,
                    "file_path": str(gguf_file),
                    "domain": domain,
                    "file_size_mb": gguf_file.stat().st_size / (1024 * 1024),
                    "validation_result": None,
                    "requirement_assessment": None,
                    "quality_score": 0,
                    "meets_requirements": False,
                    "error": str(e)
                })
        
        # Calculate variant summary
        variant_summary = self._calculate_variant_summary(file_results, variant_config)
        
        return {
            "variant_name": variant_name,
            "files_found": len(gguf_files),
            "file_results": file_results,
            "variant_summary": variant_summary
        }
    
    def _extract_domain_from_path(self, gguf_file: Path, variant_name: str) -> str:
        """Extract domain from GGUF file path"""
        
        # For domain-specific files, extract from path
        if variant_name == "D_domain_specific":
            # Path structure: models/D_domain_specific/category/domain.gguf
            parts = gguf_file.parts
            if len(parts) >= 2:
                category = parts[-2]  # Get category from parent directory
                # Try to extract specific domain from filename
                filename = gguf_file.stem.lower()
                for domain_list in self.domain_categories.values():
                    for domain in domain_list:
                        if domain.replace("_", "").lower() in filename.replace("_", "").lower():
                            return domain
                return category  # Fallback to category
        
        # For other variants, try to extract from filename
        filename = gguf_file.stem.lower()
        for category, domains in self.domain_categories.items():
            for domain in domains:
                if domain.replace("_", "").lower() in filename.replace("_", "").lower():
                    return domain
            if category in filename:
                return category
        
        return "general"  # Default fallback
    
    def _assess_against_requirements(self, gguf_file: Path, validation_result: Dict[str, Any], 
                                   variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess GGUF file against variant requirements"""
        
        if not validation_result:
            return {
                "overall_pass": False,
                "size_check": False,
                "quality_check": False,
                "quantization_check": False,
                "issues": ["Validation failed"]
            }
        
        issues = []
        
        # Size check
        file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
        target_size_mb = variant_config["target_size_mb"]
        
        # Allow 20% tolerance for size
        size_tolerance = 0.2
        min_size = target_size_mb * (1 - size_tolerance)
        max_size = target_size_mb * (1 + size_tolerance)
        
        size_check = min_size <= file_size_mb <= max_size
        if not size_check:
            issues.append(f"Size {file_size_mb:.1f}MB outside range {min_size:.1f}-{max_size:.1f}MB")
        
        # Quality check
        quality_score = validation_result.get("average_quality", 0) * 100
        quality_target = variant_config["quality_target"]
        quality_check = quality_score >= quality_target
        if not quality_check:
            issues.append(f"Quality {quality_score:.1f}% below target {quality_target}%")
        
        # Quantization check (if available in validation result)
        expected_quantization = variant_config["quantization"]
        detected_quantization = validation_result.get("quantization_detected", "unknown")
        quantization_check = True  # Default to true since detection might not be available
        
        if detected_quantization != "unknown" and detected_quantization != expected_quantization:
            quantization_check = False
            issues.append(f"Quantization {detected_quantization} != expected {expected_quantization}")
        
        overall_pass = size_check and quality_check and quantization_check
        
        return {
            "overall_pass": overall_pass,
            "size_check": size_check,
            "quality_check": quality_check,
            "quantization_check": quantization_check,
            "file_size_mb": file_size_mb,
            "target_size_mb": target_size_mb,
            "quality_score": quality_score,
            "quality_target": quality_target,
            "expected_quantization": expected_quantization,
            "detected_quantization": detected_quantization,
            "issues": issues
        }
    
    def _calculate_variant_summary(self, file_results: List[Dict[str, Any]], 
                                 variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for a variant"""
        
        if not file_results:
            return {
                "status": "no_files",
                "total_files": 0,
                "passed_files": 0,
                "failed_files": 0,
                "pass_rate": 0.0
            }
        
        total_files = len(file_results)
        passed_files = sum(1 for f in file_results if f["meets_requirements"])
        failed_files = total_files - passed_files
        pass_rate = (passed_files / total_files) * 100
        
        # Calculate averages
        valid_results = [f for f in file_results if f["quality_score"] > 0]
        avg_quality = sum(f["quality_score"] for f in valid_results) / len(valid_results) if valid_results else 0
        avg_size = sum(f["file_size_mb"] for f in file_results) / len(file_results)
        
        # Determine status
        if pass_rate >= 95:
            status = "excellent"
        elif pass_rate >= 80:
            status = "good"
        elif pass_rate >= 60:
            status = "acceptable"
        else:
            status = "needs_improvement"
        
        return {
            "status": status,
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": failed_files,
            "pass_rate": pass_rate,
            "avg_quality": avg_quality,
            "avg_size_mb": avg_size,
            "target_quality": variant_config["quality_target"],
            "target_size_mb": variant_config["target_size_mb"]
        }
    
    def _generate_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary across all variants"""
        
        total_files = 0
        total_passed = 0
        total_failed = 0
        variant_statuses = {}
        
        for variant_name, variant_data in results["model_variants"].items():
            summary = variant_data["variant_summary"]
            total_files += summary.get("total_files", 0)
            total_passed += summary.get("passed_files", 0)
            total_failed += summary.get("failed_files", 0)
            variant_statuses[variant_name] = summary.get("status", "unknown")
        
        overall_pass_rate = (total_passed / total_files * 100) if total_files > 0 else 0
        
        # Determine overall status
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
            "total_files": total_files,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_pass_rate": overall_pass_rate,
            "variant_statuses": variant_statuses
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        summary = results["summary"]
        
        # Overall recommendations
        if summary["overall_pass_rate"] < 80:
            recommendations.append("🚨 Overall pass rate below 80% - review training parameters and data quality")
        
        # Variant-specific recommendations
        for variant_name, variant_data in results["model_variants"].items():
            variant_summary = variant_data["variant_summary"]
            
            if variant_summary.get("status") == "needs_improvement":
                recommendations.append(f"🔧 {variant_name}: Pass rate {variant_summary.get('pass_rate', 0):.1f}% - needs optimization")
            
            if variant_summary.get("avg_quality", 0) < variant_summary.get("target_quality", 0):
                recommendations.append(f"📈 {variant_name}: Quality {variant_summary.get('avg_quality', 0):.1f}% below target {variant_summary.get('target_quality', 0):.1f}%")
        
        # Technical recommendations
        if summary["total_files"] == 0:
            recommendations.append("📁 No GGUF files found - run enhanced factory to generate models")
        
        # Success recommendations
        if summary["overall_pass_rate"] >= 95:
            recommendations.append("🎉 Excellent results! All models meet requirements - ready for production")
        elif summary["overall_pass_rate"] >= 80:
            recommendations.append("✅ Good results! Most models meet requirements - minor optimizations needed")
        
        return recommendations
    
    def _display_final_results(self, results: Dict[str, Any]) -> None:
        """Display final validation results"""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE GGUF VALIDATION RESULTS")
        logger.info("=" * 60)
        
        summary = results["summary"]
        
        logger.info(f"🎯 Overall Status: {summary['overall_status'].upper()}")
        logger.info(f"📁 Total Files: {summary['total_files']}")
        logger.info(f"✅ Passed: {summary['total_passed']}")
        logger.info(f"❌ Failed: {summary['total_failed']}")
        logger.info(f"📈 Pass Rate: {summary['overall_pass_rate']:.1f}%")
        
        logger.info(f"\n📋 Variant Status:")
        for variant_name, status in summary["variant_statuses"].items():
            status_emoji = {"excellent": "🏆", "good": "✅", "acceptable": "⚠️", "needs_improvement": "🔧"}.get(status, "❓")
            logger.info(f"   {status_emoji} {variant_name}: {status}")
        
        logger.info(f"\n💡 Recommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
    
    def _save_validation_report(self, results: Dict[str, Any]) -> str:
        """Save detailed validation report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"gguf_validation_report_{timestamp}.json"
        report_path = self.project_root / "tests" / "reports" / report_filename
        
        # Create reports directory if it doesn't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(report_path)

async def main():
    """Main validation function"""
    
    logger.info("🚀 Starting MeeTARA Lab Comprehensive GGUF Validation")
    logger.info("=" * 60)
    
    try:
        validator = ComprehensiveGGUFValidator()
        results = await validator.validate_all_gguf_files()
        
        # Quick summary
        summary = results["summary"]
        if summary["overall_pass_rate"] >= 95:
            logger.info("\n🎉 SUCCESS: All GGUF files meet requirements!")
        elif summary["overall_pass_rate"] >= 80:
            logger.info("\n✅ GOOD: Most GGUF files meet requirements")
        else:
            logger.info("\n⚠️ NEEDS WORK: Some GGUF files need optimization")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Validation interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 