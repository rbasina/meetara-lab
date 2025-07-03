#!/usr/bin/env python3
"""
Test Enhanced GGUF Factory with TARA Proven Implementations
Tests cleanup utilities, compression techniques, and SpeechBrain integration
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the enhanced GGUF factory
from model_factory.gguf_factory import TrinityGGUFFactory, QuantizationType, CompressionType, CleanupResult

class TestEnhancedGGUFFactory:
    """Test suite for enhanced GGUF factory with TARA proven implementations"""
    
    def setup_method(self):
        """Set up test environment"""
        self.factory = TrinityGGUFFactory()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_tara_proven_parameters(self):
        """Test TARA proven parameters are correctly set"""
        print("🧪 Testing TARA proven parameters...")
        
        # Test core parameters
        assert self.factory.tara_proven_params["batch_size"] == 2
        assert self.factory.tara_proven_params["lora_r"] == 8
        assert self.factory.tara_proven_params["max_steps"] == 846
        assert self.factory.tara_proven_params["output_format"] == "Q4_K_M"
        assert self.factory.tara_proven_params["target_size_mb"] == 8.3
        assert self.factory.tara_proven_params["validation_target"] == 101.0
        
        print("✅ TARA proven parameters validated")
        
    def test_garbage_patterns(self):
        """Test garbage patterns from cleanup_utilities.py"""
        print("🧪 Testing garbage patterns...")
        
        # Test pattern coverage
        expected_patterns = [
            '*.tmp', '*.temp', '*.bak', '*.backup',
            '*.log', '*.cache', '*.lock',
            'checkpoint-*', 'runs/', 'logs/',
            'wandb/', '.git/', '__pycache__/',
            '*.pyc', '*.pyo', '*.pyd'
        ]
        
        for pattern in expected_patterns:
            assert pattern in self.factory.garbage_patterns
        
        print(f"✅ Garbage patterns validated: {len(self.factory.garbage_patterns)} patterns")
        
    def test_voice_categories(self):
        """Test voice categories from enhanced_gguf_factory_v2.py"""
        print("🧪 Testing voice categories...")
        
        # Test all 6 voice categories
        expected_categories = ["meditative", "therapeutic", "professional", "educational", "creative", "casual"]
        
        for category in expected_categories:
            assert category in self.factory.voice_categories
            
            # Test category structure
            category_config = self.factory.voice_categories[category]
            assert "domains" in category_config
            assert "characteristics" in category_config
            assert isinstance(category_config["domains"], list)
            assert isinstance(category_config["characteristics"], dict)
        
        print(f"✅ Voice categories validated: {len(self.factory.voice_categories)} categories")
        
    def test_quantization_types(self):
        """Test quantization types from compression_utilities.py"""
        print("🧪 Testing quantization types...")
        
        # Test all quantization types
        expected_types = ["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"]
        
        for q_type in expected_types:
            assert hasattr(QuantizationType, q_type)
            assert QuantizationType[q_type].value == q_type
        
        print("✅ Quantization types validated")
        
    def test_compression_types(self):
        """Test compression types from compression_utilities.py"""
        print("🧪 Testing compression types...")
        
        # Test all compression types
        expected_types = ["STANDARD", "SPARSE", "HYBRID", "DISTILLED"]
        
        for c_type in expected_types:
            assert hasattr(CompressionType, c_type)
            assert CompressionType[c_type].value == c_type.lower()
        
        print("✅ Compression types validated")
        
    def test_enhanced_gguf_creation(self):
        """Test enhanced GGUF creation with all features"""
        print("🧪 Testing enhanced GGUF creation...")
        
        # Test domain
        test_domain = "healthcare"
        test_training_data = {
            "conversations": [
                {"input": "How are you feeling today?", "output": "I'm doing well, thank you for asking!"},
                {"input": "What's the weather like?", "output": "It's a beautiful day outside!"}
            ]
        }
        
        # Create enhanced GGUF
        result = self.factory.create_gguf_model(test_domain, test_training_data)
        
        # Validate result structure
        assert "domain" in result
        assert "output_file" in result
        assert "enhanced_features" in result
        assert "cleanup_result" in result
        assert "speech_models" in result
        assert "compression" in result
        assert "tara_validation" in result
        
        # Validate cleanup result
        cleanup_result = result["cleanup_result"]
        assert "success" in cleanup_result
        assert "removed_files" in cleanup_result
        assert "validation_score" in cleanup_result
        
        # Validate speech models
        speech_models = result["speech_models"]
        assert speech_models["speechbrain_models"] == 2
        assert speech_models["voice_profiles"] == 6
        
        # Validate compression
        compression = result["compression"]
        assert "method" in compression
        assert "quality_retention" in compression
        assert compression["quality_retention"] == 0.96
        
        print("✅ Enhanced GGUF creation validated")
        
    def test_tara_compatibility_validation(self):
        """Test TARA compatibility validation"""
        print("🧪 Testing TARA compatibility validation...")
        
        # Create test files
        test_gguf = self.temp_dir / "test_model.gguf"
        test_speech_dir = self.temp_dir / "speech_models"
        
        # Create GGUF file
        with open(test_gguf, 'w') as f:
            f.write("# Test GGUF file\n")
            f.write("# Size: 8.3MB\n")
        
        # Create speech models directory structure
        emotion_dir = test_speech_dir / "emotion"
        voice_dir = test_speech_dir / "voice"
        emotion_dir.mkdir(parents=True)
        voice_dir.mkdir(parents=True)
        
        # Create test PKL files
        for i in range(2):
            (emotion_dir / f"test_emotion_{i}.pkl").touch()
        
        for i in range(6):
            (voice_dir / f"test_voice_{i}.pkl").touch()
        
        # Test validation
        validation_result = self.factory._validate_tara_compatibility(str(test_gguf), test_speech_dir)
        
        # Validate result
        assert "tara_compatible" in validation_result
        assert "structure_match" in validation_result
        assert "quality_score" in validation_result
        assert "speechbrain_files" in validation_result
        assert "voice_files" in validation_result
        
        print("✅ TARA compatibility validation tested")
        
    def test_directory_size_calculation(self):
        """Test directory size calculation utility"""
        print("🧪 Testing directory size calculation...")
        
        # Create test files
        test_dir = self.temp_dir / "size_test"
        test_dir.mkdir()
        
        # Create files of known sizes
        (test_dir / "file1.txt").write_text("a" * 1024)  # 1KB
        (test_dir / "file2.txt").write_text("b" * 2048)  # 2KB
        
        # Calculate size
        size_mb = self.factory._get_directory_size_mb(test_dir)
        
        # Validate (should be ~3KB = 0.003MB)
        assert size_mb > 0.002
        assert size_mb < 0.01
        
        print("✅ Directory size calculation validated")
        
    def test_trinity_architecture_config(self):
        """Test Trinity Architecture configuration"""
        print("🧪 Testing Trinity Architecture configuration...")
        
        # Test Trinity config
        trinity_config = self.factory.trinity_config
        
        assert trinity_config["arc_reactor_efficiency"] == 0.90
        assert trinity_config["perplexity_intelligence"] == True
        assert trinity_config["einstein_fusion_target"] == 504.0
        assert trinity_config["speed_improvement"] == "20-100x"
        assert trinity_config["cost_target"] == 50.0
        
        print("✅ Trinity Architecture configuration validated")
        
    def test_quality_thresholds(self):
        """Test quality thresholds by domain category"""
        print("🧪 Testing quality thresholds...")
        
        # Test all domain categories have thresholds
        expected_categories = ["healthcare", "specialized", "business", "education", "technology", "daily_life", "creative"]
        
        for category in expected_categories:
            assert category in self.factory.quality_thresholds
            
            threshold_config = self.factory.quality_thresholds[category]
            assert "min_validation" in threshold_config
            assert "safety_critical" in threshold_config
            assert isinstance(threshold_config["min_validation"], float)
            assert isinstance(threshold_config["safety_critical"], bool)
        
        print("✅ Quality thresholds validated")
        
    def test_comprehensive_integration(self):
        """Test comprehensive integration of all TARA proven features"""
        print("🧪 Testing comprehensive integration...")
        
        # Test multiple domains
        test_domains = ["healthcare", "business", "creative"]
        
        for domain in test_domains:
            print(f"   Testing {domain}...")
            
            # Create test training data
            training_data = {
                "conversations": [
                    {"input": f"Hello from {domain}", "output": f"Hello! I'm your {domain} assistant."}
                ]
            }
            
            # Create GGUF with all enhancements
            result = self.factory.create_gguf_model(domain, training_data)
            
            # Validate comprehensive result
            assert result["domain"] == domain
            assert result["enhanced_features"] == True
            assert result["status"] == "success"
            
            # Validate all enhancement categories
            assert "cleanup_result" in result
            assert "speech_models" in result
            assert "compression" in result
            assert "tara_validation" in result
            
            print(f"   ✅ {domain} integration validated")
        
        print("✅ Comprehensive integration validated")

def test_enhanced_gguf_factory():
    """Run comprehensive test of enhanced GGUF factory"""
    print("🚀 STARTING ENHANCED GGUF FACTORY TESTS")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestEnhancedGGUFFactory()
    test_instance.setup_method()
    
    try:
        # Run all tests
        test_methods = [
            "test_tara_proven_parameters",
            "test_garbage_patterns", 
            "test_voice_categories",
            "test_quantization_types",
            "test_compression_types",
            "test_enhanced_gguf_creation",
            "test_tara_compatibility_validation",
            "test_directory_size_calculation",
            "test_trinity_architecture_config",
            "test_quality_thresholds",
            "test_comprehensive_integration"
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                getattr(test_instance, test_method)()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method} failed: {e}")
                
        print("\n" + "=" * 60)
        print(f"🎯 ENHANCED GGUF FACTORY TEST RESULTS:")
        print(f"   ✅ Passed: {passed_tests}/{total_tests}")
        print(f"   📊 Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TARA PROVEN ENHANCEMENTS VALIDATED!")
            print("   ✅ Cleanup utilities integrated")
            print("   ✅ Compression techniques available")
            print("   ✅ SpeechBrain integration working")
            print("   ✅ Voice profiles created")
            print("   ✅ TARA compatibility validated")
            print("   ✅ Trinity Architecture configured")
            return True
        else:
            print("⚠️ Some tests failed - check integration")
            return False
            
    finally:
        test_instance.teardown_method()

if __name__ == "__main__":
    success = test_enhanced_gguf_factory()
    exit(0 if success else 1) 
