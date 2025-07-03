"""
MeeTARA Lab Testing Configuration
Shared fixtures and test setup for Trinity Architecture components
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'mock_cloud_providers': True,
    'use_synthetic_data': True,
    'cost_protection': True,
    'max_test_duration': 300,  # 5 minutes max per test
    'quality_gates': {
        'min_validation_score': 101,
        'max_model_size_mb': 8.8,
        'min_model_size_mb': 7.8,
        'min_speed_improvement': 20,
        'max_monthly_cost': 50
    }
}

# =====================================
# Core Component Fixtures
# =====================================

@pytest.fixture
def tts_manager():
    """Shared TTS Manager fixture with mock dependencies"""
    from unittest.mock import Mock
    
    mock_tts = Mock()
    mock_tts.voice_categories = {
        'professional': ['en-US-JennyNeural', 'en-US-AriaNeural'],
        'friendly': ['en-US-ChristopherNeural', 'en-US-EricNeural'],
        'authoritative': ['en-US-GuyNeural', 'en-US-DavisNeural'],
        'empathetic': ['en-US-JaneNeural', 'en-US-NancyNeural'],
        'educational': ['en-US-BrianNeural', 'en-US-EmmaNeural'],
        'creative': ['en-US-AshleyNeural', 'en-US-CoraNeural']
    }
    
    # Mock successful voice synthesis
    mock_result = Mock()
    mock_result.status = "success"
    mock_result.audio_data = b"mock_audio_data"
    mock_result.duration = 2.5
    mock_result.voice_used = "en-US-JennyNeural"
    
    mock_tts.synthesize_voice.return_value = mock_result
    mock_tts.get_available_voices.return_value = ['en-US-JennyNeural', 'en-US-AriaNeural']
    
    return mock_tts

@pytest.fixture
def emotion_detector():
    """Shared Emotion Detector fixture with mock RoBERTa model"""
    from unittest.mock import Mock
    
    mock_detector = Mock()
    mock_detector.model_name = "j-hartmann/emotion-english-distilroberta-base"
    
    # Mock emotion detection result
    mock_result = Mock()
    mock_result.emotion = "joy"
    mock_result.confidence = 0.95
    mock_result.context = "professional"
    mock_result.all_emotions = {
        'joy': 0.95,
        'neutral': 0.03,
        'surprise': 0.02
    }
    
    mock_detector.detect_emotion.return_value = mock_result
    mock_detector.analyze_professional_context.return_value = "healthcare"
    
    return mock_detector

@pytest.fixture
def intelligent_router():
    """Shared Intelligent Router fixture with mock routing logic"""
    from unittest.mock import Mock
    
    mock_router = Mock()
    
    # Mock routing result
    mock_result = Mock()
    mock_result.domain = "healthcare"
    mock_result.tier = "balanced"
    mock_result.provider = "colab"
    mock_result.estimated_cost = 8.50
    mock_result.estimated_time = 240  # 4 minutes
    
    mock_router.route_request.return_value = mock_result
    mock_router.analyze_domain.return_value = "healthcare"
    mock_router.select_tier.return_value = "balanced"
    
    return mock_router

@pytest.fixture
def gguf_factory():
    """Shared GGUF Factory fixture with mock model creation"""
    from unittest.mock import Mock
    
    mock_factory = Mock()
    mock_factory.proven_parameters = {
        'batch_size': 6,
        'lora_r': 8,
        'max_steps': 846,
        'learning_rate': 3e-4,
        'quantization': 'Q4_K_M'
    }
    
    # Mock model creation result
    mock_model = Mock()
    mock_model.path = "/mock/path/test_model.gguf"
    mock_model.size_mb = 8.3
    mock_model.validation_score = 101
    mock_model.creation_time = 180  # 3 minutes
    mock_model.tier = "lightning"
    
    mock_factory.create_gguf.return_value = mock_model
    mock_factory.validate_quality.return_value = 101
    mock_factory.get_model_size.return_value = 8.3
    
    return mock_factory

@pytest.fixture
def training_orchestrator():
    """Shared Training Orchestrator fixture with mock training coordination"""
    from unittest.mock import Mock
    
    mock_orchestrator = Mock()
    
    # Mock training result
    mock_result = Mock()
    mock_result.status = "completed"
    mock_result.model_path = "/mock/path/trained_model.gguf"
    mock_result.training_time = 180
    mock_result.cost = 3.50
    mock_result.quality_score = 101
    
    mock_orchestrator.train_domain.return_value = mock_result
    mock_orchestrator.get_training_status.return_value = "completed"
    mock_orchestrator.estimate_cost.return_value = 3.50
    
    return mock_orchestrator

# =====================================
# Cloud & Infrastructure Fixtures
# =====================================

@pytest.fixture
def mock_cloud_provider():
    """Mock cloud provider for safe testing without actual cloud charges"""
    from unittest.mock import Mock
    
    mock_provider = Mock()
    mock_provider.name = "mock_colab"
    mock_provider.gpu_type = "T4"
    mock_provider.cost_per_hour = 0.35
    mock_provider.available = True
    
    # Mock training methods
    mock_provider.start_training.return_value = {"job_id": "mock_job_123"}
    mock_provider.get_status.return_value = "running"
    mock_provider.get_cost.return_value = 2.50
    mock_provider.stop_training.return_value = True
    
    return mock_provider

@pytest.fixture
def gpu_orchestrator(mock_cloud_provider):
    """Mock GPU Orchestrator with cost protection"""
    from unittest.mock import Mock
    
    mock_orchestrator = Mock()
    mock_orchestrator.providers = {
        'colab': mock_cloud_provider,
        'lambda': mock_cloud_provider,
        'runpod': mock_cloud_provider,
        'vast': mock_cloud_provider
    }
    
    mock_orchestrator.select_optimal_provider.return_value = mock_cloud_provider
    mock_orchestrator.get_available_providers.return_value = [mock_cloud_provider]
    mock_orchestrator.estimate_cost.return_value = 3.50
    
    return mock_orchestrator

@pytest.fixture
def cost_monitor():
    """Mock Cost Monitor with budget protection"""
    from unittest.mock import Mock
    
    mock_monitor = Mock()
    mock_monitor.monthly_budget = 50.0
    mock_monitor.current_spend = 15.50
    mock_monitor.spending_alerts = [0.5, 0.8, 0.9, 0.95]
    
    mock_monitor.get_current_spend.return_value = 15.50
    mock_monitor.check_budget.return_value = True
    mock_monitor.estimate_remaining_budget.return_value = 34.50
    
    return mock_monitor

# =====================================
# Trinity Architecture Fixtures
# =====================================

@pytest.fixture
def trinity_intelligence():
    """Mock Trinity Intelligence with fusion capabilities"""
    from unittest.mock import Mock
    
    mock_trinity = Mock()
    
    # Mock Trinity enhancement result
    mock_result = Mock()
    mock_result.amplification_factor = 5.04  # 504% improvement
    mock_result.arc_reactor_efficiency = 0.90  # 90% efficiency
    mock_result.perplexity_intelligence = True
    mock_result.einstein_fusion = True
    
    mock_trinity.process_with_trinity.return_value = mock_result
    mock_trinity.get_amplification_factor.return_value = 5.04
    
    return mock_trinity

# =====================================
# Test Data Fixtures
# =====================================

@pytest.fixture
def test_domain_data():
    """Safe synthetic domain data for testing"""
    return {
        "domain": "test_healthcare",
        "training_data": "Safe synthetic medical conversation data for testing...",
        "tier": "lightning",
        "expected_cost": 3.50,
        "expected_time": 180,
        "quality_target": 101
    }

@pytest.fixture
def test_domains():
    """Multiple test domains for batch testing"""
    return [
        {"domain": "test_healthcare", "tier": "lightning"},
        {"domain": "test_education", "tier": "fast"},
        {"domain": "test_business", "tier": "balanced"},
        {"domain": "test_creative", "tier": "quality"}
    ]

@pytest.fixture
def mock_model_data():
    """Mock GGUF model data for testing"""
    return {
        "model_bytes": b"Mock GGUF model data for testing...",
        "size_mb": 8.3,
        "format": "GGUF",
        "quantization": "Q4_K_M",
        "validation_score": 101
    }

# =====================================
# Performance Testing Fixtures
# =====================================

@pytest.fixture
def performance_targets():
    """Performance targets for validation"""
    return {
        "speed_improvement": {
            "t4_gpu": 37,  # 37x faster than CPU
            "v100_gpu": 75,  # 75x faster than CPU
            "a100_gpu": 151  # 151x faster than CPU
        },
        "cost_targets": {
            "lightning": 3.0,  # $2-3
            "fast": 4.0,       # $3-5
            "balanced": 10.0,  # $8-12
            "quality": 12.5    # $10-15
        },
        "quality_targets": {
            "min_validation_score": 101,
            "max_model_size": 8.8,
            "min_model_size": 7.8,
            "max_loading_time": 100,  # 100ms
            "max_memory_usage": 15    # 15MB
        }
    }

# =====================================
# Test Environment Setup
# =====================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with safety measures"""
    # Ensure we're using mock providers by default
    os.environ['TESTING_MODE'] = 'true'
    os.environ['USE_MOCK_PROVIDERS'] = 'true'
    os.environ['COST_PROTECTION'] = 'true'
    
    # Set safe test limits
    os.environ['MAX_TEST_COST'] = '0.10'  # 10 cents max per test
    os.environ['MAX_TEST_DURATION'] = '300'  # 5 minutes max
    
    yield
    
    # Cleanup after tests
    test_env_vars = [
        'TESTING_MODE', 'USE_MOCK_PROVIDERS', 'COST_PROTECTION',
        'MAX_TEST_COST', 'MAX_TEST_DURATION'
    ]
    for var in test_env_vars:
        os.environ.pop(var, None)

# =====================================
# Test Utilities
# =====================================

@pytest.fixture
def assert_quality_gates():
    """Custom assertion fixture for quality gates"""
    def _assert_quality_gates(result: Dict[str, Any]):
        """Assert that result meets all quality gates"""
        gates = TEST_CONFIG['quality_gates']
        
        if 'validation_score' in result:
            assert result['validation_score'] >= gates['min_validation_score'], \
                f"Validation score {result['validation_score']} below minimum {gates['min_validation_score']}"
        
        if 'model_size_mb' in result:
            assert gates['min_model_size_mb'] <= result['model_size_mb'] <= gates['max_model_size_mb'], \
                f"Model size {result['model_size_mb']}MB outside range {gates['min_model_size_mb']}-{gates['max_model_size_mb']}MB"
        
        if 'speed_improvement' in result:
            assert result['speed_improvement'] >= gates['min_speed_improvement'], \
                f"Speed improvement {result['speed_improvement']}x below minimum {gates['min_speed_improvement']}x"
        
        if 'monthly_cost' in result:
            assert result['monthly_cost'] <= gates['max_monthly_cost'], \
                f"Monthly cost ${result['monthly_cost']} exceeds maximum ${gates['max_monthly_cost']}"
    
    return _assert_quality_gates

# =====================================
# Test Markers
# =====================================

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "cost_sensitive: mark test that could incur costs"
    )

# =====================================
# Test Collection Hooks
# =====================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers"""
    for item in items:
        # Auto-mark tests based on path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        
        # Mark cloud-related tests as cost sensitive
        if any(keyword in item.name.lower() for keyword in ['cloud', 'gpu', 'colab', 'training']):
            item.add_marker(pytest.mark.cost_sensitive) 
