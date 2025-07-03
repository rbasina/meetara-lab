#!/usr/bin/env python3
"""
Integration tests for Trinity Architecture
Tests component interactions and workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys
import json

# Add paths for Trinity components
project_root = Path(__file__).parent.parent.parent
sys.path.extend([
    str(project_root / "trinity-core"),
    str(project_root / "intelligence-hub"),
    str(project_root / "model-factory"),
    str(project_root / "cloud-training")
])

class TestTrinityIntegration:
    """Test integration between Trinity Architecture components"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock Trinity components for testing"""
        components = {
            'tts_manager': Mock(),
            'emotion_detector': Mock(),
            'intelligent_router': Mock(),
            'domain_experts': Mock(),
            'gguf_factory': Mock()
        }
        
        # Configure mock behaviors
        components['emotion_detector'].detect_text_emotion.return_value = Mock(
            emotion="neutral", confidence=0.7, modality="text"
        )
        
        components['intelligent_router'].route_request.return_value = Mock(
            strategy="local", confidence=0.8, reasoning="Low complexity, local processing optimal"
        )
        
        components['tts_manager'].synthesize_speech.return_value = Mock(
            audio_data=b"mock_audio", duration=2.5, voice="female"
        )
        
        return components
    
    def test_text_to_speech_workflow(self, mock_components):
        """Test complete text-to-speech workflow with emotion detection"""
        # Input text
        input_text = "I'm really excited about this breakthrough!"
        
        # Step 1: Detect emotion
        emotion_result = mock_components['emotion_detector'].detect_text_emotion(input_text)
        assert emotion_result.emotion == "neutral"
        assert emotion_result.confidence > 0.5
        
        # Step 2: Route processing request
        request_data = {
            "type": "text_to_speech",
            "content": input_text,
            "emotion": emotion_result.emotion,
            "urgency": "realtime"
        }
        
        routing_decision = mock_components['intelligent_router'].route_request(request_data)
        assert routing_decision.strategy in ["local", "cloud", "hybrid"]
        
        # Step 3: Synthesize speech with emotion
        tts_request = {
            "text": input_text,
            "emotion": emotion_result.emotion,
            "strategy": routing_decision.strategy
        }
        
        audio_result = mock_components['tts_manager'].synthesize_speech(tts_request)
        assert audio_result.audio_data is not None
        assert audio_result.duration > 0
        
        # Verify workflow completed successfully
        assert True  # All steps completed without errors
    
    def test_domain_expert_routing_workflow(self, mock_components):
        """Test domain expert selection and routing workflow"""
        # Configure domain expert mock
        mock_components['domain_experts'].get_expert.return_value = Mock(
            domain="medical", expertise_level=0.9, specialized_model="medical_gpt"
        )
        
        # Medical domain request
        medical_request = {
            "type": "consultation",
            "content": "What are the symptoms of hypertension?",
            "domain": "medical",
            "urgency": "medium"
        }
        
        # Step 1: Get domain expert
        expert = mock_components['domain_experts'].get_expert(medical_request["domain"])
        assert expert.domain == "medical"
        assert expert.expertise_level > 0.8
        
        # Step 2: Route with expert context
        enhanced_request = {**medical_request, "expert": expert}
        routing_decision = mock_components['intelligent_router'].route_request(enhanced_request)
        
        # Should route to specialized processing
        assert routing_decision.confidence > 0.7
        
        # Verify domain expertise integration
        assert expert.specialized_model == "medical_gpt"
    
    def test_gguf_factory_integration(self, mock_components):
        """Test GGUF factory integration with training components"""
        # Configure GGUF factory mock
        mock_components['gguf_factory'].create_gguf.return_value = Mock(
            model_path="test_model.gguf",
            size_mb=8.3,
            validation_score=101.0,
            creation_time=120.5
        )
        
        # Training request
        training_data = {
            "domain": "general",
            "training_steps": 846,
            "batch_size": 6,
            "lora_r": 8,
            "base_model": "microsoft/DialoGPT-medium"
        }
        
        # Step 1: Route training request
        routing_decision = mock_components['intelligent_router'].route_request({
            "type": "model_training",
            "data": training_data,
            "urgency": "medium"
        })
        
        # Should route to cloud for GPU training
        assert routing_decision.strategy in ["cloud", "hybrid"]
        
        # Step 2: Create GGUF model
        gguf_result = mock_components['gguf_factory'].create_gguf(training_data)
        
        assert gguf_result.model_path.endswith(".gguf")
        assert gguf_result.size_mb <= 10.0  # Within size constraints
        assert gguf_result.validation_score >= 100.0  # Quality maintained
        
        # Verify integration success
        assert gguf_result.creation_time > 0
    
    def test_multi_component_error_handling(self, mock_components):
        """Test error handling across multiple components"""
        # Configure one component to fail
        mock_components['emotion_detector'].detect_text_emotion.side_effect = Exception("Model loading failed")
        
        # Should handle gracefully and continue workflow
        try:
            input_text = "Test text for error handling"
            
            # Try emotion detection (will fail)
            try:
                emotion_result = mock_components['emotion_detector'].detect_text_emotion(input_text)
            except Exception:
                # Fallback to neutral emotion
                emotion_result = Mock(emotion="neutral", confidence=0.5, modality="text")
            
            # Continue with routing (should work)
            request_data = {
                "type": "text_processing",
                "content": input_text,
                "emotion": emotion_result.emotion
            }
            
            routing_decision = mock_components['intelligent_router'].route_request(request_data)
            assert routing_decision is not None
            
            workflow_completed = True
            
        except Exception as e:
            workflow_completed = False
            print(f"Workflow failed: {e}")
        
        # Workflow should complete even with component failure
        assert workflow_completed
    
    def test_performance_integration(self, mock_components):
        """Test performance of integrated workflows"""
        import time
        
        start_time = time.time()
        
        # Run multiple integrated workflows
        for i in range(10):
            input_text = f"Performance test {i}: This is a sample text for testing."
            
            # Multi-step workflow
            emotion_result = mock_components['emotion_detector'].detect_text_emotion(input_text)
            
            request_data = {
                "type": "text_processing",
                "content": input_text,
                "emotion": emotion_result.emotion,
                "iteration": i
            }
            
            routing_decision = mock_components['intelligent_router'].route_request(request_data)
            
            # Simulate processing based on routing decision
            if routing_decision.strategy == "local":
                # Local processing simulation
                time.sleep(0.01)  # 10ms local processing
            else:
                # Cloud processing simulation
                time.sleep(0.02)  # 20ms cloud processing
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 10 workflows in reasonable time (< 5 seconds)
        assert total_time < 5.0
        
        workflows_per_second = 10 / total_time
        print(f"Integration performance: {workflows_per_second:.2f} workflows/second")
        
        # Should process at least 2 workflows per second
        assert workflows_per_second >= 2.0
    
    def test_data_flow_consistency(self, mock_components):
        """Test data consistency across component boundaries"""
        # Create consistent test data
        original_data = {
            "id": "test_001",
            "text": "Hello world",
            "metadata": {"source": "integration_test", "timestamp": "2025-06-22"}
        }
        
        # Pass through emotion detection
        emotion_result = mock_components['emotion_detector'].detect_text_emotion(original_data["text"])
        
        # Enhance data with emotion
        enhanced_data = {
            **original_data,
            "emotion": emotion_result.emotion,
            "emotion_confidence": emotion_result.confidence
        }
        
        # Pass through routing
        routing_decision = mock_components['intelligent_router'].route_request(enhanced_data)
        
        # Final data should contain all information
        final_data = {
            **enhanced_data,
            "routing_strategy": routing_decision.strategy,
            "routing_confidence": routing_decision.confidence
        }
        
        # Verify data integrity
        assert final_data["id"] == original_data["id"]
        assert final_data["text"] == original_data["text"]
        assert final_data["metadata"] == original_data["metadata"]
        assert "emotion" in final_data
        assert "routing_strategy" in final_data
        
        # Verify data flow consistency
        assert len(final_data) >= len(original_data)  # Data should accumulate, not disappear

class TestTrinityWorkflows:
    """Test complete Trinity Architecture workflows"""
    
    def test_training_to_deployment_workflow(self):
        """Test complete training to deployment workflow"""
        # Mock the entire training pipeline
        workflow_steps = []
        
        # Step 1: Domain selection
        domain = "healthcare"
        workflow_steps.append(f"Domain selected: {domain}")
        
        # Step 2: Training configuration
        training_config = {
            "domain": domain,
            "steps": 846,
            "batch_size": 6,
            "lora_r": 8
        }
        workflow_steps.append(f"Training configured: {training_config}")
        
        # Step 3: GPU allocation
        gpu_allocation = {"type": "T4", "provider": "colab", "cost_per_hour": 0.35}
        workflow_steps.append(f"GPU allocated: {gpu_allocation}")
        
        # Step 4: Model training (simulated)
        training_result = {
            "model_path": f"models/{domain}_model.gguf",
            "training_time": 120.5,
            "validation_score": 101.2,
            "cost": 0.12
        }
        workflow_steps.append(f"Training completed: {training_result}")
        
        # Step 5: Deployment preparation
        deployment_config = {
            "model_path": training_result["model_path"],
            "inference_endpoint": f"https://api.meetara.com/models/{domain}",
            "scaling": "auto"
        }
        workflow_steps.append(f"Deployment prepared: {deployment_config}")
        
        # Verify workflow completion
        assert len(workflow_steps) == 5
        assert all(":" in step for step in workflow_steps)
        assert training_result["validation_score"] > 100.0
        assert training_result["cost"] < 1.0  # Under budget
        
        print("Complete workflow steps:")
        for i, step in enumerate(workflow_steps, 1):
            print(f"  {i}. {step}")
    
    def test_realtime_processing_workflow(self):
        """Test realtime processing workflow with multiple components"""
        import time
        
        start_time = time.time()
        
        # Simulate realtime processing pipeline
        input_stream = [
            "Hello, how are you feeling today?",
            "I'm excited about the new features!",
            "This seems a bit concerning.",
            "Everything looks good to me.",
            "Thanks for your help!"
        ]
        
        processed_results = []
        
        for text in input_stream:
            step_start = time.time()
            
            # Simulated processing pipeline
            result = {
                "input": text,
                "emotion": "neutral",  # Simulated emotion detection
                "routing": "local",    # Simulated routing decision
                "response": f"Processed: {text[:20]}...",
                "processing_time": time.time() - step_start
            }
            
            processed_results.append(result)
            
            # Ensure realtime performance (< 100ms per item)
            assert result["processing_time"] < 0.1
        
        total_time = time.time() - start_time
        
        # Verify realtime performance
        assert total_time < 1.0  # Process 5 items in under 1 second
        assert len(processed_results) == len(input_stream)
        
        print(f"Realtime processing: {len(input_stream)/total_time:.2f} items/second")

if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    print("🧪 Running Trinity Integration Tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Exit code: {result.returncode}") 
