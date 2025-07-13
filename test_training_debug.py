#!/usr/bin/env python3
"""
Debug script to identify training issues
"""

import sys
import os
import time
import torch
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "trinity_core"))

def test_model_loading():
    """Test if model loading works"""
    print("🔍 Testing model loading...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "microsoft/DialoGPT-small"
        print(f"📥 Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"✅ Model loaded successfully")
        print(f"   → Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        print(f"   → Device: {next(model.parameters()).device}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_tokenization():
    """Test if tokenization works"""
    print("\n🔍 Testing tokenization...")
    try:
        from transformers import AutoTokenizer
        from datasets import Dataset
        
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test data
        test_data = [
            {"text": "User: Hello\nAssistant: Hi there!"},
            {"text": "User: How are you?\nAssistant: I'm doing well, thank you!"}
        ]
        
        dataset = Dataset.from_list(test_data)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        print(f"✅ Tokenization successful")
        print(f"   → Dataset size: {len(tokenized_dataset)}")
        print(f"   → Sample keys: {list(tokenized_dataset[0].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False

def test_training_args():
    """Test if training arguments work"""
    print("\n🔍 Testing training arguments...")
    try:
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            max_steps=5,  # Very small for testing
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=1,
            logging_steps=1,
            save_steps=5,
            save_strategy="steps",
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=False,
            bf16=False,
        )
        
        print(f"✅ Training arguments created successfully")
        print(f"   → Max steps: {training_args.max_steps}")
        print(f"   → Batch size: {training_args.per_device_train_batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ Training arguments failed: {e}")
        return False

def test_trainer_creation():
    """Test if trainer creation works"""
    print("\n🔍 Testing trainer creation...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from datasets import Dataset
        
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Small test dataset
        test_data = [
            {"text": "User: Hello\nAssistant: Hi there!"},
            {"text": "User: How are you?\nAssistant: I'm doing well!"}
        ]
        dataset = Dataset.from_list(test_data)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=1,
            logging_steps=1,
            save_steps=2,
            save_strategy="steps",
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=False,
            bf16=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        print(f"✅ Trainer created successfully")
        print(f"   → Model: {type(model).__name__}")
        print(f"   → Dataset size: {len(tokenized_dataset)}")
        
        return trainer
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return None

def test_minimal_training():
    """Test minimal training"""
    print("\n🔍 Testing minimal training...")
    try:
        trainer = test_trainer_creation()
        if trainer is None:
            return False
        
        print("🎯 Starting minimal training (2 steps)...")
        start_time = time.time()
        
        # Set a short timeout
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Training timeout")
        
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            print(f"✅ Minimal training completed in {training_time:.2f} seconds")
            return True
            
        except TimeoutError:
            print("❌ Training timed out")
            return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Minimal training setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting training debug tests...")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Tokenization", test_tokenization),
        ("Training Arguments", test_training_args),
        ("Trainer Creation", test_trainer_creation),
        ("Minimal Training", test_minimal_training),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results[test_name] = False
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training should work.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 