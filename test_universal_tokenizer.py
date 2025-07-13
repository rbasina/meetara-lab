#!/usr/bin/env python3
"""
Test universal tokenizer handling for any base model
"""

import sys
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_universal_tokenizer():
    """Test universal tokenizer handling for different models"""
    
    test_models = [
        "microsoft/DialoGPT-small",
        "microsoft/Phi-3.5-mini-instruct", 
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    print("🚀 Testing Universal Tokenizer Handling")
    print("=" * 60)
    
    for model_name in test_models:
        print(f"\n🔍 Testing: {model_name}")
        print("-" * 40)
        
        try:
            from transformers import AutoTokenizer
            
            # Load tokenizer (downloads if needed, uses cache if available)
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            load_time = time.time() - start_time
            
            print(f"✅ Tokenizer loaded in {load_time:.2f}s")
            print(f"   → Pad token: {tokenizer.pad_token}")
            print(f"   → EOS token: {tokenizer.eos_token}")
            print(f"   → Vocab size: {tokenizer.vocab_size}")
            
            # Universal padding token configuration (no model-specific logic)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    print(f"✅ Auto-configured: pad_token={tokenizer.pad_token}")
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print(f"✅ Added new pad token: [PAD]")
            
            # Test tokenization
            test_text = "User: Hello\nAssistant: Hi there!"
            tokens = tokenizer(
                test_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            )
            
            print(f"✅ Tokenization successful")
            print(f"   → Input length: {len(tokens['input_ids'])}")
            print(f"   → Attention mask: {len(tokens['attention_mask'])}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n🎉 Universal tokenizer test completed!")

if __name__ == "__main__":
    test_universal_tokenizer() 