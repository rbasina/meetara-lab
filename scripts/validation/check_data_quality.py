import json
from pathlib import Path

def check_training_data():
    data_file = Path('model-factory/real_training_data/daily_life/parenting_training_data.json')
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    print("✅ TRAINING DATA ANALYSIS:")
    print(f"   → Total samples: {len(samples)}")
    print(f"   → Generated at: {data['generation_timestamp']}")
    
    # Check sample structure
    sample = samples[0]
    print(f"\n📋 Sample structure:")
    print(f"   → Keys: {list(sample.keys())}")
    
    # Check uniqueness
    print(f"\n📈 UNIQUENESS TEST:")
    inputs = [sample.get('input', '') for sample in samples[:100]]
    outputs = [sample.get('output', '') for sample in samples[:100]]
    
    unique_inputs = len(set(inputs))
    unique_outputs = len(set(outputs))
    
    print(f"   → Unique inputs: {unique_inputs}/100 ({unique_inputs}%)")
    print(f"   → Unique outputs: {unique_outputs}/100 ({unique_outputs}%)")
    
    if unique_inputs > 80 and unique_outputs > 80:
        print("   ✅ HIGH UNIQUENESS - This IS real-time generated!")
        return True
    else:
        print("   ❌ Low uniqueness detected")
        return False
    
    # Show sample content
    print(f"\n🔍 Sample content:")
    print(f"   Input: {sample.get('input', 'N/A')[:100]}...")
    print(f"   Output: {sample.get('output', 'N/A')[:100]}...")

if __name__ == "__main__":
    check_training_data() 