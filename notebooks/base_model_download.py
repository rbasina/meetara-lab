from transformers import AutoModelForCausalLM, AutoTokenizer

# Only the base models actually used in your training configs

base_models = [
    "microsoft/Phi-3-medium-4k-instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceTB/SmolLM2-1.7B",
]

for model_name in base_models:
    print(f"📥 Downloading {model_name} ...")
    try:
        _ = AutoTokenizer.from_pretrained(model_name)
        _ = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"✅ Done: {model_name}")
    except Exception as e:
        print(f"❌ Failed: {model_name} ({e})")
print("🎉 All base models are now cached and ready for training!")