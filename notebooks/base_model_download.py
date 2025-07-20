from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "trinity_core"))

from core_components.config_manager import SmartTrinityConfigManager

# Load base models from config instead of hardcoding
config_manager = SmartTrinityConfigManager()
model_names = config_manager.get_config_dict().get('model_names', {})
if not model_names:
    raise ValueError("❌ No model_names configured")

base_models = list(model_names.values())
print(f"✅ Loading {len(base_models)} models from config:")

for model_name in base_models:
    print(f"📥 Downloading {model_name} ...")
    try:
        _ = AutoTokenizer.from_pretrained(model_name)
        _ = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"✅ Done: {model_name}")
    except Exception as e:
        print(f"❌ Failed: {model_name} ({e})")
print("🎉 All base models are now cached and ready for training!")