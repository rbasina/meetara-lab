import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from meetara_real_model_comparison import RealModelLoader

loader = RealModelLoader()
models = loader.get_available_models()
print("Discovered models:")
for name, info in models.items():
    print(f"{name} -> {info['path']} (Available: {info['available']})")