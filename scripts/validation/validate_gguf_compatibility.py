import os
from pathlib import Path
from llama_cpp import Llama

# Path to D_domain_specific models (update if needed)
BASE_DIR = Path(__file__).parent.parent.parent / "models" / "production" / "D_domain_specific"

report = []

def validate_gguf_file(gguf_path):
    try:
        llm = Llama(model_path=str(gguf_path))
        del llm
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print(f"Scanning: {BASE_DIR}")
    for category in BASE_DIR.iterdir():
        if not category.is_dir():
            continue
        for domain in category.iterdir():
            if not domain.is_dir():
                continue
            for file in domain.glob("*.gguf"):
                ok, err = validate_gguf_file(file)
                status = "OK" if ok else f"FAIL: {err}"[:120]
                report.append((str(file), status))
                print(f"{file}: {status}")

    print("\n=== GGUF Compatibility Report ===")
    for file, status in report:
        print(f"{file}: {status}")

if __name__ == "__main__":
    main() 