# MeeTARA Lab: Google Colab Execution Steps

This guide provides a robust, step-by-step workflow for setting up and running MeeTARA Lab in Google Colab. It covers package installation, repo cloning, building `llama.cpp` with CUDA, and running the training pipeline. **Follow these steps every time you start a new Colab session.**

---

## 🚦 **Step 0a: Uninstall and Install Core Packages (Expect Restart)**

```python
# Uninstall any conflicting core packages (safe to run even if not present)
!pip uninstall -y torch torchvision torchaudio transformers numpy

# Install PyTorch with CUDA 11.8 (works for T4/V100/A100 GPUs in Colab)
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install numpy (compatible version)
!pip install numpy==1.26.4

# Install transformers (core LLM library)
!pip install transformers==4.35.0
```

> **After running this cell, Colab will likely restart the runtime.**
> Once it restarts, re-run this cell (it will be fast if already installed), then continue to the next cell.

---

## 🚦 **Step 0b: Install Non-Core Packages (No Restart Expected)**

```python
# Install related LLM and utility packages
!pip install datasets peft accelerate bitsandbytes

# Install HuggingFace and monitoring tools
!pip install huggingface_hub wandb tensorboard

# Install GGUF and llama.cpp Python bindings (optional, but safe)
!pip install gguf llama-cpp-python

# Install audio processing tools
!pip install speechbrain librosa soundfile

# Install computer vision tools
!pip install opencv-python Pillow

# Install utility packages
!pip install pyyaml tqdm rich
```

---

## 🚦 **Step 1: Clean Up and Clone Repos**

```python
# Clean up any old directories
!rm -rf /content/meetara-lab

# Clone MeeTARA Lab
!git clone https://github.com/rbasina/meetara-lab.git /content/meetara-lab

# Go into the project directory
%cd /content/meetara-lab

# Remove any old or broken llama.cpp
!rm -rf llama.cpp

# Clone the official llama.cpp repo
!git clone https://github.com/ggerganov/llama.cpp.git

# Verify CMakeLists.txt exists
!ls -l /content/meetara-lab/llama.cpp/CMakeLists.txt
```

---

## 🚦 **Step 2: Build llama.cpp with CUDA**

```python
# Enter llama.cpp directory
%cd /content/meetara-lab/llama.cpp

# Remove any old build directory
!rm -rf build

# Create and enter build directory
!mkdir build
%cd build

# Run CMake with CUDA support
!cmake .. -DGGML_CUDA=ON

# Build the binaries
!make -j$(nproc)
```

---

## 🚦 **Step 3: Verify the Build**

```python
# Check for built binaries
!ls -la /content/meetara-lab/llama.cpp/build/bin/
```
You should see files like `main`, `quantize`, etc.

---

## 🚦 **Step 4: (Optional) Install GGUF Python Tools**

```python
%cd /content/meetara-lab/llama.cpp/gguf-py
!pip install -e .
%cd /content/meetara-lab
```
*Only needed if you use the Python GGUF tools for conversion.*

---

## 🚦 **Step 5: Run Your Training/Conversion Pipeline**

```python
# Example: Run the Trinity Production Launcher for a specific domain category
# Uncomment the line for the category you want to train

# Healthcare domains (11 domains)
# !python cloud-training/production_launcher.py --category healthcare --environment production

# Business domains (12 domains)
# !python cloud-training/production_launcher.py --category business --environment production

# Education domains (8 domains)
# !python cloud-training/production_launcher.py --category education --environment production

# Daily Life domains (12 domains)
# !python cloud-training/production_launcher.py --category daily_life --environment production

# Creative domains (8 domains)
# !python cloud-training/production_launcher.py --category creative --environment production

# Technology domains (6 domains)
# !python cloud-training/production_launcher.py --category technology --environment production

# Specialized domains (4 domains)
# !python cloud-training/production_launcher.py --category specialized --environment production

print("✅ Category-specific training options available")
```

---

## 🟢 **Best Practices & Troubleshooting**

- **Always run each cell in order.**
- **If Colab restarts after package install, re-run all setup cells from the top.**
- **If you get any error, check the output of the previous cell (especially for path or build errors).**
- **If you want to run all domains at once, you can loop over the categories in Python.**
- **If you want to save time, you can skip the GGUF Python tools install unless you need them.**

---

**This guide ensures a clean, reliable setup and build for MeeTARA Lab in Colab every time!** 