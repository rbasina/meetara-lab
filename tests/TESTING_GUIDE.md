# 🧪 MeeTARA Lab - Voice & Translation Testing Guide

This guide provides comprehensive instructions for testing your voice recognition and translation models using the generated GGUF files and translation models.

## 📋 Prerequisites

Before running tests, ensure you have:

1. **✅ MeeTARA Lab Setup Complete**
   - All models generated successfully
   - Translation models created (Hindi, Telugu)
   - GGUF files available in `models/production/B_universal/`

2. **🔨 llama.cpp Built**
   - Clone: `git clone https://github.com/ggerganov/llama.cpp.git`
   - Build: `cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release`

3. **🐍 Python Environment**
   - Python 3.8+ with required packages
   - PyAudio (for real speech recognition)

## 🚀 Quick Start Testing

### Option 1: PowerShell Script (Recommended)
```powershell
# Run from meetara-lab root directory
.\tests\test_voice_translation.ps1
```

**Features:**
- Comprehensive testing of all components
- Color-coded output for easy reading
- Detailed error reporting
- Results saved to JSON file
- Interactive testing with GGUF inference

**Note:** If you encounter PowerShell parsing errors, use the batch file alternative below.

### Option 2: Batch File (Windows Alternative)
```cmd
# Run from meetara-lab root directory
tests\test_voice_translation_simple.bat
```

**Features:**
- Simple Windows batch file (no PowerShell dependencies)
- Basic model existence checks
- File size reporting
- No parsing errors
- Suitable for automated testing

### Option 3: Python Quick Test
```bash
# Run from meetara-lab root directory
python tests/quick_test.py
```

### Option 3: Comprehensive Python Test
```bash
# Run from meetara-lab root directory
python tests/voice_translation_test.py
```

### Option 4: Complete Voice Pipeline Test ⭐ **NEW**
```bash
# Run from meetara-lab root directory
python tests/voice_pipeline_test.py
```

**What it tests**:
- **Complete end-to-end voice pipeline**
- Speech Recognition (Hindi/Telugu → Text)
- Translation (Hindi/Telugu → English)
- LLM Processing (GGUF inference)
- Response Translation (English → Hindi/Telugu)
- Text-to-Speech (Voice output)
- **Pipeline flow simulation and validation**

## 🧪 Test Categories

### 1. 🤖 GGUF Model Testing
**Purpose**: Validate the generated GGUF model works with llama.cpp

**What it tests**:
- Model file existence and size
- llama.cpp compatibility
- Basic inference capabilities
- Translation prompts

**Expected output**:
```
✅ Found GGUF model: meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf
✅ Model size: 4.4 GB
🔄 Testing with prompt: Translate 'Hello, how are you?' to Hindi
🚀 Running: llama-cli.exe -m model.gguf -p "prompt" -n 50 -t 4 --temp 0.7
✅ GGUF inference successful!
📝 Output: [Hindi translation response]
```

**Manual testing**:
```bash
cd llama.cpp/build/bin
./llama-cli.exe -m "../../../models/production/B_universal/meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf" -p "Translate 'Hello, how are you?' to Hindi" -n 100 -t 4 --temp 0.7
```

### 2. 🌐 Translation Models Testing
**Purpose**: Validate offline translation capabilities

**What it tests**:
- Hindi model availability (194MB)
- Telugu model availability (1.6GB)
- Bundle configuration
- Model file integrity

**Expected output**:
```
✅ Hindi model found: models/production/translation_models/translation_bundle_20250809_222911/hi_model
   Model size: 194.0 MB
✅ Telugu model found: models/production/translation_models/translation_bundle_20250809_222911/te_model
   Model size: 1.6 GB
✅ Bundle config: translation_bundle_20250809_222911
   Languages: hi, te
```

**Manual testing**:
```python
from trinity_core.agents.translation_factory import TranslationFactory
from trinity_core.core_components.translation_models import TranslationRequest

factory = TranslationFactory()

# Test Hindi to English
request = TranslationRequest(
    text="नमस्ते, कैसे हो आप?",
    source_language="hi",
    target_language="en",
    quality="high",
    method="offline"
)

result = await factory.translate(request)
print(f"Hindi → English: {result.translated_text}")
```

### 3. 🎤 Speech Models Testing
**Purpose**: Validate speech recognition and voice capabilities

**What it tests**:
- Speech models directory structure
- Voice mapping configurations
- Emotion detection models
- Routing models

**Expected output**:
```
✅ Speech models directory found: models/production/speech_models
   ✅ emotion: models/production/speech_models/emotion
   ✅ voice: models/production/speech_models/voice
   ✅ routing: models/production/speech_models/routing
✅ Speech config loaded: X voice mappings
```

**Manual testing**:
```python
from trinity_core.core_components.speech_recognition import SpeechRecognition

sr = SpeechRecognition()

if sr.pyaudio_available:
    print("🎤 Real-time speech recognition available")
    # Start listening
    result = await sr.start_recognition()
    print(f"Recognized: {result}")
else:
    print("📝 Text input mode only")
```

### 4. 🎯 Voice → Translation Pipeline Testing
**Purpose**: Test complete end-to-end workflow

**What it tests**:
- Voice input (real or simulated)
- Speech-to-text conversion
- Text translation (English → Hindi/Telugu)
- Pipeline integration

**Expected output**:
```
🎯 Testing Complete Voice → Translation Pipeline...
🎤 Voice input detected: 'Hello, how are you today?'
🔄 Testing English → Hindi translation...
✅ Translation successful: [Hindi text]
🔄 Testing English → Telugu translation...
✅ Translation successful: [Telugu text]
✅ Voice → Translation pipeline completed successfully!
```

### 5. 🚀 Complete Voice Pipeline Testing ⭐ **NEW**
**Purpose**: Validate the complete end-to-end voice interaction flow

**What it tests**:
- **Speech Recognition**: User speaks in Hindi/Telugu → Text conversion
- **Translation**: Hindi/Telugu text → English translation
- **LLM Processing**: English input → GGUF model → English response
- **Response Translation**: English response → Hindi/Telugu translation
- **Text-to-Speech**: Hindi/Telugu text → Voice output

**Pipeline Flow**:
```
User speaks in Hindi → Speech Recognition → Hindi Text → 
Translation (Hindi→English) → English Text → 
LLM Processing (GGUF) → English Response → 
Translation (English→Hindi) → Hindi Response → 
Text-to-Speech → Hindi Voice Output
```

**Expected output**:
```
✅ Speech Recognition: success
✅ Translation: success (Hindi→English, Telugu→English)
✅ LLM Processing: success (GGUF inference working)
✅ Response Translation: success (English→Hindi, English→Telugu)
✅ Text-to-Speech: success
🎯 Pipeline completion: 100.0% (5/5 steps)
🎉 Complete voice pipeline is working!
```

**Manual testing**:
```bash
# Test complete pipeline
python tests/voice_pipeline_test.py

# This will test each component and show the complete flow
```

## 🔍 Troubleshooting Common Issues

### Issue: PowerShell Parsing Errors
**Symptoms**:
```
ParserError: The term '$_:' is not recognized as the name of a cmdlet, function, script file, or operable program.
ParserError: Unterminated string.
```

**Solutions**:
1. **Use the batch file alternative**: `tests\test_voice_translation_simple.bat`
2. **Check PowerShell version**: Ensure you're using PowerShell 5.1 or later
3. **Run as Administrator**: Right-click PowerShell and "Run as Administrator"
4. **Execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Issue: llama-cli.exe not found
**Solution**:
```bash
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Issue: GGUF model not found
**Check**:
- Model path: `models/production/B_universal/meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf`
- File size should be ~4.4GB
- Run factory script again if missing

### Issue: Translation models not found
**Check**:
- Bundle directory: `models/production/translation_models/translation_bundle_20250809_222911/`
- Hindi model: `hi_model/model.pt` (194MB)
- Telugu model: `te_model/model.pt` (1.6GB)

### Issue: PyAudio not available
**Solution**:
```bash
pip install pyaudio
# On Windows, you might need:
pip install pipwin
pipwin install pyaudio
```

### Issue: Memory errors during GGUF inference
**Solutions**:
- Reduce threads: `-t 2` instead of `-t 4`
- Reduce max tokens: `-n 25` instead of `-n 50`
- Close other applications
- Use lower quantization if available

## 📊 Test Results Interpretation

### ✅ All Tests Passed
Your MeeTARA Lab is fully operational for:
- Voice input and recognition
- Offline translation (Hindi, Telugu)
- GGUF model inference
- Complete voice → translation pipeline

### ⚠️ Some Tests Failed
Check the specific failures:
- **llama_cpp**: Build issues
- **gguf_model**: Model file problems
- **translation_models**: Missing or corrupted models
- **speech_models**: Configuration issues
- **python_components**: Import or dependency problems

### ❌ Critical Failures
If core components fail:
1. Re-run the factory script: `python scripts/factory/working_enhanced_factory.py --step 5`
2. Check model generation logs
3. Verify configuration files
4. Ensure sufficient disk space

## 🎯 Advanced Testing Scenarios

### 1. Multi-language Translation
```python
# Test all language combinations
languages = ["en", "hi", "te"]
test_phrases = {
    "en": "Hello, how are you?",
    "hi": "नमस्ते, कैसे हो आप?",
    "te": "నమస్కారం, మీరు ఎలా ఉన్నారు?"
}

for source in languages:
    for target in languages:
        if source != target:
            result = await factory.translate(TranslationRequest(
                text=test_phrases[source],
                source_language=source,
                target_language=target
            ))
            print(f"{source} → {target}: {result.translated_text}")
```

### 2. Voice Quality Testing
```python
# Test different voice inputs
voice_samples = [
    "Hello world",
    "Translate this text",
    "What is the weather like?",
    "I am a computer programmer"
]

for sample in voice_samples:
    # Simulate voice input
    translation = await factory.translate(TranslationRequest(
        text=sample,
        source_language="en",
        target_language="hi"
    ))
    print(f"Voice: '{sample}' → Hindi: {translation.translated_text}")
```

### 3. Performance Benchmarking
```python
import time

# Test translation speed
start_time = time.time()
for i in range(10):
    result = await factory.translate(TranslationRequest(
        text=f"Test sentence number {i}",
        source_language="en",
        target_language="hi"
    ))
end_time = time.time()

avg_time = (end_time - start_time) / 10
print(f"Average translation time: {avg_time:.2f} seconds")
```

## 📈 Performance Expectations

### Translation Speed
- **Hindi model**: ~0.5-1.0 seconds per sentence
- **Telugu model**: ~1.0-2.0 seconds per sentence
- **NLLB models**: Slightly slower but more accurate

### GGUF Inference
- **Q4_K_M quantization**: ~2-5 seconds for 50 tokens
- **Memory usage**: ~4-6GB RAM
- **CPU threads**: 4 threads recommended

### Speech Recognition
- **Real-time**: ~100-200ms latency
- **Accuracy**: 90%+ with clear audio
- **Language support**: English, Hindi, Telugu

## 🔄 Continuous Testing

### Automated Testing
```bash
# Run tests daily
python tests/voice_translation_test.py > test_logs/$(date +%Y%m%d).log

# Monitor performance
python tests/performance_benchmark.py
```

### Health Checks
```bash
# Quick health check
python tests/quick_test.py

# Detailed health check
python tests/voice_translation_test.py
```

## 📚 Additional Resources

- **Factory Script**: `scripts/factory/working_enhanced_factory.py`
- **Configuration**: `config/translation_config.json`
- **Model Directory**: `models/production/`
- **Test Results**: `tests/*_test_results.json`

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in test output
2. **Verify prerequisites**: Ensure llama.cpp is built and models exist
3. **Run quick test**: Use `python tests/quick_test.py` for basic validation
4. **Check configuration**: Verify `translation_config.json` and `trinity_config.yaml`
5. **Review recent changes**: Check what was modified in the last factory run

---

**🎉 Happy Testing! Your MeeTARA Lab is ready to revolutionize voice and translation capabilities!**
