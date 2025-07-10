# Getting Started
*Quick Setup and Installation Guide for MeeTARA Lab*

## 🚀 Quick Start

Get MeeTARA Lab running in **5 minutes** with Trinity Architecture and cloud GPU acceleration.

### Prerequisites Checklist
- [ ] Python 3.12.x (NOT 3.13) installed
- [ ] Windows PowerShell (for Windows users)
- [ ] Git for version control
- [ ] Stable internet connection
- [ ] Google Colab Pro+ account (for GPU training)

## 📥 Installation

### 1. Clone Repository
```powershell
git clone <repository-url> meetara-lab
cd meetara-lab
```

### 2. Setup Python Environment
```powershell
# Create virtual environment
python -m venv .venv-tara-py312

# Activate virtual environment
.venv-tara-py312\Scripts\Activate.ps1

# Install dependencies (54 packages)
pip install -r requirements.txt
```

### 3. Verify Installation
```powershell
# Check core dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import speechbrain; print(f'SpeechBrain: {speechbrain.__version__}')"
```

Expected output:
```
PyTorch: 2.7.1
Transformers: 4.45.2
SpeechBrain: 1.0.2
```

## ⚡ Quick Configuration

### Environment Variables
Create `.env` file in project root:
```bash
# Cloud Provider API Keys (Optional for local testing)
OPENAI_API_KEY=your_key_here
COLAB_AUTH_TOKEN=your_token_here
LAMBDA_LABS_API_KEY=your_key_here

# Local Development
DEBUG_MODE=true
LOG_LEVEL=INFO
```

### Port Configuration
Default ports used by MeeTARA Lab:
```yaml
Frontend: 2025          # MeeTARA frontend
WebSocket: 8765         # Real-time communication
Session API: 8766       # HTTP REST API
TARA Universal: 5000    # Voice service
```

## 🧪 Test Your Setup

### 1. Basic Component Test
```python
# Test TTS Manager
from trinity_core.tts_manager import TTSManager

tts = TTSManager()
result = tts.synthesize_voice("Hello from MeeTARA Lab!", voice_category="friendly")
print(f"Voice synthesis: {result.status}")
```

### 2. Trinity Architecture Test
```python
# Test Trinity Intelligence
from intelligence_hub.trinity_intelligence import TrinityIntelligence

trinity = TrinityIntelligence()
enhanced = trinity.process_with_trinity("Test input")
print(f"Trinity enhancement: {enhanced}")
```

### 3. Cloud Connection Test
```python
# Test Cloud Orchestrator
from cloud_training.gpu_orchestrator import GPUOrchestrator

orchestrator = GPUOrchestrator()
providers = orchestrator.get_available_providers()
print(f"Available cloud providers: {len(providers)}")
```

## 🎯 First Training Run

### Test Domain Training
```python
# Quick GGUF model creation test
from model_factory.gguf_factory import GGUFFactory

factory = GGUFFactory()
test_result = factory.create_test_model(
    domain="test_domain",
    tier="lightning"  # $2-3 cost tier
)
print(f"Test model created: {test_result.model_path}")
print(f"Size: {test_result.size_mb}MB")
print(f"Quality score: {test_result.validation_score}%")
```

Expected results:
- **Model Size**: ~8.3MB
- **Quality Score**: 101%
- **Creation Time**: 3-15 minutes (depending on GPU tier)

## 📖 Next Steps

### For Developers
1. **[Component Development](../development/component-guide.md)** - Build custom TARA components
2. **[API Reference](../api/README.md)** - Explore available APIs
3. **[Testing Guide](../development/testing.md)** - Write and run tests

### For System Administrators
1. **[Production Deployment](../deployment/production-setup.md)** - Deploy to production
2. **[Monitoring Setup](../monitoring/dashboard-setup.md)** - Set up monitoring
3. **[Configuration Guide](../configuration/system-config.md)** - Advanced configuration

### For Researchers
1. **[Performance Analysis](../performance/gpu-acceleration.md)** - Benchmark and optimize
2. **[Research Documentation](../research/training-optimization.md)** - Research findings
3. **[Advanced Features](../tutorials/advanced-features.md)** - Explore advanced capabilities

## 🔧 Common Setup Issues

### Python 3.13 Compatibility
**Issue**: OpenCV errors with Python 3.13
```bash
Error: OpenCV compatibility issues
```
**Solution**: Use Python 3.12.x only
```powershell
python --version  # Should show 3.12.x
```

### PowerShell Command Issues
**Issue**: `&&` operator not recognized
```bash
cd directory && python script.py  # ❌ Doesn't work
```
**Solution**: Use semicolon separator
```powershell
cd directory; python script.py  # ✅ Works
```

### Virtual Environment Issues
**Issue**: Virtual environment not activating
```powershell
# If activation fails, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv-tara-py312\Scripts\Activate.ps1
```

### Memory Issues
**Issue**: GPU out of memory during training
**Solution**: Uses proven batch_size=6 automatically
```python
# Automatic optimization included
factory = GGUFFactory()
# batch_size=6 used automatically for 8.3MB models
```

## 📊 System Status Check

Run this command to verify all components:
```python
from trinity_core.system_checker import SystemChecker

checker = SystemChecker()
status = checker.check_all_components()
print(status.summary())
```

Expected output:
```
✅ Trinity Core: Operational
✅ TTS Manager: Ready
✅ Emotion Detector: Ready  
✅ Intelligent Router: Ready
✅ GGUF Factory: Ready
✅ Training Orchestrator: Ready
✅ Cloud Orchestrator: Connected
✅ Cost Monitor: Active
🎯 System Status: Ready for Training
```

## 🎉 Success Indicators

You're ready to use MeeTARA Lab when:
- [x] All dependencies installed without errors
- [x] System status check passes
- [x] Test model creation succeeds
- [x] Trinity enhancement working
- [x] Cloud providers accessible

**Congratulations!** You now have MeeTARA Lab running with Trinity Architecture and 20-100x GPU acceleration capability.

## 📞 Support

- **Documentation**: [Full Documentation](../README.md)
- **Troubleshooting**: [Common Issues](../troubleshooting/common-issues.md)
- **Architecture**: [System Design](../architecture/README.md)
- **Memory Bank**: [Project Context](../../memory-bank/README.md)

---

*Ready to transform AI training with Trinity Architecture? Start with your [first component tutorial](../tutorials/first-component.md)!* 