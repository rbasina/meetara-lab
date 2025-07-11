# 🤖 MeeTARA Lab - Model Comparison UI

Welcome to the MeeTARA Lab Model Comparison Interface! This lightweight UI allows you to compare responses from actual GGUF models in real-time.

## 📁 Current File Structure

```
ui/
├── meetara_real_model_comparison.py    # 🚀 Main backend (Flask + real GGUF loading)
├── templates/
│   └── real_model_comparison.html      # 🎨 Modern web interface
├── launch_real_comparison.py           # 🔧 Easy launcher script
├── trinity_routing_engine.py           # 🧠 Smart routing logic
└── README.md                           # 📚 This documentation
```

**Note**: The old comparison interface files (`meetara_comparison_backend.py` and `meetara_comparison_ui.html`) have been archived to `scripts/archive/` as they were replaced by the enhanced real model comparison system.

## 🚀 Quick Start

### Option 1: Easy Launch (Recommended)
```bash
cd ui
python launch_real_comparison.py
```

### Option 2: Manual Launch
```bash
cd ui
python meetara_real_model_comparison.py
```

Then open your browser to: **http://localhost:5001**

## 🎯 Features

### Real Model Loading
- **A_universal_full (3.5GB)**: Maximum intelligence with Qwen 2.5-14B + 62 domains
- **B_universal_lite (800MB)**: Universal speed with Phi-3.5-mini + 62 domains  
- **C_category_specific (8.3MB)**: Healthcare specialist for urgent responses

### Smart Routing Analysis
- Automatic query analysis and model recommendation
- Emergency detection for healthcare queries
- Complexity assessment for optimal model selection

### Real-time Comparison
- Side-by-side response comparison
- Performance metrics (response time, tokens, model size)
- Visual indicators for recommended models

## 🔧 Setup Requirements

### Essential Dependencies
```bash
pip install flask flask-cors
```

### Real Model Loading (Optional)
```bash
pip install llama-cpp-python
```

Without `llama-cpp-python`, the UI runs in simulation mode with realistic sample responses.

## 🤖 Model Files

The UI looks for GGUF models in these locations:
```
models/
├── A_universal_full/
│   └── meetara_a_universal_full.gguf
├── B_universal_lite/
│   └── meetara_b_universal_lite.gguf
└── C_category_specific/
    └── meetara_c_category_specific.gguf
```

If models are not found, the UI automatically switches to simulation mode.

## 🎮 Usage Guide

### 1. Enter Your Prompt
Type any question or request in the text area. Examples:
- "Explain quantum computing in simple terms"
- "I have a headache, what should I do?"
- "Write a business plan for a coffee shop"

### 2. Select Models
Choose which models to compare:
- ✅ **A_universal_full**: For complex reasoning and detailed analysis
- ✅ **B_universal_lite**: For fast, universal responses
- ✅ **C_category_specific**: For healthcare and urgent queries

### 3. Compare or Analyze
- **🚀 Compare Models**: Get responses from all selected models
- **🧠 Analyze Query**: Get smart routing recommendation only

### 4. View Results
- See responses side-by-side
- Check performance metrics
- Identify recommended model based on query analysis

## 🧠 Smart Routing Logic

The system automatically analyzes your query and recommends the optimal model:

| Query Type | Recommended Model | Reason |
|------------|-------------------|---------|
| Emergency/Healthcare | C_category_specific | Urgent specialist response |
| Complex Analysis | A_universal_full | Deep reasoning required |
| General Questions | B_universal_lite | Fast universal coverage |

### Keywords Detection
- **Emergency**: "emergency", "urgent", "critical", "pain", "help", "doctor"
- **Complex**: "analyze", "explain", "comprehensive", "detailed", "research"
- **Healthcare**: "health", "medical", "symptom", "treatment", "medicine"

## 📊 Performance Metrics

Each model response includes:
- **Response Time**: How long the model took to respond
- **Tokens Generated**: Number of words/tokens in the response
- **Model Type**: Real or simulated inference
- **Model Size**: Physical size of the GGUF file

## 🎭 Simulation Mode

When real models aren't available, the UI provides realistic simulations:
- **A_universal_full**: Comprehensive, detailed responses (0.8s response time)
- **B_universal_lite**: Quick, universal responses (0.2s response time)
- **C_category_specific**: Specialist healthcare responses (0.05s response time)

## 🔍 Troubleshooting

### Models Not Loading
1. Check if GGUF files exist in the `models/` directory
2. Verify file permissions
3. Ensure sufficient RAM for large models

### Real Model Loading Failed
1. Install llama-cpp-python: `pip install llama-cpp-python`
2. For Windows with CUDA: `pip install llama-cpp-python[cuda]`
3. Check model file integrity

### UI Not Accessible
1. Ensure Flask is installed: `pip install flask flask-cors`
2. Check if port 5001 is available
3. Try running with `python -m flask run --port 5002`

## 🌟 Advanced Features

### Keyboard Shortcuts
- **Ctrl+Enter**: Compare models from prompt input
- **Escape**: Clear results

### URL Parameters
- `http://localhost:5001/?prompt=your+question`: Pre-fill prompt
- `http://localhost:5001/?models=A,B`: Pre-select models

### API Endpoints
- `GET /api/models`: Get available models
- `POST /api/compare`: Compare model responses
- `POST /api/analyze_query`: Analyze query for routing
- `GET /api/status`: Get system status

## 🔧 Configuration

### Model Loading Settings
Edit `meetara_real_model_comparison.py` to adjust:
- `n_ctx`: Context window size
- `n_threads`: CPU threads for inference
- `temperature`: Response creativity (0.0-1.0)

### UI Customization
Edit `templates/real_model_comparison.html` to customize:
- Color scheme
- Layout
- Additional metrics

## 📈 Performance Tips

### For Best Real Model Performance
1. Use SSD storage for model files
2. Allocate sufficient RAM (4GB+ for A_universal_full)
3. Use GPU acceleration if available
4. Close other applications to free resources

### For Development
1. Use simulation mode for UI testing
2. Test with smaller models first
3. Monitor system resources during inference

## 🎯 Integration with MeeTARA Lab

This UI is part of the larger MeeTARA Lab ecosystem:
- **Backend**: Creates and optimizes GGUF models
- **Frontend**: Provides human interaction interface
- **Comparison UI**: Tests and validates model performance

The comparison results help validate that the Trinity Architecture is working correctly and that models are performing as expected.

## 🤝 Contributing

To add new features:
1. Backend logic: Edit `meetara_real_model_comparison.py`
2. Frontend UI: Edit `templates/real_model_comparison.html`
3. Launcher: Edit `launch_real_comparison.py`

## 📚 Related Documentation

- [Trinity Architecture Overview](../docs/architecture/README.md)
- [Model Factory Guide](../model-factory/README.md)
- [GGUF Creation Process](../docs/performance/LIGHTWEIGHT_GGUF_GUIDE.md)

## 🗂️ Archive Note

The original comparison interface files have been moved to `scripts/archive/` for reference:
- `meetara_comparison_backend.py` → `scripts/archive/meetara_comparison_backend.py`
- `meetara_comparison_ui.html` → `scripts/archive/meetara_comparison_ui.html`

These were replaced by the enhanced real model comparison system that supports actual GGUF model loading.

---

**🚀 Ready to compare your models? Run `python launch_real_comparison.py` and start testing!** 