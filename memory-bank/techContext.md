# MeeTARA Lab - Technical Context
*Technologies, Development Setup, and Technical Constraints*

## Technology Stack

### Core Languages & Frameworks
```yaml
backend:
  primary_language: "Python 3.12"
  ml_frameworks:
    - "PyTorch 2.7.1"
    - "Transformers 4.45.2"
    - "SpeechBrain 1.0.2"
    - "OpenCV 4.10.0"
  web_frameworks:
    - "FastAPI 0.115.4"
    - "WebSockets 13.1"
  
frontend:
  primary_language: "TypeScript"
  framework: "Next.js"
  runtime: "Node.js"
  
cloud:
  primary: "Google Colab Pro+"
  secondary: ["Lambda Labs", "RunPod", "Vast.ai"]
  gpu_types: ["T4", "V100", "A100"]
```

### AI/ML Dependencies
```python
# Core ML Stack
torch==2.7.1
transformers==4.45.2
speechbrain==1.0.2
torch-audio==2.5.1

# NLP & Emotion Analysis
j-hartmann/emotion-english-distilroberta-base  # RoBERTa emotion model
sentence-transformers==3.3.1

# Voice Processing
edge-tts==6.1.18
pyttsx3==2.98
faster-whisper==1.1.0

# Computer Vision
opencv-python==4.10.0.84
numpy==1.26.4  # Downgraded for OpenCV compatibility

# GGUF & Model Management
llama-cpp-python==0.3.2
gguf==0.10.0

# Cloud & API
google-colab-auth==0.0.5
google-cloud-storage==2.18.0
requests==2.32.3
websockets==13.1
```

### Development Environment

#### Virtual Environment Setup
```bash
# Python 3.12 Virtual Environment
location: .venv-tara-py312/
activation: .venv-tara-py312\Scripts\Activate.ps1  # Windows PowerShell
python_version: 3.12.0
compatibility: Python 3.12.x (NOT 3.13 - OpenCV issues)
```

#### Port Configuration
```yaml
services:
  frontend: 2025          # MeeTARA frontend (HAI collaboration port)
  websocket: 8765         # WebSocket communication
  session_api: 8766       # HTTP REST API
  tara_universal: 5000    # TARA Universal Model voice service
  
development:
  hot_reload: true        # Frontend auto-refresh
  debug_mode: true        # Enhanced logging
  cors_enabled: true      # Cross-origin support
```

#### Directory Structure
```
meetara-lab/
├── .venv-tara-py312/           # Python virtual environment
├── memory-bank/                # Project memory system
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── activeContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   └── progress.md
├── docs/                       # Documentation
├── tests/                      # Testing framework
├── trinity-core/               # Core components
├── model-factory/              # GGUF creation
├── cloud-training/             # GPU orchestration
├── intelligence-hub/           # Advanced AI
├── cost-optimization/          # Budget management
├── deployment-engine/          # Production deployment
├── notebooks/                  # Google Colab integration
└── research-workspace/         # Advanced research
```

## Development Setup

### Prerequisites
```bash
# Required Software
- Python 3.12.x (NOT 3.13)
- Node.js 18+ with npm
- Git for version control
- Windows PowerShell (for Windows users)
- Google Colab Pro+ account (for GPU training)

# Optional
- Docker (for containerized deployment)
- CUDA 12.1+ (for local GPU development)
```

### Installation Process
```powershell
# 1. Clone Repository
git clone <repository-url> meetara-lab
cd meetara-lab

# 2. Setup Python Environment
python -m venv .venv-tara-py312
.venv-tara-py312\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Setup Frontend (if needed)
npm install
npm run dev

# 4. Verify Installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Configuration Files
```yaml
# requirements.txt - Python dependencies
torch==2.7.1
transformers==4.45.2
speechbrain==1.0.2
# ... (54 total dependencies)

# package.json - Node.js dependencies (if applicable)
{
  "dependencies": {
    "next": "^14.0.0",
    "typescript": "^5.0.0"
  }
}

# .env - Environment variables
OPENAI_API_KEY=your_key_here
COLAB_AUTH_TOKEN=your_token_here
LAMBDA_LABS_API_KEY=your_key_here
```

## Technical Constraints

### Hardware Requirements
```yaml
minimum:
  ram: "8GB"
  storage: "20GB available"
  internet: "Stable broadband connection"
  
recommended:
  ram: "16GB+"
  storage: "50GB+ SSD"
  gpu: "Optional for local development"
  internet: "High-speed for cloud training"
  
cloud_requirements:
  google_colab_pro: "T4/V100/A100 GPU access"
  monthly_budget: "<$50 for all domains"
```

### Platform Compatibility
```yaml
operating_systems:
  primary: "Windows 10/11"
  supported: ["macOS", "Linux Ubuntu 20.04+"]
  
python_versions:
  required: "3.12.x"
  not_supported: ["3.13.x"]  # OpenCV compatibility issues
  
gpu_compatibility:
  local: ["CUDA 12.1+", "ROCm 5.0+"]
  cloud: ["T4", "V100", "A100"]
```

### Performance Constraints
```yaml
training_limits:
  max_batch_size: 6          # Proven optimal for 8.3MB models
  max_steps: 846             # Validated for quality
  memory_usage: "12MB runtime"  # Target for efficiency
  loading_time: "50ms"       # Target speed
  
cost_constraints:
  monthly_budget: "$50"      # All 60+ domains
  per_domain_max: "$15"      # Quality tier maximum
  emergency_shutdown: "95% budget"  # Automatic protection
```

## Integration Requirements

### MeeTARA Ecosystem
```yaml
frontend_integration:
  port_compatibility: [2025, 8765, 8766]
  websocket_support: true
  rest_api_support: true
  gguf_model_serving: true
  
backend_services:
  tara_universal_model: "Port 5000"
  meetara_backend: "Port 8765/8766"
  voice_synthesis: "Edge-TTS integration"
  emotion_detection: "Multi-modal support"
```

### TARA Universal Model Compatibility
```yaml
parameter_preservation:
  batch_size: 6
  lora_r: 8
  max_steps: 846
  learning_rate: 3e-4
  quantization: "Q4_K_M"
  
output_requirements:
  model_size: "8.3MB"
  format: "GGUF"
  quality_score: "101%"
  loading_speed: "50ms"
  memory_usage: "12MB"
```

### Cloud Provider APIs
```yaml
google_colab:
  authentication: "OAuth 2.0"
  gpu_access: ["T4", "V100", "A100"]
  storage: "Google Drive integration"
  
lambda_labs:
  api_version: "v1"
  authentication: "API Key"
  gpu_types: ["RTX 4090", "A100"]
  
runpod:
  api_version: "v2"
  authentication: "Bearer Token"
  serverless: true
  
vast_ai:
  api_version: "v0"
  spot_instances: true
  cost_optimization: true
```

## Security & Privacy

### Data Protection
```yaml
privacy_requirements:
  local_processing: "No sensitive data to cloud"
  encryption: "All data encrypted in transit and at rest"
  compliance: ["GDPR", "HIPAA ready"]
  
access_control:
  authentication: "Secure model serving"
  authorization: "Role-based access"
  audit_logging: "All operations tracked"
```

### API Security
```python
# Environment Variables (Never hardcode)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LAMBDA_LABS_API_KEY = os.getenv('LAMBDA_LABS_API_KEY')

# Secure Configuration
class SecurityConfig:
    def __init__(self):
        self.api_keys = self.load_secure_keys()
        self.encryption_key = self.generate_encryption_key()
    
    def load_secure_keys(self):
        # Load from secure environment or key vault
        pass
```

## Deployment Architecture

### Development Environment
```yaml
local_development:
  frontend: "http://localhost:2025"
  backend_websocket: "ws://localhost:8765"
  backend_api: "http://localhost:8766"
  tara_voice: "http://localhost:5000"
  
development_tools:
  hot_reload: true
  debug_logging: true
  performance_monitoring: true
```

### Production Environment
```yaml
production_deployment:
  containerization: "Docker"
  orchestration: "Docker Compose"
  monitoring: "Real-time dashboards"
  backup: "Automated model backup"
  
scalability:
  horizontal_scaling: "Multi-instance support"
  load_balancing: "Automatic traffic distribution"
  auto_scaling: "Based on demand"
```

### Model Deployment
```yaml
model_serving:
  format: "GGUF"
  size: "8.3MB"
  loading: "50ms"
  memory: "12MB"
  
distribution:
  local_cache: "./models/cache/"
  cloud_storage: "Google Cloud Storage"
  cdn: "Global distribution"
```

## Quality Assurance

### Testing Requirements
```yaml
testing_framework:
  unit_tests: "pytest"
  integration_tests: "Custom framework"
  performance_tests: "Speed and cost validation"
  quality_tests: "Model validation"
  
coverage_requirements:
  minimum_coverage: "80%"
  critical_components: "95%"
  performance_tests: "All components"
```

### Monitoring & Observability
```yaml
monitoring_stack:
  performance: "Real-time metrics"
  cost_tracking: "Budget monitoring"
  error_logging: "Comprehensive error capture"
  health_checks: "Service availability"
  
alerting:
  cost_alerts: [50%, 80%, 90%, 95%]
  performance_degradation: "Automatic detection"
  service_failures: "Immediate notification"
```

## Known Technical Limitations

### Current Constraints
1. **Python 3.13 Incompatibility**: OpenCV issues require Python 3.12
2. **Windows PowerShell**: Semicolon (;) required instead of && for command chaining
3. **GPU Memory**: Batch size limited to 6 for optimal 8.3MB models
4. **Cloud Dependencies**: Requires stable internet for training

### Workarounds Implemented
1. **Python Version**: Locked to 3.12.x with numpy==1.26.4
2. **PowerShell Commands**: Use semicolon separators in batch scripts
3. **Memory Management**: Automatic batch size optimization
4. **Offline Capability**: Local GGUF model serving for privacy

This technical foundation ensures robust, scalable, and maintainable development while preserving compatibility with existing TARA Universal Model success patterns. 