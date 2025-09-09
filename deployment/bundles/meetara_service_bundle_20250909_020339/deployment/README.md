# MeeTARA Service Bundle Integration Guide

## 📦 Bundle: meetara_service_bundle_20250909_020339

This bundle contains all necessary components for MeeTARA frontend integration:

### 🤖 Models Included:
- **Mobile Universal Model**: Optimized for mobile devices (4B parameters)
- **Desktop Universal Model**: Full-featured for desktop applications (8B parameters)
- **62+ Emotional Intelligence Domains**: Complete domain coverage

### 🎤 Services Included:
- **Emotion Detection**: Real-time emotion analysis
- **Voice Synthesis**: Category-specific voice profiles
- **Translation**: Multi-language support (Hindi, Telugu)
- **Intelligent Routing**: Domain and emotion-based routing

### 🚀 Quick Installation:

1. **Automatic Installation**:
```bash
python deployment/install.py
```

2. **Manual Installation**:
```bash
# Copy to MeeTARA frontend repository
cp -r models/ /path/to/meetara/models/
cp -r services/ /path/to/meetara/services/
cp -r config/ /path/to/meetara/config/services/
```

### 🔧 Configuration:
- Service config: `config/service_config.json`
- Model mapping: `config/model_mapping.json`
- API endpoints: `config/api_endpoints.json`

### 📱 Frontend Integration:
1. Import service configurations
2. Initialize model loaders
3. Set up API endpoints
4. Configure routing logic

### 🎯 Usage in MeeTARA Frontend:
```javascript
// Initialize MeeTARA services
import { MeeTARAServices } from './services/meetara-services';

const meetara = new MeeTARAServices({
    modelPath: './models/meetara_mobile_universal.gguf',
    servicesPath: './services/',
    configPath: './config/services/'
});

// Use emotional intelligence
const response = await meetara.chat("I'm feeling stressed about work");
const emotion = await meetara.detectEmotion(response);
const voice = await meetara.synthesizeVoice(response, 'healthcare');
```

Generated: 2025-09-09 02:04:06
