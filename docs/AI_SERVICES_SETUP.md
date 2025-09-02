# AI Services Setup Guide for MeeTARA Lab

This guide explains how to configure AI services (OpenAI, Gemini, DeepSeek) for enhanced training data generation in MeeTARA Lab.

## Overview

MeeTARA Lab now supports AI-powered training data generation using:
- **OpenAI GPT-4o-mini** - For realistic conversation scenarios
- **Google Gemini 1.5 Flash** - For emotionally intelligent responses
- **DeepSeek Chat** - For domain-specific expertise

## Setup Options

### Option 1: .env File (Recommended for Development)

1. Copy the environment template:
   ```bash
   cp env.template .env
   ```

2. Edit the `.env` file and add your actual API keys:
   ```bash
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   GEMINI_API_KEY=AIzaSy-your-actual-key-here
   DEEPSEEK_API_KEY=sk-your-actual-key-here
   ```

3. **IMPORTANT**: Never commit the `.env` file to version control!

### Option 2: System Environment Variables (Recommended for Production)

Set these environment variables in your system:

```bash
# For Windows PowerShell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:DEEPSEEK_API_KEY="your_deepseek_api_key_here"

# For Linux/Mac
export OPENAI_API_KEY="your_openai_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### Option 3: Configuration File (Service Settings Only)

The `config/ai_services_config.yaml` file contains only service configuration (models, tokens, temperature) - no API keys.

## API Key Sources

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-proj-`)

### Google Gemini
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `AIzaSy`)

### DeepSeek
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

## Configuration Options

### AI Service Models
```yaml
ai_services:
  openai_model: "gpt-4o-mini"      # Cost-effective, high-quality
  gemini_model: "gemini-1.5-flash"  # Fast, efficient
  deepseek_model: "deepseek-chat"   # Specialized reasoning
```

### Generation Parameters
```yaml
ai_services:
  openai_max_tokens: 2000           # Maximum response length
  openai_temperature: 0.7           # Creativity level (0.0-1.0)
  gemini_max_tokens: 2000
  gemini_temperature: 0.7
  deepseek_max_tokens: 2000
  deepseek_temperature: 0.7
```

### AI Configuration
```yaml
ai_config:
  cache_duration: 3600              # Cache duration in seconds
  max_scenarios_per_domain: 100     # Max AI scenarios per domain
  enable_ai_enhancement: true       # Enable/disable AI features
  fallback_to_templates: true       # Use templates if AI fails
  quality_threshold: 0.8            # Minimum quality score
```

## Usage

Once configured, the AI services will automatically:

1. **Initialize** when the `TrinityDataGenerator` starts
2. **Generate** enhanced training scenarios using AI
3. **Fallback** to template-based generation if AI services fail
4. **Cache** responses to minimize API calls
5. **Enhance** existing conversations with AI variations

## Testing

### Prerequisites

1. Install python-dotenv (optional but recommended):
   ```bash
   pip install python-dotenv
   ```

2. Set up your `.env` file with API keys (see Setup Options above)

### Test AI Services

```python
from trinity_core.agents.data_generator import TrinityDataGenerator

# Initialize the generator
generator = TrinityDataGenerator(hub, environment='dev')

# Check if AI services are enabled
if generator.ai_config.get("enable_ai_enhancement"):
    print("✅ AI services are enabled")
    print(f"Available services: {list(generator.ai_services.keys())}")
    
    # Test AI scenario generation
    try:
        scenario = generator._generate_ai_enhanced_scenario("healthcare", "crisis", "medical emergency")
        print(f"✅ AI scenario generated: {scenario['conversation_starter'][:100]}...")
    except Exception as e:
        print(f"⚠️ AI generation failed: {e}")
else:
    print("⚠️ AI services are disabled - check your configuration")
    print("Make sure your .env file contains the required API keys")
```

## Troubleshooting

### Common Issues

1. **"No AI services configured"**
   - Check that API keys are set correctly
   - Verify environment variables are loaded
   - Ensure configuration file is readable

2. **"API key invalid"**
   - Verify your API key is correct
   - Check if the service is active
   - Ensure you have sufficient credits

3. **"Rate limit exceeded"**
   - Reduce `max_scenarios_per_domain`
   - Increase `cache_duration`
   - Check your service plan limits

### Debug Mode

Enable debug logging to see detailed API interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## .env File Structure

Your `.env` file should look like this:

```bash
# AI Services API Keys
OPENAI_API_KEY=sk-proj-your-actual-openai-key-here
GEMINI_API_KEY=AIzaSy-your-actual-gemini-key-here
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key-here

# Other Configuration
ENVIRONMENT=dev
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

**Important**: 
- The `.env` file is automatically loaded by python-dotenv
- Never commit this file to version control
- Add `.env` to your `.gitignore` file

## Security Notes

- **Never hardcode API keys** in your scripts
- **Use .env files** for development (with .gitignore protection)
- **Use environment variables** for production deployments
- **Keep API keys separate** from configuration files
- **Rotate API keys** regularly
- **Monitor API usage** to control costs

## Cost Optimization

- Use `gpt-4o-mini` instead of `gpt-4` for cost savings
- Set reasonable `max_tokens` limits
- Enable caching to reduce API calls
- Use `fallback_to_templates` for non-critical scenarios

## Support

For issues with:
- **OpenAI**: [OpenAI Help Center](https://help.openai.com/)
- **Gemini**: [Google AI Support](https://ai.google.dev/support)
- **DeepSeek**: [DeepSeek Support](https://platform.deepseek.com/support)

For MeeTARA Lab integration issues, check the project documentation or create an issue in the repository.
