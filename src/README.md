# MeeTARA Lab - Modern Source Structure

This directory provides clean, modern imports from existing scattered directories without breaking changes.

## Usage

```python
# Instead of complex imports from scattered directories:
from model_factory.trinity_master_gguf_factory import create_gguf_model
from trinity_core.intelligent_router import route_request

# Use clean, organized imports:
from src.models import create_gguf_model
from src.intelligence import route_request
```

## Structure

- `models/` - Model training and GGUF creation (wraps model-factory/)
- `intelligence/` - AI routing and processing (wraps trinity-core/, intelligence-hub/)
- `data/` - Data generation and validation (wraps model-factory/ data functions)
- `infrastructure/` - Cloud and system management (wraps cloud-training/, cost-optimization/)
- `utils/` - Shared utilities and configuration
- `main.py` - Single entry point for common operations 