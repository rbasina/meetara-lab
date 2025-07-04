# Logs Directory

Centralized logging system for MeeTARA Lab Trinity Architecture.

## Directory Structure

```
logs/
├── training/          # Training logs for all domains
├── validation/        # Validation and testing logs
├── system/           # System-level logs
└── README.md         # This file
```

## Log Categories

### Training Logs (`training/`)
- **Domain Training**: Individual domain training progress
- **GPU Usage**: GPU utilization and performance metrics
- **Error Tracking**: Training failures and recovery logs
- **Performance Metrics**: Speed, loss, validation scores

### Validation Logs (`validation/`)
- **Model Validation**: GGUF model validation results
- **Quality Assurance**: Testing and QA logs
- **Production Readiness**: Deployment validation logs
- **Integration Testing**: Component integration logs

### System Logs (`system/`)
- **Application Logs**: Core system operation logs
- **Error Logs**: System errors and exceptions
- **Performance Logs**: System performance metrics
- **Security Logs**: Access and security events

## Log Rotation

- **Daily Rotation**: Logs rotated daily at midnight
- **Size Limit**: 100MB per log file
- **Retention**: 30 days for training logs, 90 days for system logs
- **Compression**: Older logs automatically compressed

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Potential issues that don't stop operation
- **ERROR**: Error conditions that may affect functionality
- **CRITICAL**: Serious errors that may cause system failure

## Usage Examples

```bash
# View recent training logs
tail -f logs/training/healthcare_training.log

# Search for errors in system logs
grep -r "ERROR" logs/system/

# Monitor validation results
tail -f logs/validation/model_validation.log
```

## Integration

All MeeTARA Lab components automatically log to appropriate directories:
- Trinity Core → `system/`
- Training Orchestrator → `training/`
- Model Factory → `validation/`
- Cloud Training → `training/` 