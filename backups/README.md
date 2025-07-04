# Backups Directory

Automated backup system for MeeTARA Lab critical assets.

## Directory Structure

```
backups/
├── models/           # Model backups (GGUF files, checkpoints)
├── data/            # Training data backups
├── configs/         # Configuration backups
└── README.md        # This file
```

## Backup Categories

### Model Backups (`models/`)
- **GGUF Files**: Production-ready GGUF models
- **Checkpoints**: Training checkpoints for recovery
- **Adapters**: LoRA adapters and fine-tuned models
- **Metadata**: Model metadata and validation results

### Data Backups (`data/`)
- **Training Data**: Domain-specific training datasets
- **Validation Data**: Test and validation datasets
- **Synthetic Data**: Generated training data
- **Raw Data**: Original unprocessed data

### Configuration Backups (`configs/`)
- **Trinity Config**: Core system configurations
- **Training Config**: Training parameters and settings
- **Domain Config**: Domain-specific configurations
- **Deployment Config**: Production deployment settings

## Backup Strategy

### Automated Backups
- **Daily**: Critical model checkpoints
- **Weekly**: Complete model backups
- **Monthly**: Full system backup including data
- **On-Demand**: Before major updates or deployments

### Retention Policy
- **Models**: 30 days for checkpoints, 1 year for production models
- **Data**: 90 days for training data, permanent for validated datasets
- **Configs**: 1 year for all configurations

### Backup Validation
- **Integrity Checks**: SHA256 checksums for all backups
- **Restore Testing**: Monthly restore tests
- **Corruption Detection**: Automated corruption detection

## Recovery Procedures

### Model Recovery
```bash
# Restore latest model checkpoint
python scripts/utilities/restore_model.py --domain healthcare --latest

# Restore specific checkpoint
python scripts/utilities/restore_model.py --checkpoint backups/models/healthcare_checkpoint_846.pth
```

### Data Recovery
```bash
# Restore training data
python scripts/utilities/restore_data.py --domain healthcare --date 2025-01-01

# Restore all data
python scripts/utilities/restore_data.py --full-restore
```

### Configuration Recovery
```bash
# Restore system configuration
python scripts/utilities/restore_config.py --type system

# Restore domain configuration
python scripts/utilities/restore_config.py --domain healthcare
```

## Backup Monitoring

- **Storage Usage**: Monitor backup storage usage
- **Backup Success**: Track backup completion status
- **Alerts**: Automated alerts for backup failures
- **Reporting**: Weekly backup status reports

## Security

- **Encryption**: All backups encrypted at rest
- **Access Control**: Role-based access to backups
- **Audit Trail**: Complete audit trail for backup access
- **Compliance**: GDPR/HIPAA compliant backup procedures 