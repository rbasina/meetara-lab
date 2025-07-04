# Temporary Directory

Temporary file management for MeeTARA Lab Trinity Architecture operations.

## Directory Structure

```
temp/
├── processing/      # Temporary processing files
├── downloads/       # Temporary download files
└── README.md        # This file
```

## Temporary File Categories

### Processing Files (`processing/`)
- **Model Processing**: Temporary files during model training/conversion
- **Data Processing**: Temporary files during data preprocessing
- **Pipeline Processing**: Temporary files during pipeline operations
- **Batch Processing**: Temporary files during batch operations

### Download Files (`downloads/`)
- **Model Downloads**: Temporary model downloads
- **Dataset Downloads**: Temporary dataset downloads
- **Dependency Downloads**: Temporary dependency downloads
- **Update Downloads**: Temporary update downloads

## Temporary File Management

### Automatic Cleanup
- **Age-based Cleanup**: Remove files older than specified age
- **Size-based Cleanup**: Remove files when directory size exceeds limit
- **Pattern-based Cleanup**: Remove files matching specific patterns
- **Scheduled Cleanup**: Regular cleanup schedule

### Cleanup Configuration
```yaml
temp_cleanup:
  processing:
    max_age: "24h"
    max_size: "10GB"
    cleanup_schedule: "0 2 * * *"  # Daily at 2 AM
    
  downloads:
    max_age: "7d"
    max_size: "50GB"
    cleanup_schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
```

### Manual Cleanup
```bash
# Clean all temporary files
python temp/tools/cleanup.py --all

# Clean processing files only
python temp/tools/cleanup.py --processing

# Clean downloads only
python temp/tools/cleanup.py --downloads

# Clean files older than 1 day
python temp/tools/cleanup.py --older-than 1d
```

## File Naming Conventions

### Processing Files
- **Format**: `{operation}_{timestamp}_{uuid}.{ext}`
- **Example**: `model_training_20250104_143022_a1b2c3d4.tmp`
- **Components**:
  - `operation`: Type of operation (model_training, data_processing, etc.)
  - `timestamp`: ISO timestamp (YYYYMMDD_HHMMSS)
  - `uuid`: Unique identifier
  - `ext`: File extension

### Download Files
- **Format**: `{source}_{filename}_{timestamp}.{ext}`
- **Example**: `huggingface_model_20250104_143022.tar.gz`
- **Components**:
  - `source`: Download source (huggingface, github, etc.)
  - `filename`: Original filename
  - `timestamp`: Download timestamp
  - `ext`: File extension

## Temporary File Operations

### Safe File Creation
```python
import tempfile
import os
from pathlib import Path

# Create temporary processing file
def create_temp_processing_file(operation: str, extension: str = "tmp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_str = str(uuid.uuid4())[:8]
    filename = f"{operation}_{timestamp}_{uuid_str}.{extension}"
    return Path("temp/processing") / filename

# Create temporary download file
def create_temp_download_file(source: str, filename: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{source}_{filename}_{timestamp}"
    return Path("temp/downloads") / safe_filename
```

### File Locking
```python
import fcntl
import contextlib

@contextlib.contextmanager
def locked_temp_file(filepath):
    """Context manager for locked temporary files"""
    with open(filepath, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

## Monitoring and Alerts

### Disk Usage Monitoring
- **Space Monitoring**: Monitor temporary directory disk usage
- **Growth Rate**: Track temporary file growth rate
- **Cleanup Effectiveness**: Monitor cleanup operation effectiveness
- **Storage Alerts**: Alert when temporary storage exceeds thresholds

### File Lifecycle Tracking
- **Creation Tracking**: Track temporary file creation
- **Usage Tracking**: Track temporary file access patterns
- **Cleanup Tracking**: Track temporary file cleanup
- **Retention Analysis**: Analyze temporary file retention patterns

### Performance Monitoring
- **I/O Performance**: Monitor temporary file I/O performance
- **Cleanup Performance**: Monitor cleanup operation performance
- **Storage Performance**: Monitor temporary storage performance
- **Access Patterns**: Analyze temporary file access patterns

## Security Considerations

### Access Control
- **File Permissions**: Appropriate file permissions for temporary files
- **Directory Permissions**: Secure directory permissions
- **User Isolation**: Isolate temporary files by user
- **Process Isolation**: Isolate temporary files by process

### Data Protection
- **Sensitive Data**: Handle sensitive data in temporary files
- **Encryption**: Encrypt sensitive temporary files
- **Secure Deletion**: Securely delete temporary files
- **Data Leakage Prevention**: Prevent data leakage through temporary files

### Audit and Compliance
- **Access Logging**: Log temporary file access
- **Cleanup Logging**: Log temporary file cleanup operations
- **Retention Compliance**: Ensure compliance with data retention policies
- **Audit Trail**: Maintain audit trail for temporary file operations

## Integration with System Components

### Training Pipeline Integration
- **Model Checkpoints**: Temporary storage for training checkpoints
- **Data Preprocessing**: Temporary storage for preprocessing operations
- **Model Conversion**: Temporary storage for model conversion
- **Validation Files**: Temporary storage for validation operations

### GGUF Factory Integration
- **Model Processing**: Temporary storage for GGUF creation
- **Compression**: Temporary storage for model compression
- **Validation**: Temporary storage for model validation
- **Packaging**: Temporary storage for model packaging

### Cloud Training Integration
- **Download Cache**: Temporary storage for cloud downloads
- **Upload Staging**: Temporary storage for cloud uploads
- **Sync Operations**: Temporary storage for synchronization
- **Backup Operations**: Temporary storage for backup operations

## Best Practices

### File Management
- **Unique Names**: Use unique names for temporary files
- **Atomic Operations**: Use atomic operations for temporary file creation
- **Error Handling**: Proper error handling for temporary file operations
- **Resource Cleanup**: Always clean up temporary files

### Performance Optimization
- **Fast Storage**: Use fast storage for temporary files
- **Parallel Operations**: Parallelize temporary file operations when possible
- **Batch Operations**: Batch temporary file operations for efficiency
- **Memory Mapping**: Use memory mapping for large temporary files

### Reliability
- **Crash Recovery**: Handle temporary files after system crashes
- **Concurrent Access**: Handle concurrent access to temporary files
- **Disk Full Handling**: Handle disk full conditions gracefully
- **Backup Considerations**: Consider backup implications of temporary files

## Troubleshooting

### Common Issues
- **Disk Full**: Temporary directory fills up disk space
- **Permission Denied**: Insufficient permissions for temporary files
- **File Locked**: Temporary files locked by other processes
- **Cleanup Failures**: Automatic cleanup operations fail

### Diagnostic Tools
```bash
# Check temporary directory usage
python temp/tools/usage_report.py

# Find large temporary files
python temp/tools/find_large_files.py --size 1GB

# Check temporary file permissions
python temp/tools/check_permissions.py

# Analyze temporary file patterns
python temp/tools/analyze_patterns.py --period 7d
```

### Recovery Procedures
- **Manual Cleanup**: Manual cleanup procedures for emergency situations
- **Permission Repair**: Repair file permissions for temporary files
- **Disk Space Recovery**: Recover disk space from temporary files
- **Corrupted File Recovery**: Handle corrupted temporary files 