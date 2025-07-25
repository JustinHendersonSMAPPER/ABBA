# Configuration Guide

This document explains the ABBA configuration system, including all available options, their precedence, and usage examples.

## Configuration Hierarchy

ABBA uses a hierarchical configuration system with the following precedence (highest to lowest):

1. **Command Line Arguments** - Override everything
2. **Environment Variables** - Persistent user preferences
3. **Configuration File** - Project-specific settings
4. **Default Values** - Built-in fallbacks

This design allows maximum flexibility while maintaining sensible defaults.

## Command Line Arguments

### Basic Options

```bash
# Show help
python abba/main.py --help

# List available translations
python abba/main.py --list

# Show version
python abba/main.py --version
```

### Data Management

```bash
# Custom data directory
python abba/main.py --data-dir /path/to/data

# Force download of bible.db even if it exists
python abba/main.py --force-download

# Skip download, only process existing data
python abba/main.py --no-download

# Rebuild database from scratch
python abba/main.py --rebuild-db
```

### Translation Selection

```bash
# Import specific translations (space or comma separated)
python abba/main.py --translations KJV ASV ESV

# Import multiple translations
python abba/main.py --translations eng_kjv,eng_asv,eng_bbe,heb_wlc
```

### Output Control

```bash
# Verbose output (detailed progress)
python abba/main.py --verbose

# Quiet mode (minimal output)
python abba/main.py --quiet
```

### Configuration File

```bash
# Use custom configuration file
python abba/main.py --config-file my_config.json

# Combine with other options (CLI takes precedence)
python abba/main.py --config-file prod.json --verbose
```

## Environment Variables

All environment variables are prefixed with `ABBA_` to avoid conflicts.

### Core Settings

```bash
# Data directory location (default: ./bible_data)
ABBA_DATA_DIR=/path/to/bible/data

# Bible database URL (default: https://bible.helloao.org/bible.db)
ABBA_BIBLE_DB_URL=https://mirror.example.com/bible.db

# Specific translations to import (comma-separated)
ABBA_TRANSLATIONS=KJV,ASV,ESV,eng_bbe

# Force download even if files exist
ABBA_FORCE_DOWNLOAD=true

# Skip download phase
ABBA_NO_DOWNLOAD=false

# Rebuild database from scratch
ABBA_REBUILD_DB=false
```

### Output Settings

```bash
# Enable verbose output
ABBA_VERBOSE=true

# Enable quiet mode
ABBA_QUIET=false

# Log level (DEBUG, INFO, WARNING, ERROR)
ABBA_LOG_LEVEL=INFO
```

### Advanced Settings

```bash
# Custom path to abba.db
ABBA_DB_PATH=/custom/path/to/abba.db

# Import batch size (default: 1000)
ABBA_IMPORT_BATCH_SIZE=5000

# Connection timeout in seconds (default: 30)
ABBA_DOWNLOAD_TIMEOUT=60

# Number of retry attempts for downloads (default: 3)
ABBA_DOWNLOAD_RETRIES=5
```

### Using .env Files

Create a `.env` file in the project root:

```bash
# .env file example
ABBA_DATA_DIR=/home/user/bible_data
ABBA_TRANSLATIONS=KJV,ASV,ESV,NASB
ABBA_VERBOSE=false
ABBA_QUIET=true
ABBA_FORCE_DOWNLOAD=false
```

The .env file is automatically loaded if present. Example `.env.example` is provided as a template.

## Configuration Files

Configuration files use JSON format and can specify any setting.

### Basic Configuration File

```json
{
  "data_dir": "/path/to/data",
  "translations": ["KJV", "ASV", "ESV"],
  "verbose": false,
  "quiet": true
}
```

### Complete Configuration Example

```json
{
  "data_dir": "/opt/abba/data",
  "bible_db_url": "https://local-mirror.org/bible.db",
  "translations": [
    "eng_kjv",
    "eng_asv", 
    "eng_bbe",
    "heb_wlc",
    "grc_byzantine"
  ],
  "force_download": false,
  "no_download": false,
  "rebuild_db": false,
  "verbose": true,
  "quiet": false,
  "log_level": "INFO",
  "import_batch_size": 2000,
  "download_timeout": 60,
  "download_retries": 5
}
```

### Environment-Specific Configurations

Create multiple configuration files for different environments:

```bash
# Development
python abba/main.py --config-file config/development.json

# Production
python abba/main.py --config-file config/production.json

# Testing
python abba/main.py --config-file config/test.json
```

Example `config/development.json`:
```json
{
  "data_dir": "./dev_data",
  "translations": ["KJV", "ASV"],
  "verbose": true,
  "rebuild_db": true
}
```

Example `config/production.json`:
```json
{
  "data_dir": "/var/lib/abba",
  "translations": ["KJV", "ASV", "ESV", "NIV", "NASB"],
  "verbose": false,
  "quiet": true,
  "import_batch_size": 5000
}
```

## Configuration Loading Process

The configuration system loads settings in this order:

1. **Built-in defaults** are loaded first
2. **Configuration file** is loaded if specified via `--config-file`
3. **Environment variables** override file settings
4. **Command line arguments** override everything

### Example Loading Sequence

Given:
- Default: `verbose=False`
- Config file: `{"verbose": true}`
- Environment: `ABBA_VERBOSE=false`
- CLI: `--verbose`

Result: `verbose=True` (CLI wins)

## Common Configuration Patterns

### Minimal Setup

Just run with defaults:
```bash
python abba/main.py
```

### Development Setup

`.env` file:
```bash
ABBA_DATA_DIR=./dev_data
ABBA_VERBOSE=true
ABBA_TRANSLATIONS=KJV,ASV
```

Command:
```bash
python abba/main.py --rebuild-db
```

### Production Setup

`production.json`:
```json
{
  "data_dir": "/var/lib/abba",
  "quiet": true,
  "import_batch_size": 10000,
  "translations": ["KJV", "ASV", "ESV", "NIV", "NASB", "NLT"]
}
```

Systemd service file:
```ini
[Service]
Environment="ABBA_CONFIG_FILE=/etc/abba/production.json"
ExecStart=/usr/local/bin/python /opt/abba/main.py
```

### CI/CD Setup

```yaml
# GitHub Actions example
env:
  ABBA_DATA_DIR: ${{ github.workspace }}/test_data
  ABBA_TRANSLATIONS: KJV,ASV
  ABBA_QUIET: true
  ABBA_FORCE_DOWNLOAD: true
```

### Docker Setup

`Dockerfile`:
```dockerfile
ENV ABBA_DATA_DIR=/data
ENV ABBA_QUIET=true
ENV ABBA_TRANSLATIONS=KJV,ASV,ESV
```

`docker-compose.yml`:
```yaml
services:
  abba:
    environment:
      - ABBA_DATA_DIR=/data
      - ABBA_VERBOSE=true
      - ABBA_TRANSLATIONS=${TRANSLATIONS:-KJV,ASV}
    volumes:
      - ./data:/data
```

## Validation and Error Handling

### Configuration Validation

The system validates configuration values:

- **data_dir**: Must be a valid directory path
- **translations**: Must be comma-separated string or list
- **URLs**: Must be valid HTTP/HTTPS URLs
- **Booleans**: Accepts true/false, yes/no, 1/0
- **Numbers**: Must be positive integers where applicable

### Error Messages

Common configuration errors and solutions:

```bash
# Invalid translation ID
Error: Translation 'INVALID' not found in bible.db
Solution: Use --list to see available translations

# Invalid data directory
Error: Data directory '/invalid/path' does not exist
Solution: Create directory or use valid path

# Conflicting options
Error: Cannot use --quiet and --verbose together
Solution: Choose one output mode

# Missing required file
Error: bible.db not found at specified path
Solution: Use --force-download or check --data-dir
```

## Best Practices

### 1. Use .env for Personal Preferences

Keep personal settings in `.env` (git-ignored):
```bash
ABBA_DATA_DIR=/home/myuser/bible_data
ABBA_VERBOSE=true
```

### 2. Use Config Files for Project Settings

Share project settings via config files:
```json
{
  "translations": ["KJV", "ASV", "ESV"],
  "import_batch_size": 2000
}
```

### 3. Use CLI for One-Time Overrides

```bash
# Temporarily use verbose mode
python abba/main.py --verbose

# Force rebuild just this once
python abba/main.py --rebuild-db
```

### 4. Document Production Settings

Keep production configuration in version control:
```bash
# config/production.json - DO NOT EDIT without approval
{
  "data_dir": "/var/lib/abba",
  "translations": ["KJV", "ASV", "ESV", "NIV"],
  "quiet": true
}
```

### 5. Use Environment Variables in Containers

For Docker/Kubernetes deployments:
```yaml
env:
  - name: ABBA_DATA_DIR
    value: /data
  - name: ABBA_TRANSLATIONS
    valueFrom:
      configMapKeyRef:
        name: abba-config
        key: translations
```

## Debugging Configuration

### View Effective Configuration

Use verbose mode to see loaded configuration:
```bash
python abba/main.py --verbose --list
```

### Check Environment Variables

```bash
# List all ABBA environment variables
env | grep ^ABBA_

# Check specific variable
echo $ABBA_DATA_DIR
```

### Validate Configuration File

```python
# validate_config.py
import json
import sys

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
    print("Configuration is valid JSON")
    print(f"Settings: {list(config.keys())}")
except Exception as e:
    print(f"Invalid configuration: {e}")
```

## Advanced Configuration

### Custom Configuration Sources

Extend the configuration system:

```python
# custom_config.py
from abba.config import config_manager

# Add custom configuration source
class DatabaseConfig:
    def load(self):
        # Load settings from database
        return {
            "translations": ["KJV", "ASV"],
            "verbose": True
        }

# Register custom source
config_manager.add_source(DatabaseConfig(), priority=2)
```

### Dynamic Configuration

Use environment variables for dynamic settings:

```bash
# Set translations based on user locale
if [ "$LANG" = "es_ES" ]; then
    export ABBA_TRANSLATIONS="spn_vbl,KJV"
else
    export ABBA_TRANSLATIONS="KJV,ASV"
fi

python abba/main.py
```

### Configuration Templates

Create reusable configuration templates:

```python
# generate_config.py
import json
import sys

template = {
    "data_dir": f"/data/{sys.argv[1]}",
    "translations": sys.argv[2].split(","),
    "verbose": "--verbose" in sys.argv
}

with open(f"config_{sys.argv[1]}.json", "w") as f:
    json.dump(template, f, indent=2)
```

Usage:
```bash
python generate_config.py production KJV,ASV,ESV --verbose
python abba/main.py --config-file config_production.json
```