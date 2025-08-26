#!/usr/bin/env python3
"""Test that purge properly removes import tracking file."""

import json
from pathlib import Path

# Check what tracking files exist
tracking_files = {
    "Import tracker": Path("bible_data/.import_status.json"),
    "Embedding tracker": Path("bible_data/.embedding_progress.json"),
    "State tracker": Path("bible_data/.abba_state.json"),
    "Database": Path("bible_data/abba.db"),
    "Vectors": Path("bible_data/vectors"),
}

print("Current tracking files:")
print("="*50)

for name, path in tracking_files.items():
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            print(f"✓ {name:20} exists ({size:,} bytes)")
            
            # Show some content for JSON files
            if path.suffix == '.json' and size > 0:
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if 'translations' in data:
                            print(f"  - Contains {len(data['translations'])} translations")
                        elif 'verses' in data:
                            print(f"  - Contains verse embeddings")
                        elif 'operations' in data:
                            print(f"  - Contains operation tracking")
                except:
                    pass
        else:
            print(f"✓ {name:20} exists (directory)")
    else:
        print(f"✗ {name:20} not found")

print("\n" + "="*50)
print("Run 'python abba/main.py --purge-all -y' to remove all files")