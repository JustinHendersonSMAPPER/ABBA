#!/usr/bin/env python3
"""Test the STEPBible update checker."""

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.stepbible_updater import STEPBibleUpdater

def test_update_checker():
    """Test the STEPBible update checking functionality."""
    print("Testing STEPBible Update Checker")
    print("="*60)
    
    data_dir = Path("bible_data")
    stepbible_dir = data_dir / "stepbible"
    
    if not stepbible_dir.exists():
        print("❌ STEPBible directory not found. Run main.py first to download.")
        return
    
    # List current files and their hashes
    print("\nCurrent STEPBible files:")
    print("-"*60)
    
    files = sorted(stepbible_dir.glob("*.txt"))
    for file_path in files:
        if file_path.name == "ATTRIBUTION.txt":
            continue
            
        # Calculate hash
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"{file_path.name:30} {size_mb:6.1f} MB  {sha256_hash.hexdigest()[:16]}...")
    
    print("\n" + "="*60)
    print("Update check process:")
    print("-"*60)
    print("1. Downloads STEPBible files to temporary location")
    print("2. Compares SHA256 hashes with existing files")
    print("3. If changes detected:")
    print("   - Backs up current files")
    print("   - Updates with new versions")
    print("   - Clears import tracking to force re-import")
    print("   - Triggers word embedding regeneration")
    print("\n" + "="*60)
    
    # Test the updater (without actually running it)
    updater = STEPBibleUpdater(data_dir)
    print(f"\n✅ STEPBibleUpdater initialized for: {data_dir}")
    print("\nTo check for updates, run:")
    print("  python abba/main.py --check-for-updates")
    print("\nThis will:")
    print("  - Check for any changes in STEPBible data")
    print("  - Re-import STEPBible data if updates found")
    print("  - Re-generate word embeddings automatically")

def show_tracking_status():
    """Show current import tracking status."""
    import json
    
    print("\n\nCurrent Import Tracking Status")
    print("="*60)
    
    import_status_file = Path("bible_data/.import_status.json")
    if import_status_file.exists():
        with open(import_status_file) as f:
            data = json.load(f)
        
        if 'stepbible_files' in data:
            files = data['stepbible_files']
            if files:
                print("STEPBible files tracked as imported:")
                for key, timestamp in files.items():
                    print(f"  - {key}: {timestamp}")
            else:
                print("No STEPBible files tracked yet")
        else:
            print("No STEPBible tracking data found")
    else:
        print("Import status file not found")

if __name__ == "__main__":
    test_update_checker()
    show_tracking_status()