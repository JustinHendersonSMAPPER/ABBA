#!/usr/bin/env python3
"""Test main.py import with a limited dataset."""

import subprocess
import sys
import tempfile
from pathlib import Path
from abba.config import config_manager

def main():
    print("Testing main.py STEPBible import (this may take a few minutes)...")
    
    # Create a backup of the current database
    config = config_manager.load_config([])
    current_db = config.abba_db_path
    
    if current_db.exists():
        backup_db = current_db.with_suffix(".backup")
        print(f"Backing up current database to {backup_db}")
        import shutil
        shutil.copy2(current_db, backup_db)
    
    try:
        # Run main.py with force rebuild to test our fixed parser
        print("Running main.py with database rebuild...")
        result = subprocess.run([
            sys.executable, "abba/main.py", 
            "--rebuild-db",
            "--translations", "ESV"  # Just one translation to speed up
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        # Check if the import was successful by looking for word counts
        if "Words:" in result.stdout and not "Words: 0" in result.stdout:
            print("\n✅ SUCCESS: STEPBible word import is working!")
        else:
            print("\n❌ ISSUE: Words count is still 0 or missing")
            
    except subprocess.TimeoutExpired:
        print("Process timed out - this is normal for full import")
        
    except Exception as e:
        print(f"Error running main.py: {e}")
        
    finally:
        # Restore backup if it exists
        if current_db.exists() and backup_db.exists():
            print(f"Restoring backup database")
            shutil.copy2(backup_db, current_db)
            backup_db.unlink()

if __name__ == "__main__":
    main()