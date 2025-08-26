#!/usr/bin/env python3
"""Test the --purge-all functionality."""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_purge():
    """Test that purge-all properly removes all data."""
    print("Testing --purge-all functionality")
    print("="*60)
    
    # Get data directory
    data_dir = Path("bible_data")
    
    # Files/directories that should be removed
    targets = [
        data_dir / "abba.db",
        data_dir / "vectors",
        data_dir / ".import_progress.json",
        data_dir / ".embedding_progress.json",
        data_dir / ".abba_state.json"
    ]
    
    # Check what currently exists
    print("\nChecking current state:")
    for target in targets:
        exists = "EXISTS" if target.exists() else "NOT FOUND"
        print(f"  {target}: {exists}")
    
    # Create some dummy files to test removal
    print("\nCreating test files...")
    data_dir.mkdir(exist_ok=True)
    
    # Create dummy tracking files
    for file in [f for f in targets if f.suffix == '.json']:
        file.write_text('{"test": true}')
        
    # Create dummy vectors directory
    vectors_dir = data_dir / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    (vectors_dir / "test.txt").write_text("test")
    
    # Create dummy database
    db_path = data_dir / "abba.db"
    db_path.write_text("dummy database")
    
    print("\nAfter creating test files:")
    for target in targets:
        exists = "EXISTS" if target.exists() else "NOT FOUND"
        print(f"  {target}: {exists}")
    
    # Test with --yes flag (auto-confirm)
    print("\nTesting purge with --yes flag...")
    
    # Run the purge command with --yes to skip confirmation
    cmd = ["python", "abba/main.py", "--purge-all", "--yes"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # No need to send input with --yes flag
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error running purge: {stderr}")
        return False
    
    # Check that all files were removed
    print("\nAfter purge:")
    all_removed = True
    for target in targets:
        exists = target.exists()
        status = "STILL EXISTS!" if exists else "REMOVED"
        print(f"  {target}: {status}")
        if exists:
            all_removed = False
    
    if all_removed:
        print("\n✅ All files successfully removed!")
        return True
    else:
        print("\n❌ Some files were not removed!")
        return False

def test_purge_cancellation():
    """Test that purge can be cancelled."""
    print("\n\nTesting purge cancellation")
    print("="*60)
    
    # Create a test file
    data_dir = Path("bible_data")
    data_dir.mkdir(exist_ok=True)
    test_file = data_dir / ".test_cancel.json"
    test_file.write_text('{"test": true}')
    
    print(f"Created test file: {test_file}")
    
    # Run purge but cancel it
    cmd = ["python", "abba/main.py", "--purge-all"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send 'no' to cancel
    stdout, stderr = process.communicate(input="no\n")
    
    # Check that file still exists
    if test_file.exists():
        print("✅ File still exists - cancellation worked!")
        test_file.unlink()  # Clean up
        return True
    else:
        print("❌ File was removed - cancellation failed!")
        return False

if __name__ == "__main__":
    # Test both purge and cancellation
    purge_ok = test_purge()
    cancel_ok = test_purge_cancellation()
    
    if purge_ok and cancel_ok:
        print("\n\n✅ All purge tests passed!")
    else:
        print("\n\n❌ Some tests failed!")
        sys.exit(1)