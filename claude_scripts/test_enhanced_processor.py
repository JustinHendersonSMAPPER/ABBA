#!/usr/bin/env python3
"""
Test the enhanced Bible processor with all real data integration.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the enhanced processor."""
    # Path to the enhanced CLI
    cli_path = Path(__file__).parent.parent / "src" / "abba" / "cli_enhanced.py"
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output" / "enhanced_test"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the processor with limited books for testing
    cmd = [
        sys.executable,
        str(cli_path),
        "--data-dir", str(data_dir),
        "--output", str(output_dir),
        "--translations", "eng_web", "eng_kjv",
        "--books", "Gen", "Mat", "Jhn",  # Just Genesis, Matthew, and John for testing
        "--log-level", "INFO"
    ]
    
    print("Running enhanced Bible processor...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Processing completed successfully!")
            print(f"\nCheck output in: {output_dir}")
            
            # List generated files
            print("\nGenerated files:")
            for file in output_dir.iterdir():
                print(f"  - {file.name}")
        else:
            print(f"\n✗ Processing failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Error running processor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())