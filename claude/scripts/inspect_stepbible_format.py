#!/usr/bin/env python3
"""Inspect actual STEPBible file format to understand the data structure."""

from pathlib import Path
from abba.config import config_manager

def main():
    config = config_manager.load_config([])
    stepbible_dir = config.data_dir / "stepbible"
    
    # Check the actual format of tahot_gen_deu.txt
    tahot_file = stepbible_dir / "tahot_gen_deu.txt"
    
    print("Inspecting tahot_gen_deu.txt format...")
    print("=" * 60)
    
    with open(tahot_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print("\nFirst 30 lines:")
    for i, line in enumerate(lines[:30]):
        print(f"{i+1:3}: {line.rstrip()}")
    
    print("\nLooking for lines starting with $...")
    dollar_lines = []
    for i, line in enumerate(lines[:1000]):  # Check first 1000 lines
        if line.strip().startswith('$'):
            dollar_lines.append((i+1, line.strip()))
            if len(dollar_lines) >= 10:  # Get first 10 dollar lines
                break
    
    if dollar_lines:
        print(f"\nFound {len(dollar_lines)} lines starting with $:")
        for line_num, content in dollar_lines:
            print(f"  Line {line_num}: {content}")
    else:
        print("\nNo lines starting with $ found in first 1000 lines")
        
        # Look for other patterns
        print("\nLooking for other patterns...")
        patterns = ['Gen', 'Genesis', '@', '#', '==']
        for pattern in patterns:
            found_lines = []
            for i, line in enumerate(lines[:100]):
                if pattern in line:
                    found_lines.append((i+1, line.strip()))
                    if len(found_lines) >= 3:
                        break
            if found_lines:
                print(f"\nLines containing '{pattern}':")
                for line_num, content in found_lines:
                    print(f"  Line {line_num}: {content}")

if __name__ == "__main__":
    main()