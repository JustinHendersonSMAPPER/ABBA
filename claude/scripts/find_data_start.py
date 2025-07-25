#!/usr/bin/env python3
"""Find where the actual biblical data starts in STEPBible files."""

from pathlib import Path
from abba.config import config_manager

def main():
    config = config_manager.load_config([])
    stepbible_dir = config.data_dir / "stepbible"
    tahot_file = stepbible_dir / "tahot_gen_deu.txt"
    
    print("Looking for data format patterns...")
    
    with open(tahot_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Look for lines containing Gen.1.1 or similar patterns
    data_lines = []
    for i, line in enumerate(lines):
        if "Gen.1.1" in line and "#" in line:
            data_lines.append((i+1, line.strip()))
            if len(data_lines) >= 10:
                break
    
    if data_lines:
        print(f"Found {len(data_lines)} lines with Gen.1.1:")
        for line_num, content in data_lines:
            print(f"  Line {line_num}: {content}")
    
    # Look for the general pattern: book.chapter.verse#word
    print(f"\nLooking for lines with pattern like 'Gen.*.#'...")
    pattern_lines = []
    for i, line in enumerate(lines[80:150]):  # Check around where we found the first one
        if "Gen." in line and "#" in line and "\t" in line:
            pattern_lines.append((i+81, line.strip()))
            if len(pattern_lines) >= 20:
                break
    
    if pattern_lines:
        print(f"Found {len(pattern_lines)} data lines:")
        for line_num, content in pattern_lines[:10]:
            print(f"  Line {line_num}: {content}")
            
        # Analyze the format
        print(f"\nAnalyzing format of first data line:")
        first_line = pattern_lines[0][1]
        print(f"Raw: {repr(first_line)}")
        
        # Split by tabs
        parts = first_line.split('\t')
        print(f"Tab-separated parts ({len(parts)}):")
        for i, part in enumerate(parts):
            print(f"  {i}: {repr(part)}")

if __name__ == "__main__":
    main()