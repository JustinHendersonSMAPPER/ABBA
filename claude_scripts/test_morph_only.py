#!/usr/bin/env python3
"""Test morphology parsing directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abba.morphology.hebrew_morphology import HebrewMorphology

def main():
    # Create parser
    morph = HebrewMorphology()
    
    # Check if it's an instance
    print(f"Type: {type(morph)}")
    print(f"Has parse method: {hasattr(morph, 'parse')}")
    
    # List all methods
    print("\nAvailable methods:")
    for attr in dir(morph):
        if not attr.startswith('_'):
            print(f"  {attr}")

if __name__ == "__main__":
    main()