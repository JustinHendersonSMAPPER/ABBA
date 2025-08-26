#!/usr/bin/env python3
"""
Direct test of semantic concordance for the 'love' concept.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_semantic_concept_evaluation import test_semantic_concept

# Test love concept with limited semantic results
if __name__ == "__main__":
    success = test_semantic_concept(
        'love',
        max_semantic_results=20,
        validate_semantic=True
    )
    sys.exit(0 if success else 1)