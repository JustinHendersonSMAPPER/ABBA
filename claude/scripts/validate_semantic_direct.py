#!/usr/bin/env python3
"""Direct validation without CLI interference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validate_semantic_concordance import main

if __name__ == "__main__":
    sys.exit(0 if main() else 1)