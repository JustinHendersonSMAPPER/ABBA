#!/usr/bin/env python3
"""
Update the CLI to use the new representative alignment models.
"""

import shutil
from pathlib import Path

# Copy the representative models to be the default models
models_dir = Path("models/alignment")

# Backup existing models
if (models_dir / "hebrew_alignment.json").exists():
    shutil.copy(
        models_dir / "hebrew_alignment.json",
        models_dir / "hebrew_alignment_sample.json"
    )

if (models_dir / "greek_alignment.json").exists():
    shutil.copy(
        models_dir / "greek_alignment.json", 
        models_dir / "greek_alignment_sample.json"
    )

# Use representative models as default
shutil.copy(
    models_dir / "hebrew_alignment_representative.json",
    models_dir / "hebrew_alignment.json"
)

shutil.copy(
    models_dir / "greek_alignment_representative.json",
    models_dir / "greek_alignment.json"
)

print("Updated default alignment models to use representative training")
print("Old models backed up with '_sample' suffix")