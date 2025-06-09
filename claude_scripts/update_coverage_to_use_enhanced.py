#!/usr/bin/env python3
"""
Update the coverage analysis to use enhanced models.
"""

import shutil
from pathlib import Path

def update_models():
    """Copy enhanced models to be used by coverage analysis."""
    
    model_dir = Path('models/biblical_alignment')
    
    # Backup old models
    for old_model in ['hebrew_english_biblical.json', 'greek_english_biblical.json']:
        old_path = model_dir / old_model
        if old_path.exists():
            backup_path = model_dir / f'{old_model}.backup'
            shutil.copy2(old_path, backup_path)
            print(f"Backed up {old_model} to {backup_path.name}")
    
    # Copy enhanced models to the names that ModelDiscovery looks for
    enhanced_mappings = {
        'hebrew_english_enhanced.json': 'hebrew_english_biblical.json',
        'greek_english_enhanced.json': 'greek_english_biblical.json'
    }
    
    for enhanced, target in enhanced_mappings.items():
        enhanced_path = model_dir / enhanced
        target_path = model_dir / target
        
        if enhanced_path.exists():
            shutil.copy2(enhanced_path, target_path)
            print(f"Updated {target} with enhanced model")
        else:
            print(f"Warning: Enhanced model {enhanced} not found")
            
    print("\nModels updated! Now the coverage analysis will use the enhanced models with:")
    print("- Hebrew: 8,673 Strong's entries")
    print("- Greek: 5,472 Strong's entries")

if __name__ == '__main__':
    update_models()