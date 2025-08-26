#!/usr/bin/env python3
"""
Show how to update parallel_import.py to use canon-aware book validation.
"""

print("""
To implement canon-aware import, update parallel_import.py:

1. Add the canon mapping near the top of the file:

# Add after the BOOK_ID_MAP definition
from enum import Enum

class Canon(Enum):
    PROTESTANT = "protestant"
    CATHOLIC = "catholic"
    ORTHODOX = "orthodox"

# Simple translation pattern matching
def get_translation_canon(translation_id: str) -> Canon:
    tid = translation_id.upper()
    
    # Catholic indicators
    if any(x in tid for x in ['NABRE', 'DRC', 'CPDV', 'CE', 'CATHOLIC', 'NJB']):
        return Canon.CATHOLIC
        
    # Orthodox indicators
    if any(x in tid for x in ['EOB', 'OSB', 'ORTHODOX', 'LXX']):
        return Canon.ORTHODOX
        
    # Default to Protestant
    return Canon.PROTESTANT

# Extended book map for Catholic/Orthodox
EXTENDED_BOOKS = {
    Canon.CATHOLIC: {'TOB', 'JDT', 'ESG', 'WIS', 'SIR', 'BAR', '1MA', '2MA'},
    Canon.ORTHODOX: {'TOB', 'JDT', 'ESG', 'WIS', 'SIR', 'BAR', '1MA', '2MA', 
                     '1ES', '3MA', 'MAN', 'PS2', '4MA'}
}

2. Update the verse import method to check canon:

# In _import_verses_batch method, replace:
            if book_id == 0:
                logger.warning(f"Unknown book ID: {book_str}")
                continue

# With:
            if book_id == 0:
                # Check if this book is expected for this translation's canon
                canon = get_translation_canon(translation_id)
                if canon != Canon.PROTESTANT and book_str in EXTENDED_BOOKS.get(canon, set()):
                    # This is a deuterocanonical book - skip silently
                    continue
                else:
                    # This is truly unknown - warn
                    logger.warning(f"Unknown book ID: {book_str} in {translation_id}")
                continue

This way:
- Catholic Bibles with WIS (Wisdom) won't generate warnings
- Protestant Bibles with WIS would still warn (unexpected)
- Truly unknown books still generate warnings
""")