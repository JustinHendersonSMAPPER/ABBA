#!/usr/bin/env python3
"""
Fix STEPBible Schema Script

Updates the stepbible_verses table to include:
1. Normalized text columns (without vowel points/accents)
2. Clean Strong's numbers (extracted from braces)
3. Proper indexing for efficient searches

This will make concept validation and searches much more efficient.
"""

import sys
import re
import unicodedata
from pathlib import Path
import sqlite3

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.database.sqlite_manager import SQLiteManager
from abba.logging_setup import logger


def normalize_hebrew_text(text: str) -> str:
    """Remove Hebrew vowel points and cantillation marks."""
    if not text:
        return text
    
    # Remove morphological separators
    text = text.replace('/', '').replace('\\', '')
    
    # Remove niqqud (vowel points) and cantillation marks
    # Unicode ranges: 0591-05C7 (Hebrew accents and points)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    
    return text.strip()


def normalize_greek_text(text: str) -> str:
    """Remove Greek accents and breathing marks."""
    if not text:
        return text
    
    # Normalize to NFD (decomposed) then remove combining marks
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text.strip()


def extract_strongs_numbers(strongs_raw: str) -> list:
    """Extract clean Strong's numbers from raw format.
    
    Examples:
    - H9003/{H7225G} -> ['H7225']
    - {H1254A} -> ['H1254']
    - G3962 -> ['G3962']
    - {H0430G}+{H0559} -> ['H0430', 'H0559']
    """
    if not strongs_raw:
        return []
    
    numbers = []
    
    # Find all numbers in braces
    brace_matches = re.findall(r'\{([HG]\d+)', strongs_raw)
    numbers.extend(brace_matches)
    
    # If no braces found and it looks like a direct Strong's number
    if not brace_matches and re.match(r'^[HG]\d+$', strongs_raw.strip()):
        numbers.append(strongs_raw.strip())
    
    return numbers


def update_schema(db_path: Path):
    """Update the database schema to add normalized columns."""
    logger.info(f"Updating schema in {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(stepbible_verses)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add normalized text columns if they don't exist
        if 'normalized_word' not in columns:
            logger.info("Adding normalized_word column...")
            cursor.execute("ALTER TABLE stepbible_verses ADD COLUMN normalized_word TEXT")
        
        if 'strongs_lexical' not in columns:
            logger.info("Adding strongs_lexical column...")
            cursor.execute("ALTER TABLE stepbible_verses ADD COLUMN strongs_lexical TEXT")
        
        # Create indexes for efficient searching
        logger.info("Creating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stepbible_normalized_word 
            ON stepbible_verses(normalized_word)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stepbible_strongs_lexical 
            ON stepbible_verses(strongs_lexical)
        """)
        
        conn.commit()


def populate_normalized_data(db_path: Path):
    """Populate the normalized columns with processed data."""
    logger.info("Populating normalized data...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Process in batches for efficiency
        batch_size = 10000
        offset = 0
        
        while True:
            # Get batch of records - also fetch strongs_primary for Greek
            cursor.execute("""
                SELECT id, original_word, language, strongs_raw, strongs_primary
                FROM stepbible_verses
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            updates = []
            for row_id, original_word, language, strongs_raw, strongs_primary in rows:
                # Normalize text based on language
                if language == 'hebrew':
                    normalized = normalize_hebrew_text(original_word)
                elif language == 'greek':
                    normalized = normalize_greek_text(original_word)
                else:
                    normalized = original_word
                
                # Extract clean Strong's numbers
                if language == 'greek' and strongs_primary and not strongs_raw:
                    # Greek uses strongs_primary directly
                    strongs_lexical = strongs_primary
                else:
                    # Hebrew uses extraction from strongs_raw
                    strongs_list = extract_strongs_numbers(strongs_raw)
                    strongs_lexical = ','.join(strongs_list) if strongs_list else None
                
                updates.append((normalized, strongs_lexical, row_id))
            
            # Update in batch
            cursor.executemany("""
                UPDATE stepbible_verses
                SET normalized_word = ?, strongs_lexical = ?
                WHERE id = ?
            """, updates)
            
            conn.commit()
            logger.info(f"Processed {offset + len(rows)} records...")
            
            offset += batch_size


def verify_update(db_path: Path):
    """Verify the update worked correctly."""
    logger.info("Verifying update...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check some examples
        test_cases = [
            ("Hebrew אלהים", "SELECT COUNT(*) FROM stepbible_verses WHERE normalized_word = 'אלהים'"),
            ("Greek θεός", "SELECT COUNT(*) FROM stepbible_verses WHERE normalized_word LIKE '%θεος%'"),
            ("Strong's H430", "SELECT COUNT(*) FROM stepbible_verses WHERE strongs_lexical LIKE '%H430%'"),
            ("Strong's G2316", "SELECT COUNT(*) FROM stepbible_verses WHERE strongs_lexical = 'G2316'"),
        ]
        
        for description, query in test_cases:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            logger.info(f"{description}: {count} occurrences")


def main():
    """Main function to update STEPBible schema."""
    # Load configuration
    config = config_manager.load_config()
    db_path = config.data_dir / "abba.db"
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False
    
    try:
        # Update schema
        update_schema(db_path)
        
        # Populate normalized data
        populate_normalized_data(db_path)
        
        # Verify
        verify_update(db_path)
        
        logger.info("✅ Schema update completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)