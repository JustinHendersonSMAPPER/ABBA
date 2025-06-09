"""
ABBA Data Downloader Module

Handles downloading and converting source data for the ABBA project.
"""

import json
import logging
import re
import subprocess
import sys
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
import requests
from tqdm import tqdm

logger = logging.getLogger('ABBA.DataDownloader')


class ABBADataDownloader:
    """Manages downloading and converting source data."""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.sources_dir = self.data_dir / 'sources'
        self.db_dir = self.sources_dir / 'db'
        self.lexicons_dir = self.sources_dir / 'lexicons'
        self.translations_dir = self.sources_dir / 'translations'
        
        # Create directories
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        
    def check_data_exists(self) -> bool:
        """Check if all required source data exists."""
        # Check for bible.db
        bible_db = self.db_dir / 'bible.db'
        if not bible_db.exists():
            return False
        
        # Check if we have extracted translations (don't require all, just some)
        if self.translations_dir.exists():
            translation_files = list(self.translations_dir.glob('*.json'))
            # As long as we have at least one translation, consider data exists
            # We'll check for missing ones separately
            if not translation_files:
                return False
        else:
            return False
            
        return True
    
    def validate_and_sync_translations(self) -> int:
        """Validate existing translations and extract any missing ones."""
        bible_db_path = self.db_dir / 'bible.db'
        
        if not bible_db_path.exists():
            logger.warning("bible.db not found - cannot validate translations")
            return 0
            
        return self._extract_missing_translations()
    
    def download_all(self) -> bool:
        """Download all required source data."""
        try:
            # Download bible.db if needed
            bible_db_path = self.db_dir / 'bible.db'
            if not bible_db_path.exists():
                logger.info("Downloading bible.db...")
                if not self._download_bible_db():
                    return False
                logger.info("✓ Bible database downloaded")
            
            # Extract missing translations from bible.db
            logger.info("Checking for missing translations...")
            missing_count = self._extract_missing_translations()
            if missing_count > 0:
                logger.info(f"✓ Extracted {missing_count} missing translations")
            else:
                logger.info("✓ All translations already extracted")
            
            # Download Hebrew and Greek texts using existing script
            script_path = Path('scripts/download_sources.py')
            if script_path.exists():
                logger.info("Downloading Hebrew and Greek texts...")
                result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Download failed: {result.stderr}")
                    return False
                logger.info("✓ Hebrew and Greek texts downloaded")
            else:
                logger.error("download_sources.py script not found")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _download_bible_db(self) -> bool:
        """Download bible.db from the server."""
        url = "https://bible.helloao.org/bible.db"
        output_path = self.db_dir / 'bible.db'
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading bible.db") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            return True
            
        except Exception as e:
            logger.error(f"Failed to download bible.db: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return False
    
    def _extract_missing_translations(self) -> int:
        """Extract only missing translations from bible.db to JSON files."""
        bible_db_path = self.db_dir / 'bible.db'
        
        if not bible_db_path.exists():
            logger.error("bible.db not found")
            return 0
            
        try:
            conn = sqlite3.connect(bible_db_path)
            cursor = conn.cursor()
            
            # Get list of all translations in database
            cursor.execute("""
                SELECT id, name, englishName, language 
                FROM Translation 
                ORDER BY language, englishName
            """)
            
            all_translations = cursor.fetchall()
            logger.info(f"Found {len(all_translations)} translations in database")
            
            # Check which ones are already extracted
            existing_files = set()
            if self.translations_dir.exists():
                existing_files = {f.stem for f in self.translations_dir.glob('*.json')}
                logger.info(f"Found {len(existing_files)} already extracted translations")
            
            # Find missing translations
            missing_translations = []
            for trans_id, name, english_name, language in all_translations:
                if trans_id not in existing_files:
                    missing_translations.append((trans_id, name, english_name, language))
            
            if not missing_translations:
                logger.info("All translations already extracted")
                conn.close()
                return 0
            
            logger.info(f"Found {len(missing_translations)} missing translations to extract")
            
            # Extract missing translations
            extracted_count = 0
            for trans_id, name, english_name, language in tqdm(missing_translations, desc="Extracting translations"):
                try:
                    logger.debug(f"Extracting {english_name} ({trans_id})")
                    self._extract_single_translation(conn, trans_id)
                    extracted_count += 1
                except Exception as e:
                    logger.error(f"Failed to extract {trans_id}: {e}")
                
            conn.close()
            return extracted_count
            
        except Exception as e:
            logger.error(f"Failed to extract translations: {e}")
            return 0
    
    def _extract_single_translation(self, conn: sqlite3.Connection, translation_id: str):
        """Extract a single translation to JSON format using bible.helloao.org schema."""
        cursor = conn.cursor()
        
        # Get all books for this translation
        cursor.execute("""
            SELECT id, name, commonName, "order", numberOfChapters
            FROM Book
            WHERE translationId = ?
            ORDER BY "order"
        """, (translation_id,))
        
        books_list = cursor.fetchall()
        if not books_list:
            logger.warning(f"No books found for translation {translation_id}")
            return
        
        translation_data = []
        
        for book_id, book_name, common_name, order, num_chapters in books_list:
            # Create book structure
            book_data = {
                'book': book_name,
                'book_id': order,  # Use order as book_id (1-66)
                'chapters': []
            }
            
            # Get all chapters for this book
            for chapter_num in range(1, num_chapters + 1):
                chapter_data = {
                    'chapter': chapter_num,
                    'verses': []
                }
                
                # Get verses for this chapter
                cursor.execute("""
                    SELECT number, text
                    FROM ChapterVerse
                    WHERE translationId = ? AND bookId = ? AND chapterNumber = ?
                    ORDER BY number
                """, (translation_id, book_id, chapter_num))
                
                verses = cursor.fetchall()
                
                for verse_num, verse_text in verses:
                    # Clean verse text
                    verse_text = verse_text.strip() if verse_text else ""
                    verse_text = re.sub(r'\s+', ' ', verse_text)
                    
                    chapter_data['verses'].append({
                        'verse': verse_num,
                        'text': verse_text
                    })
                
                if chapter_data['verses']:
                    book_data['chapters'].append(chapter_data)
            
            if book_data['chapters']:
                translation_data.append(book_data)
        
        # Save to JSON
        output_path = self.translations_dir / f'{translation_id}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)
    
    def check_conversions_needed(self) -> List[Path]:
        """Check which XML files need JSON conversion."""
        conversions_needed = []
        
        if not self.lexicons_dir.exists():
            return conversions_needed
            
        # Check for XML files without corresponding JSON
        for xml_file in self.lexicons_dir.glob('*.xml'):
            json_file = xml_file.with_suffix('.json')
            if not json_file.exists():
                conversions_needed.append(xml_file)
                
        return conversions_needed
    
    def convert_xml_to_json(self, xml_path: Path) -> bool:
        """Convert XML lexicon to JSON format."""
        try:
            # Import the conversion functionality from existing code
            from scripts.download_sources import SourceDownloader
            
            downloader = SourceDownloader()
            
            # Determine language from filename
            if 'hebrew' in xml_path.name.lower():
                language = 'hebrew'
            elif 'greek' in xml_path.name.lower():
                language = 'greek'
            else:
                logger.error(f"Cannot determine language for {xml_path}")
                return False
            
            # Convert
            json_path = xml_path.with_suffix('.json')
            downloader._convert_strongs_xml_to_json(xml_path, json_path, language)
            
            return json_path.exists()
            
        except Exception as e:
            logger.error(f"Failed to convert {xml_path}: {e}")
            return False