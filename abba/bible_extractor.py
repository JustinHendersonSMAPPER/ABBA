#!/usr/bin/env python3
"""
Minimal Bible Data Extractor

Downloads bible.db and extracts translations to JSON files.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm


class BibleExtractor:
    """Simple Bible data downloader and extractor."""

    def __init__(self, data_dir: str = "bible_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "bible.db"
        self.translations_dir = self.data_dir / "translations"
        self.stepbible_dir = self.data_dir / "stepbible"

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.translations_dir.mkdir(exist_ok=True)
        self.stepbible_dir.mkdir(exist_ok=True)

    def download_bible_db(self) -> bool:
        """Download bible.db from the server."""
        if self.db_path.exists():
            print(f"bible.db already exists at {self.db_path}")
            return True

        url = "https://bible.helloao.org/bible.db"
        print(f"Downloading bible.db from {url}...")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(self.db_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"✓ Downloaded bible.db to {self.db_path}")
            return True

        except Exception as e:
            print(f"✗ Failed to download bible.db: {e}")
            if self.db_path.exists():
                self.db_path.unlink()
            return False

    def download_stepbible_data(self) -> bool:
        """Download STEPBible lexicon and morphology data."""
        print("Downloading STEPBible data...")

        # STEPBible data URLs - Complete system: texts, lexicons, and morphology
        stepbible_files = {
            # Biblical Texts - TAHOT (Hebrew OT split into 4 files)
            "tahot_gen_deu.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAHOT%20Gen-Deu%20-%20Translators%20Amalgamated%20Hebrew%20OT%20-%20STEPBible.org%20CC%20BY.txt"
            ),
            "tahot_jos_est.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAHOT%20Jos-Est%20-%20Translators%20Amalgamated%20Hebrew%20OT%20-%20STEPBible.org%20CC%20BY.txt"
            ),
            "tahot_job_sng.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAHOT%20Job-Sng%20-%20Translators%20Amalgamated%20Hebrew%20OT%20-%20STEPBible.org%20CC%20BY.txt"
            ),
            "tahot_isa_mal.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAHOT%20Isa-Mal%20-%20Translators%20Amalgamated%20Hebrew%20OT%20-%20STEPBible.org%20CC%20BY.txt"
            ),
            # Biblical Texts - TAGNT (Greek NT split into 2 files)
            "tagnt_mat_jhn.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAGNT%20Mat-Jhn%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt"
            ),
            "tagnt_act_rev.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Translators%20Amalgamated%20OT%2BNT/"
                "TAGNT%20Act-Rev%20-%20Translators%20Amalgamated%20Greek%20NT%20-%20STEPBible.org%20CC-BY.txt"
            ),
            # Lexicons (definitions/meanings)
            "hebrew_lexicon.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Lexicons/"
                "TBESH - Translators Brief lexicon of Extended Strongs for Hebrew - STEPBible.org CC BY.txt"
            ),
            "greek_lexicon.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/Lexicons/"
                "TBESG - Translators Brief lexicon of Extended Strongs for Greek - STEPBible.org CC BY.txt"
            ),
            # Morphology code explanations
            "hebrew_morphology.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/"
                "TEHMC - Translators Expansion of Hebrew Morphology Codes - STEPBible.org CC BY.txt"
            ),
            "greek_morphology.txt": (
                "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/"
                "TEGMC - Translators Expansion of Greek Morphhology Codes - STEPBible.org CC BY.txt"
            ),
        }

        success_count = 0

        for filename, url in stepbible_files.items():
            file_path = self.stepbible_dir / filename

            if file_path.exists():
                print(f"✓ {filename} already exists")
                success_count += 1
                continue

            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(file_path, "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                print(f"✓ Downloaded {filename}")
                success_count += 1

            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                if file_path.exists():
                    file_path.unlink()

        # Create attribution file
        attribution_path = self.stepbible_dir / "ATTRIBUTION.txt"
        attribution_content = """STEPBible Data Attribution
========================

This folder contains data from the STEPBible project, used under the CC BY 4.0 license.

Source: STEPBible.org
Repository: https://github.com/STEPBible/STEPBible-Data
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
License URL: https://creativecommons.org/licenses/by/4.0/

Files included:
- Hebrew OT text (TAHOT - Translators Amalgamated Hebrew OT, 4 files: Gen-Deu, Jos-Est, Job-Sng, Isa-Mal)
- Greek NT text (TAGNT - Translators Amalgamated Greek NT, 2 files: Mat-Jhn, Act-Rev)
- Hebrew lexicon (TBESH - Translators Brief lexicon of Extended Strongs for Hebrew)
- Greek lexicon (TBESG - Translators Brief lexicon of Extended Strongs for Greek)
- Hebrew morphology codes (TEHMC - Translators Expansion of Hebrew Morphology Codes)
- Greek morphology codes (TEGMC - Translators Expansion of Greek Morphology Codes)

Attribution: This data is provided by STEPBible.org and Tyndale House, Cambridge.
For more information, corrections, or updates, contact: STEPBible@gmail.com

The data includes lexical, morphological, and textual information for Hebrew, Aramaic,
and Koine Greek biblical texts, designed to support biblical language study and research.
"""

        try:
            with open(attribution_path, "w", encoding="utf-8") as f:
                f.write(attribution_content)
            print(f"✓ Created attribution file at {attribution_path}")
        except Exception as e:
            print(f"✗ Failed to create attribution file: {e}")

        print(f"STEPBible data download complete: {success_count}/{len(stepbible_files)} files downloaded")

        # Consider success if we got all essential files
        # We need at least the lexicons and morphology files
        core_files = [
            "hebrew_lexicon.txt",
            "greek_lexicon.txt",  # Lexicons
            "hebrew_morphology.txt",
            "greek_morphology.txt",  # Morphology
        ]
        core_success = all((self.stepbible_dir / f).exists() for f in core_files)

        # Check if we have at least some text files (ideally all 6)
        text_files = [
            "tahot_gen_deu.txt",
            "tahot_jos_est.txt",
            "tahot_job_sng.txt",
            "tahot_isa_mal.txt",
            "tagnt_mat_jhn.txt",
            "tagnt_act_rev.txt",
        ]
        text_count = sum(1 for f in text_files if (self.stepbible_dir / f).exists())

        # Success if we have core files and at least some text files
        return core_success and (text_count >= 2 or success_count == len(stepbible_files))

    def parse_stepbible_lexicon(self, language: str, db_manager) -> bool:
        """Parse STEPBible lexicon files and import into database.

        Args:
            language: 'hebrew' or 'greek'
            db_manager: SQLiteManager instance

        Returns:
            True if successful, False otherwise
        """
        lexicon_file = f"{language}_lexicon.txt"
        lexicon_path = self.stepbible_dir / lexicon_file

        if not lexicon_path.exists():
            print(f"Lexicon file not found: {lexicon_path}")
            return False

        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse lexicon entries - STEPBible format varies but generally:
            # Strong's|Original|Transliteration|PartOfSpeech|Gloss|Definition
            entries_added = 0
            for _line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("="):
                    continue

                # Split on tab or pipe character
                parts = line.split("\t") if "\t" in line else line.split("|")
                if len(parts) < 4:
                    continue

                # Extract lexicon data
                strongs_number = parts[0].strip()
                original_word = parts[1].strip() if len(parts) > 1 else ""
                transliteration = parts[2].strip() if len(parts) > 2 else ""
                part_of_speech = parts[3].strip() if len(parts) > 3 else ""
                gloss = parts[4].strip() if len(parts) > 4 else ""
                definition = parts[5].strip() if len(parts) > 5 else ""

                if not strongs_number:
                    continue

                lexicon_data = {
                    "strongs_number": strongs_number,
                    "original_word": original_word,
                    "transliteration": transliteration,
                    "part_of_speech": part_of_speech,
                    "gloss": gloss,
                    "definition": definition,
                    "language": language,
                }

                db_manager.insert_lexicon_entry(lexicon_data)
                entries_added += 1

            print(f"✓ Imported {entries_added} {language} lexicon entries")
            return True

        except Exception as e:
            print(f"✗ Failed to parse {language} lexicon: {e}")
            return False

    def parse_stepbible_morphology(self, language: str, db_manager) -> bool:
        """Parse STEPBible morphology files and import into database.

        Args:
            language: 'hebrew' or 'greek'
            db_manager: SQLiteManager instance

        Returns:
            True if successful, False otherwise
        """
        morphology_file = f"{language}_morphology.txt"
        morphology_path = self.stepbible_dir / morphology_file

        if not morphology_path.exists():
            print(f"Morphology file not found: {morphology_path}")
            return False

        try:
            with open(morphology_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse morphology codes - format varies but generally:
            # Code|Description|Components
            entries_added = 0
            for _line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("="):
                    continue

                # Split on tab or pipe character
                parts = line.split("\t") if "\t" in line else line.split("|")
                if len(parts) < 2:
                    continue

                code = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                components = parts[2].strip() if len(parts) > 2 else ""

                if not code:
                    continue

                morphology_data = {
                    "code": code,
                    "description": description,
                    "components": components,
                    "language": language,
                }

                db_manager.insert_morphology_entry(morphology_data)
                entries_added += 1

            print(f"✓ Imported {entries_added} {language} morphology codes")
            return True

        except Exception as e:
            print(f"✗ Failed to parse {language} morphology: {e}")
            return False

    def parse_stepbible_text(self, filename: str, db_manager) -> bool:
        """Parse STEPBible TAHOT/TAGNT text files and import word data.

        Args:
            filename: Name of the text file (e.g., 'tahot_gen_deu.txt')
            db_manager: SQLiteManager instance

        Returns:
            True if successful, False otherwise
        """
        text_path = self.stepbible_dir / filename

        if not text_path.exists():
            print(f"Text file not found: {text_path}")
            return False

        # Determine language and file type
        language = "hebrew" if filename.startswith("tahot") else "greek"
        is_greek = language == "greek"

        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse word data - STEPBible format differs between Hebrew and Greek:
            # Hebrew (TAHOT): Book.Chapter.Verse#WordNum=Source<TAB>HebrewText<TAB>Transliteration<TAB>Translation<TAB>StrongsRaw<TAB>Morphology<TAB>...<TAB>StrongsPrimary
            # Greek (TAGNT): Book.Chapter.Verse#WordNum=Source<TAB>GreekText (transliteration)<TAB>EnglishGloss<TAB>Strongs=Morphology<TAB>LexiconEntry<TAB>...
            words_added = 0

            for _line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                    
                # Skip header lines and comments
                if (line.startswith("#") or line.startswith("=") or 
                    line.startswith("TAHOT") or line.startswith("TAGNT") or
                    "STEPBible.org" in line or "Data created by" in line or
                    line.startswith("FIELD DESCRIPTIONS") or "licence allows" in line):
                    continue

                # Parse data lines that match pattern: Book.Chapter.Verse#WordNum=Source
                if "." in line and "#" in line and "=" in line and "\t" in line:
                    try:
                        # Split by tabs
                        parts = line.split("\t")
                        if len(parts) < 9:  # Need at least 9 parts for essential data
                            continue

                        # Parse reference: Gen.1.1#01=L
                        ref_part = parts[0]
                        if "=" not in ref_part:
                            continue
                            
                        ref_no_source = ref_part.split("=")[0]  # Remove =L part
                        
                        # Extract book.chapter.verse#word
                        if "#" not in ref_no_source:
                            continue
                            
                        verse_ref, word_num_str = ref_no_source.split("#")
                        
                        # Parse book.chapter.verse
                        ref_parts = verse_ref.split(".")
                        if len(ref_parts) < 3:
                            continue
                            
                        book = ref_parts[0]
                        chapter = int(ref_parts[1])
                        verse = int(ref_parts[2])
                        word_num = int(word_num_str)

                        # Extract data from tab-separated parts
                        # Column layout differs between Hebrew and Greek
                        if is_greek:
                            # Greek format (TAGNT):
                            # 1: Greek text with transliteration (Βίβλος (Biblos))
                            # 2: English gloss ([The] book)
                            # 3: Strong's + morphology (G0976=N-NSF)
                            # 4: Lexicon entry (βίβλος=book)
                            
                            greek_with_trans = parts[1] if len(parts) > 1 else ""
                            english_gloss = parts[2] if len(parts) > 2 else ""
                            strongs_morph = parts[3] if len(parts) > 3 else ""
                            
                            # Extract Greek text and transliteration
                            if "(" in greek_with_trans and ")" in greek_with_trans:
                                greek_text = greek_with_trans.split("(")[0].strip()
                                transliteration = greek_with_trans.split("(")[1].rstrip(")")
                            else:
                                greek_text = greek_with_trans
                                transliteration = ""
                            
                            # Extract Strong's number from format like "G0976=N-NSF"
                            if "=" in strongs_morph:
                                strongs_primary = strongs_morph.split("=")[0]
                                morphology_code = strongs_morph.split("=")[1]
                            else:
                                strongs_primary = strongs_morph
                                morphology_code = ""
                            
                            # Use English gloss as translation
                            translation = english_gloss
                            hebrew_text = ""
                            strongs_raw = strongs_morph
                            
                        else:
                            # Hebrew format (TAHOT) - existing logic
                            hebrew_text = parts[1] if len(parts) > 1 else ""
                            transliteration = parts[2] if len(parts) > 2 else ""
                            translation = parts[3] if len(parts) > 3 else ""
                            strongs_raw = parts[4] if len(parts) > 4 else ""
                            morphology_code = parts[5] if len(parts) > 5 else ""
                            strongs_primary = parts[8] if len(parts) > 8 else ""
                            greek_text = ""

                        # Clean up the Hebrew/Greek text (remove pointing markers, etc.)
                        if hebrew_text:
                            # Remove common STEPBible markers like / and \ 
                            hebrew_text = hebrew_text.replace("/", "").replace("\\", "")
                            # Remove trailing punctuation markers
                            hebrew_text = hebrew_text.rstrip("׃־֑֖֥֣֖֑֣֥֛֢֣֤֥֦֧֪֭֮֔֔֗֙֜֝֞֟֠֡֨֩֫֬֯")

                        # Extract primary Strong's number
                        if strongs_primary:
                            # Clean up Strong's number - remove suffixes like _A, _B
                            clean_strongs = strongs_primary.split("_")[0]
                        elif strongs_raw:
                            # Try to extract from raw if primary is empty
                            if "{" in strongs_raw and "}" in strongs_raw:
                                # Extract from {H1234} format
                                start = strongs_raw.find("{") + 1
                                end = strongs_raw.find("}")
                                clean_strongs = strongs_raw[start:end].split()[0] if start < end else ""
                            else:
                                clean_strongs = strongs_raw.split()[0] if strongs_raw else ""
                        else:
                            clean_strongs = ""

                        word_data = {
                            "book": book,
                            "chapter": chapter,
                            "verse": verse,
                            "word_num": word_num,
                            "word_ref": f"{book}.{chapter}.{verse}.{word_num}",
                            "hebrew_text": hebrew_text if language == "hebrew" else None,
                            "greek_text": greek_text if language == "greek" else None,
                            "transliteration": transliteration,
                            "translation": translation,
                            "strongs_raw": strongs_raw,
                            "strongs_primary": clean_strongs,
                            "morphology_code": morphology_code,
                            "language": language,
                        }

                        db_manager.insert_word(word_data)
                        words_added += 1

                    except (ValueError, IndexError) as e:
                        # Skip lines that don't parse correctly (likely headers or metadata)
                        continue
                    except Exception as e:
                        print(f"Error parsing line {_line_num}: {e}")
                        continue

            print(f"✓ Imported {words_added} words from {filename}")
            return True

        except Exception as e:
            print(f"✗ Failed to parse {filename}: {e}")
            return False

    def import_stepbible_data(self, db_manager, tracker=None) -> bool:
        """Import all STEPBible data into the database.

        Args:
            db_manager: SQLiteManager instance
            tracker: ImportTracker instance for tracking progress (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.stepbible_dir.exists():
            print("STEPBible data directory not found")
            return False

        print("Importing STEPBible data into database...")

        # Parse lexicons
        lexicon_files = [
            ("lexicon", "tbesh.txt"),  # Hebrew lexicon
            ("lexicon", "tbesg.txt"),  # Greek lexicon
        ]
        
        for file_type, filename in lexicon_files:
            if tracker and tracker.is_stepbible_file_imported(file_type, filename):
                print(f"  Skipping {filename} - already imported")
                continue
                
            if filename.startswith("tbesh"):
                success = self.parse_stepbible_lexicon("hebrew", db_manager)
            else:
                success = self.parse_stepbible_lexicon("greek", db_manager)
                
            if success and tracker:
                tracker.mark_stepbible_file_imported(file_type, filename)

        # Parse morphology
        morph_files = [
            ("morphology", "tehmc.txt"),  # Hebrew morphology
            ("morphology", "tegmc.txt"),  # Greek morphology
        ]
        
        for file_type, filename in morph_files:
            if tracker and tracker.is_stepbible_file_imported(file_type, filename):
                print(f"  Skipping {filename} - already imported")
                continue
                
            if filename.startswith("teh"):
                self.parse_stepbible_morphology("hebrew", db_manager)
            else:
                self.parse_stepbible_morphology("greek", db_manager)
                
            if tracker:
                tracker.mark_stepbible_file_imported(file_type, filename)

        # Parse text files
        text_files = [
            ("tahot", "tahot_gen_deu.txt"),
            ("tahot", "tahot_jos_est.txt"),
            ("tahot", "tahot_job_sng.txt"),
            ("tahot", "tahot_isa_mal.txt"),
            ("tagnt", "tagnt_mat_jhn.txt"),
            ("tagnt", "tagnt_act_rev.txt"),
        ]

        text_success_count = 0
        for file_type, text_file in text_files:
            if tracker and tracker.is_stepbible_file_imported(file_type, text_file):
                print(f"  Skipping {text_file} - already imported")
                text_success_count += 1
                continue
            if self.parse_stepbible_text(text_file, db_manager):
                text_success_count += 1
                if tracker:
                    tracker.mark_stepbible_file_imported(file_type, text_file)

        # Consider success if we imported at least some files
        # Count already-imported files as successful
        total_expected = len(lexicon_files) + len(morph_files) + len(text_files)
        imported_count = 0
        
        if tracker:
            summary = tracker.get_import_summary()
            for file_type in ["lexicon", "morphology", "tahot", "tagnt"]:
                imported_count += summary["stepbible_files"].get(file_type, 0)
        
        success = imported_count > 0 or text_success_count > 0

        if success:
            print(f"✓ STEPBible import complete")
        else:
            print("✗ STEPBible import failed")

        return success

    def list_translations(self) -> List[Dict[str, str]]:
        """List all available translations in bible.db."""
        if not self.db_path.exists():
            print("bible.db not found. Please download it first.")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, name, englishName, language
                FROM Translation
                ORDER BY language, englishName
            """
            )

            translations = []
            for trans_id, name, english_name, language in cursor.fetchall():
                translations.append({"id": trans_id, "name": name, "english_name": english_name, "language": language})

            conn.close()
            return translations

        except Exception as e:
            print(f"Error listing translations: {e}")
            return []

    def extract_translation(self, translation_id: str) -> bool:
        """Extract a single translation to JSON format."""
        if not self.db_path.exists():
            print("bible.db not found. Please download it first.")
            return False

        output_path = self.translations_dir / f"{translation_id}.json"
        if output_path.exists():
            print(f"Translation {translation_id} already extracted")
            return True

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get translation info
            cursor.execute(
                """
                SELECT name, englishName, language
                FROM Translation
                WHERE id = ?
            """,
                (translation_id,),
            )

            result = cursor.fetchone()
            if not result:
                print(f"Translation {translation_id} not found")
                return False

            name, english_name, language = result
            print(f"Extracting {english_name} ({translation_id})...")

            # Get all books
            cursor.execute(
                """
                SELECT id, name, commonName, "order", numberOfChapters
                FROM Book
                WHERE translationId = ?
                ORDER BY "order"
            """,
                (translation_id,),
            )

            books_list = cursor.fetchall()
            if not books_list:
                print(f"No books found for translation {translation_id}")
                return False

            translation_data = {
                "translation_id": translation_id,
                "name": name,
                "english_name": english_name,
                "language": language,
                "books": [],
            }

            for book_id, book_name, _common_name, order, _num_chapters in tqdm(books_list, desc="Books"):
                book_data = {"book": book_name, "book_id": order, "chapters": []}

                # Get all verses for this book
                cursor.execute(
                    """
                    SELECT chapterNumber, number, text
                    FROM ChapterVerse
                    WHERE translationId = ? AND bookId = ?
                    ORDER BY chapterNumber, number
                """,
                    (translation_id, book_id),
                )

                verses = cursor.fetchall()

                # Group by chapter
                current_chapter = None
                chapter_data = None

                for chapter_num, verse_num, verse_text in verses:
                    if chapter_num != current_chapter:
                        if chapter_data:
                            book_data["chapters"].append(chapter_data)
                        current_chapter = chapter_num
                        chapter_data = {"chapter": chapter_num, "verses": []}

                    if chapter_data:
                        chapter_data["verses"].append(
                            {"verse": verse_num, "text": verse_text.strip() if verse_text else ""}
                        )

                # Add last chapter
                if chapter_data:
                    book_data["chapters"].append(chapter_data)

                if book_data["chapters"]:
                    translation_data["books"].append(book_data)

            # Save to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translation_data, f, ensure_ascii=False, indent=2)

            conn.close()
            print(f"✓ Extracted {english_name} to {output_path}")
            return True

        except Exception as e:
            print(f"✗ Failed to extract {translation_id}: {e}")
            return False

    def import_translation_to_db(self, translation_id: str, db_manager) -> bool:
        """Import a translation directly into the ABBA database.

        Args:
            translation_id: Translation identifier
            db_manager: SQLiteManager instance

        Returns:
            True if successful, False otherwise
        """
        if not self.db_path.exists():
            print("bible.db not found. Please download it first.")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get translation info
            cursor.execute(
                """
                SELECT name, englishName, language
                FROM Translation
                WHERE id = ?
            """,
                (translation_id,),
            )

            result = cursor.fetchone()
            if not result:
                print(f"Translation {translation_id} not found")
                return False

            name, english_name, language = result

            # Get all books for this translation
            cursor.execute(
                """
                SELECT id, name, commonName, "order", numberOfChapters
                FROM Book
                WHERE translationId = ?
                ORDER BY "order"
            """,
                (translation_id,),
            )

            books_list = cursor.fetchall()
            if not books_list:
                print(f"No books found for translation {translation_id}")
                return False

            # Import books metadata
            for book_id, book_name, _common_name, order, _num_chapters in books_list:
                book_data = {
                    "translation_id": translation_id,
                    "book_id": order,  # Use order as book_id (integer)
                    "name": book_name,
                    "common_name": _common_name,
                    "book_order": order,
                    "number_of_chapters": _num_chapters,
                    "testament": "old" if order <= 39 else "new",
                }

                # Insert book using raw SQL
                try:
                    with db_manager.get_connection() as db_conn:
                        db_conn.execute(
                            """
                            INSERT OR REPLACE INTO books
                            (translation_id, book_id, name, common_name, book_order, number_of_chapters, testament)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                book_data["translation_id"],
                                book_data["book_id"],
                                book_data["name"],
                                book_data["common_name"],
                                book_data["book_order"],
                                book_data["number_of_chapters"],
                                book_data["testament"],
                            ),
                        )
                except Exception as e:
                    print(f"✗ Error inserting book {book_name}: {e}")
                    raise

                # Get all verses for this book
                cursor.execute(
                    """
                    SELECT chapterNumber, number, text
                    FROM ChapterVerse
                    WHERE translationId = ? AND bookId = ?
                    ORDER BY chapterNumber, number
                """,
                    (translation_id, book_id),
                )

                verses = cursor.fetchall()

                # Import verses in batches for better performance
                verse_batch = []
                for chapter_num, verse_num, verse_text in verses:
                    # Create tuple for SQL insertion
                    verse_tuple = (
                        str(translation_id),  # translation_id
                        int(order),  # book_id (book order from our schema)
                        int(chapter_num),  # chapter
                        int(verse_num),  # verse
                        str(verse_text.strip()) if verse_text else "",  # text
                    )
                    verse_batch.append(verse_tuple)

                    # Insert in batches of 1000
                    if len(verse_batch) >= 1000:
                        self._insert_verse_batch(db_manager, verse_batch)
                        verse_batch = []

                # Insert remaining verses
                if verse_batch:
                    self._insert_verse_batch(db_manager, verse_batch)

            conn.close()
            return True

        except Exception as e:
            print(f"✗ Failed to import {translation_id}: {e}")
            return False

    def _insert_verse_batch(self, db_manager, verse_batch):
        """Insert a batch of verses into the database."""
        with db_manager.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO verses (translation_id, book_id, chapter, verse, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                verse_batch,
            )
            conn.commit()

    def extract_all_translations(self):
        """Extract all translations from bible.db."""
        translations = self.list_translations()

        if not translations:
            print("No translations found")
            return

        print(f"Found {len(translations)} translations")

        success_count = 0
        for trans in tqdm(translations, desc="Extracting translations"):
            if self.extract_translation(trans["id"]):
                success_count += 1

        print(f"\n✓ Successfully extracted {success_count}/{len(translations)} translations")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Bible Data Extractor")
    parser.add_argument("--data-dir", default="bible_data", help="Data directory (default: bible_data)")
    parser.add_argument("--download", action="store_true", help="Download bible.db")
    parser.add_argument("--list", action="store_true", help="List available translations")
    parser.add_argument("--extract", metavar="ID", help="Extract specific translation by ID")
    parser.add_argument("--extract-all", action="store_true", help="Extract all translations")

    args = parser.parse_args()

    extractor = BibleExtractor(args.data_dir)

    # Download bible.db if requested
    if args.download:
        if not extractor.download_bible_db():
            sys.exit(1)

    # List translations
    if args.list:
        translations = extractor.list_translations()
        if translations:
            print(f"\nAvailable translations ({len(translations)}):")
            print("-" * 60)
            for trans in translations:
                print(f"{trans['id']:10} {trans['language']:10} {trans['english_name']}")
        else:
            print("No translations found")

    # Extract specific translation
    if args.extract:
        if not extractor.extract_translation(args.extract):
            sys.exit(1)

    # Extract all translations
    if args.extract_all:
        extractor.extract_all_translations()

    # If no action specified, show help
    if not any([args.download, args.list, args.extract, args.extract_all]):
        parser.print_help()


if __name__ == "__main__":
    main()
