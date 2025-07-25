#!/usr/bin/env python3
"""Fix the STEPBible parser to correctly extract Strong's numbers for Greek text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def parse_stepbible_text_fixed(self, filename: str, db_manager) -> bool:
    """Parse STEPBible TAHOT/TAGNT text files and import word data.
    
    This is a fixed version that correctly handles the different column layouts
    for Hebrew (TAHOT) and Greek (TAGNT) files.
    
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
                        # 0: Reference (Mat.1.1#01=NKO)
                        # 1: Greek text with transliteration (Βίβλος (Biblos))
                        # 2: English gloss ([The] book)
                        # 3: Strong's + morphology (G0976=N-NSF)
                        # 4: Lexicon entry (βίβλος=book)
                        # 5: Manuscripts
                        # 6-7: Usually empty
                        # 8: Spanish translation (Libro)
                        # 9: English word (book)
                        
                        greek_with_trans = parts[1] if len(parts) > 1 else ""
                        english_gloss = parts[2] if len(parts) > 2 else ""
                        strongs_morph = parts[3] if len(parts) > 3 else ""
                        lexicon_entry = parts[4] if len(parts) > 4 else ""
                        
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
                        
                    else:
                        # Hebrew format (TAHOT):
                        # 0: Reference (Gen.1.1#01=L)
                        # 1: Hebrew text (בְּ/רֵאשִׁ֖ית)
                        # 2: Transliteration (be./re.Shit)
                        # 3: Translation (in/ beginning)
                        # 4: Strong's raw (H9003/{H7225G})
                        # 5: Morphology (HR/Ncfsa)
                        # 6-7: Usually empty
                        # 8: Strong's primary (H7225G)
                        
                        hebrew_text = parts[1] if len(parts) > 1 else ""
                        transliteration = parts[2] if len(parts) > 2 else ""
                        translation = parts[3] if len(parts) > 3 else ""
                        strongs_raw = parts[4] if len(parts) > 4 else ""
                        morphology_code = parts[5] if len(parts) > 5 else ""
                        strongs_primary = parts[8] if len(parts) > 8 else ""
                        
                        # Clean up Hebrew text
                        if hebrew_text:
                            # Remove common STEPBible markers like / and \ 
                            greek_text = ""
                            hebrew_text = hebrew_text.replace("/", "").replace("\\", "")
                            # Remove trailing punctuation markers
                            hebrew_text = hebrew_text.rstrip("׃־֑֖֥֣֖֑֣֥֛֢֣֤֥֦֧֪֭֮֔֔֗֙֜֝֞֟֠֡֨֩֫֬֯")
                        else:
                            hebrew_text = ""
                            greek_text = ""
                        
                        # Extract primary Strong's number
                        if not strongs_primary and strongs_raw:
                            # Try to extract from raw if primary is empty
                            if "{" in strongs_raw and "}" in strongs_raw:
                                # Extract from {H1234} format
                                start = strongs_raw.find("{") + 1
                                end = strongs_raw.find("}")
                                strongs_primary = strongs_raw[start:end].split()[0] if start < end else ""
                            else:
                                strongs_primary = strongs_raw.split()[0] if strongs_raw else ""
                    
                    # Clean up Strong's number - remove suffixes like _A, _B, G, H at end
                    if strongs_primary:
                        clean_strongs = strongs_primary.split("_")[0]
                        # Remove trailing G or H if it's a suffix (like H7225G)
                        if len(clean_strongs) > 2 and clean_strongs[-1] in ['G', 'H'] and clean_strongs[-2].isdigit():
                            clean_strongs = clean_strongs[:-1]
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
                        "strongs_raw": strongs_raw if language == "hebrew" else strongs_morph,
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


if __name__ == "__main__":
    print("This script contains the fixed parse_stepbible_text method.")
    print("\nTo fix the issue, the parse_stepbible_text method in bible_extractor.py")
    print("needs to be updated to handle the different column layouts for Hebrew and Greek files.")
    print("\nKey changes:")
    print("1. Detect if file is Greek (TAGNT) or Hebrew (TAHOT)")
    print("2. For Greek files, extract Strong's from column 3 (format: G0976=N-NSF)")
    print("3. For Hebrew files, keep existing logic (Strong's in column 8)")
    print("4. For Greek files, use column 2 as the English gloss/translation")