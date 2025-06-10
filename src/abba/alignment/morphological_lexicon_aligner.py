"""
Morphological Lexicon-Based Aligner

Uses the existing OSHB and MorphGNT morphological data as the lexicon source.
This approach leverages the rich lemma and morphological information already
available in our data to create proper semantic alignments.

Instead of hard-coding lexicon entries, this aligner:
1. Uses OSHB lemmas for Hebrew word definitions  
2. Uses MorphGNT lemmas for Greek word definitions
3. Handles morphological variations through the existing parsing data
4. Creates alignments based on lemma-to-translation correspondence
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import re

logger = logging.getLogger('ABBA.MorphologicalLexiconAligner')


class MorphologicalLexiconLookup:
    """
    Lexicon lookup using existing OSHB and MorphGNT morphological data.
    This uses the lemmas and morphological information already in our data.
    """
    
    def __init__(self):
        self.hebrew_lemma_cache = {}
        self.greek_lemma_cache = {}
        self.morphology_data_loaded = False
        self._load_morphological_data()
        
    def _load_morphological_data(self):
        """Load morphological data from OSHB and MorphGNT to build lemma dictionary."""
        hebrew_morph_dir = Path("data/sources/morphology/hebrew")
        greek_morph_dir = Path("data/sources/morphology/greek")
        
        # Load Hebrew lemmas from OSHB data
        if hebrew_morph_dir.exists():
            for json_file in hebrew_morph_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._extract_hebrew_lemmas(data)
                except Exception as e:
                    logger.warning(f"Error loading Hebrew morphology from {json_file}: {e}")
        
        # Load Greek lemmas from MorphGNT data
        if greek_morph_dir.exists():
            for json_file in greek_morph_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._extract_greek_lemmas(data)
                except Exception as e:
                    logger.warning(f"Error loading Greek morphology from {json_file}: {e}")
        
        self.morphology_data_loaded = True
        logger.info(f"Loaded {len(self.hebrew_lemma_cache)} Hebrew lemmas, {len(self.greek_lemma_cache)} Greek lemmas from morphological data")
    
    def _extract_hebrew_lemmas(self, data: Dict):
        """Extract Hebrew lemmas from OSHB morphological data."""
        for verse in data.get('verses', []):
            for word in verse.get('words', []):
                text = word.get('text', '').strip()
                lemma = word.get('lemma', '').strip()
                pos = word.get('morph', word.get('POS', word.get('pos', ''))).strip()
                
                if text and lemma:
                    # Clean the text (remove vowel points and slashes)
                    clean_text = self._clean_hebrew_word(text)
                    # Don't clean the lemma - preserve compound structure like 'l/120'
                    
                    # Store multiple forms pointing to same lemma
                    for form in [clean_text, text]:
                        if form not in self.hebrew_lemma_cache:
                            self.hebrew_lemma_cache[form] = {
                                'lemma': lemma,  # Store original lemma with compound structure
                                'pos': pos,
                                'forms': set()
                            }
                        self.hebrew_lemma_cache[form]['forms'].add(clean_text)
    
    def _extract_greek_lemmas(self, data: Dict):
        """Extract Greek lemmas from MorphGNT morphological data."""
        for verse in data.get('verses', []):
            for word in verse.get('words', []):
                text = word.get('text', '').strip()
                normalized = word.get('normalized', '').strip()
                lemma = word.get('lemma', '').strip()
                pos = word.get('parse', word.get('pos', '')).strip()
                
                if lemma:
                    # Clean the text (remove diacritics)
                    clean_text = self._clean_greek_word(text)
                    clean_normalized = self._clean_greek_word(normalized)
                    clean_lemma = self._clean_greek_word(lemma)
                    
                    # Store multiple forms pointing to same lemma
                    for form in [clean_text, text, clean_normalized, normalized]:
                        if form and form not in self.greek_lemma_cache:
                            self.greek_lemma_cache[form] = {
                                'lemma': clean_lemma,
                                'pos': pos,
                                'forms': set()
                            }
                        if form:
                            self.greek_lemma_cache[form]['forms'].add(clean_text)
    
    def _clean_hebrew_word(self, word: str) -> str:
        """Clean Hebrew word by removing vowel points and morphological markup."""
        if not word:
            return ""
        # Remove vowel points and cantillation marks (Unicode range U+0591 to U+05C7)
        cleaned = re.sub(r'[\u0591-\u05C7]', '', word)
        # Remove slashes used in morphological markup
        cleaned = cleaned.replace('/', '')
        return cleaned.strip()
    
    def _clean_greek_word(self, word: str) -> str:
        """Clean Greek word by removing diacritics."""
        if not word:
            return ""
        # Remove diacritics (Unicode combining marks)
        cleaned = re.sub(r'[\u0300-\u036F]', '', word)
        return cleaned.lower().strip()
    
    def lookup_lemma(self, word: str, language: str) -> Optional[Dict]:
        """
        Look up lemma for a word using morphological data.
        Returns lemma information if found.
        """
        if not self.morphology_data_loaded:
            return None
        
        cache = self.hebrew_lemma_cache if language == 'hebrew' else self.greek_lemma_cache
        clean_word = self._clean_hebrew_word(word) if language == 'hebrew' else self._clean_greek_word(word)
        
        # Direct lookup
        if clean_word in cache:
            return cache[clean_word]
        
        # Try original word
        if word in cache:
            return cache[word]
        
        # For Hebrew, try removing common prefixes
        if language == 'hebrew':
            return self._hebrew_prefix_lookup(word, cache)
        
        # For Greek, try partial matching for inflected forms
        elif language == 'greek':
            return self._greek_inflection_lookup(word, cache)
        
        return None
    
    def _hebrew_prefix_lookup(self, word: str, cache: Dict) -> Optional[Dict]:
        """Handle Hebrew words with prefixes by trying to strip common prefixes."""
        clean_word = self._clean_hebrew_word(word)
        
        # Common Hebrew prefixes
        prefixes = ['ו', 'ה', 'ב', 'ל', 'כ', 'מ']
        
        # Try removing one or two prefixes
        for i in range(1, min(3, len(clean_word))):
            potential_prefix = clean_word[:i]
            potential_root = clean_word[i:]
            
            # Check if this looks like a prefix combination
            if all(char in prefixes for char in potential_prefix):
                if potential_root in cache:
                    return cache[potential_root]
        
        # Try partial matching
        for cached_word, entry in cache.items():
            if len(cached_word) >= 3 and len(clean_word) >= 3:
                # Check if root is contained (allowing for prefixes/suffixes)
                if cached_word in clean_word or clean_word in cached_word:
                    if abs(len(cached_word) - len(clean_word)) <= 3:  # Allow some variation
                        return entry
        
        return None
    
    def _greek_inflection_lookup(self, word: str, cache: Dict) -> Optional[Dict]:
        """Handle Greek inflected forms by partial matching."""
        clean_word = self._clean_greek_word(word)
        
        # Try partial matching for inflected forms
        for cached_word, entry in cache.items():
            if len(cached_word) >= 3 and len(clean_word) >= 3:
                # Check for stem similarity
                if (cached_word[:3] == clean_word[:3] or  # Same stem
                    cached_word in clean_word or clean_word in cached_word):
                    return entry
        
        return None
    
    def get_english_glosses(self, lemma: str, pos: str, language: str) -> List[str]:
        """
        Get English glosses for a lemma using comprehensive lexicon data.
        First tries external lexicons (BDB, Thayer's, Strong's), then falls back to basic mappings.
        """
        # Try comprehensive lexicons first
        comprehensive_glosses = self._get_comprehensive_glosses(lemma, language)
        if comprehensive_glosses:
            return comprehensive_glosses
        
        # Fall back to basic mappings
        if language == 'hebrew':
            return self._get_hebrew_glosses(lemma, pos)
        elif language == 'greek':
            return self._get_greek_glosses(lemma, pos)
        return []
    
    def _get_comprehensive_glosses(self, lemma: str, language: str) -> List[str]:
        """
        Get glosses from comprehensive lexicon files (BDB, Thayer's, Strong's).
        Handles both Hebrew/Greek lemmas and Strong's numbers.
        """
        glosses = []
        
        # Extract Strong's number if present (e.g., 'b7225' -> '7225', '430' -> '430')
        strongs_number = self._extract_strongs_number(lemma)
        
        # Handle conjunction prefixes (e.g., 'c853' = 'and' + '853')
        conjunction_glosses = self._get_conjunction_glosses(lemma)
        glosses.extend(conjunction_glosses)
        
        if language == 'hebrew':
            # Try basic Hebrew lexicon first (uses Strong's numbers as keys)
            basic_hebrew_file = "data/lexicons/hebrew_lexicon.json"
            glosses.extend(self._extract_from_strongs_lexicon(strongs_number, basic_hebrew_file))
            
            # Try comprehensive BDB lexicon files
            bdb_files = [
                "data/lexicons/bdb_lexicon.xml",
                "data/lexicons/bdb_step_lexicon.xml", 
                "data/lexicons/bdb_morphological.json",
                "data/lexicons/enhanced_hebrew_lexicon.json"
            ]
            glosses.extend(self._extract_from_lexicon_files(lemma, bdb_files, 'hebrew'))
            
            # Try Strong's Hebrew
            strongs_files = [
                "data/lexicons/strongs_hebrew.json",
                "data/lexicons/strongs_hebrew.xml"
            ]
            glosses.extend(self._extract_from_lexicon_files(strongs_number, strongs_files, 'hebrew'))
            
        elif language == 'greek':
            # Try basic Greek lexicon first (uses Strong's numbers as keys)
            basic_greek_file = "data/lexicons/greek_lexicon.json"
            glosses.extend(self._extract_from_strongs_lexicon(strongs_number, basic_greek_file))
            
            # Try comprehensive Thayer's lexicon files
            thayers_files = [
                "data/lexicons/thayers_lexicon.xml",
                "data/lexicons/thayers_openscriptures.xml",
                "data/lexicons/thayers_morphological.json",
                "data/lexicons/enhanced_greek_lexicon.json"
            ]
            glosses.extend(self._extract_from_lexicon_files(lemma, thayers_files, 'greek'))
            
            # Try Strong's Greek
            strongs_files = [
                "data/lexicons/strongs_greek.json",
                "data/lexicons/strongs_greek.xml"
            ]
            glosses.extend(self._extract_from_lexicon_files(strongs_number, strongs_files, 'greek'))
        
        # Remove duplicates and return
        return list(set(glosses))
    
    def _extract_strongs_number(self, lemma: str) -> str:
        """Extract Strong's number from lemma (e.g., 'b7225' -> '7225', '430' -> '430')."""
        if not lemma:
            return ""
        
        # Remove common prefixes from OSHB morphological data
        clean_lemma = lemma
        for prefix in ['b', 'c', 'd', 'a']:  # Common OSHB prefixes
            if clean_lemma.startswith(prefix) and len(clean_lemma) > 1:
                clean_lemma = clean_lemma[1:]
                break
        
        # Remove common suffixes (e.g., '1254 a' -> '1254')
        clean_lemma = clean_lemma.split()[0]
        
        return clean_lemma
    
    def _get_conjunction_glosses(self, lemma: str) -> List[str]:
        """Extract conjunction glosses for compound forms like 'c853' (and + particle)."""
        if lemma.startswith('c') and len(lemma) > 1:
            # This is a conjunction + word combination
            return ["and"]  # The conjunction part
        return []
    
    def _extract_from_strongs_lexicon(self, strongs_number: str, file_path: str) -> List[str]:
        """Extract definitions from Strong's number-keyed lexicon files."""
        if not strongs_number:
            return []
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return []
            
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Look for entries by Strong's number
            if 'entries' in data:
                entry = data['entries'].get(strongs_number)
                if entry and isinstance(entry, dict):
                    meanings = entry.get('meanings', [])
                    if meanings:
                        logger.debug(f"Found {len(meanings)} meanings for Strong's {strongs_number}: {meanings}")
                        return meanings
            
        except Exception as e:
            logger.debug(f"Error reading Strong's lexicon {file_path}: {e}")
        
        return []
    
    def _extract_from_lexicon_files(self, lemma: str, file_paths: List[str], language: str) -> List[str]:
        """Extract definitions from lexicon files."""
        glosses = []
        clean_lemma = self._clean_hebrew_word(lemma) if language == 'hebrew' else self._clean_greek_word(lemma)
        
        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    continue
                
                if file_path.endswith('.json'):
                    glosses.extend(self._extract_from_json_lexicon(file_path_obj, lemma, clean_lemma))
                elif file_path.endswith('.xml'):
                    glosses.extend(self._extract_from_xml_lexicon(file_path_obj, lemma, clean_lemma, language))
                    
            except Exception as e:
                logger.debug(f"Error reading lexicon file {file_path}: {e}")
        
        return glosses
    
    def _extract_from_json_lexicon(self, file_path: Path, lemma: str, clean_lemma: str) -> List[str]:
        """Extract definitions from JSON lexicon files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if 'entries' in data:
                entries = data['entries']
                # Try exact matches first
                for key, entry in entries.items():
                    if (isinstance(entry, dict) and 
                        (entry.get('hebrew') == lemma or entry.get('greek') == lemma or
                         entry.get('hebrew') == clean_lemma or entry.get('greek') == clean_lemma)):
                        return entry.get('meanings', [])
                
                # Try partial matches
                for key, entry in entries.items():
                    if isinstance(entry, dict):
                        word_forms = [entry.get('hebrew', ''), entry.get('greek', '')]
                        if any(lemma in form or clean_lemma in form for form in word_forms if form):
                            return entry.get('meanings', [])
            
        except Exception as e:
            logger.debug(f"Error parsing JSON lexicon {file_path}: {e}")
        
        return []
    
    def _extract_from_xml_lexicon(self, file_path: Path, lemma: str, clean_lemma: str, language: str) -> List[str]:
        """Extract definitions from XML lexicon files (BDB, Thayer's, Strong's)."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Look for entries matching the lemma
            # This is a simplified approach - real XML parsing would be more complex
            for entry in root.findall('.//entry'):
                word_elem = entry.find('word') or entry.find('lemma') or entry.find('headword')
                if word_elem is not None:
                    word_text = word_elem.text or ''
                    if lemma in word_text or clean_lemma in word_text:
                        # Extract definitions
                        definition_elems = entry.findall('.//definition') or entry.findall('.//meaning') or entry.findall('.//gloss')
                        definitions = []
                        for def_elem in definition_elems:
                            if def_elem.text:
                                # Split on common separators and clean
                                defs = [d.strip() for d in def_elem.text.split(',') if d.strip()]
                                definitions.extend(defs)
                        if definitions:
                            return definitions[:5]  # Limit to first 5 definitions
                            
        except Exception as e:
            logger.debug(f"Error parsing XML lexicon {file_path}: {e}")
        
        return []
    
    def _get_hebrew_glosses(self, lemma: str, pos: str) -> List[str]:
        """Fallback Hebrew glosses - only used when comprehensive lexicons unavailable."""
        # Minimal fallback for most common words only
        # Note: This should rarely be used if BDB/Strong's lexicons are available
        basic_hebrew_glosses = {
            # Core function words that appear in every verse
            'את': [],  # Untranslatable object marker
            'ו': ['and'],
            'ה': ['the'],
            'ב': ['in', 'on', 'with'],
            'ל': ['to', 'for'],
            'מ': ['from', 'of'],
            'כ': ['like', 'as'],
            # Most frequent content words 
            'אלהים': ['God', 'god'],
            'ארץ': ['earth', 'land'],
            'שמים': ['heaven', 'heavens'],
            'יום': ['day'],
            'מים': ['water', 'waters'],
            'איש': ['man', 'person'],
            'בית': ['house', 'home'],
            'יד': ['hand'],
            'עין': ['eye'],
        }
        
        clean_lemma = self._clean_hebrew_word(lemma)
        glosses = basic_hebrew_glosses.get(clean_lemma, basic_hebrew_glosses.get(lemma, []))
        
        if not glosses and clean_lemma:
            logger.debug(f"No fallback Hebrew gloss for lemma: {lemma} (cleaned: {clean_lemma}) - comprehensive lexicons should provide this")
        
        return glosses
    
    def _get_greek_glosses(self, lemma: str, pos: str) -> List[str]:
        """Fallback Greek glosses - only used when comprehensive lexicons unavailable."""
        # Minimal fallback for most common words only
        # Note: This should rarely be used if Thayer's/Strong's lexicons are available
        basic_greek_glosses = {
            # Core function words that appear in every verse
            'ὁ': ['the'],
            'καί': ['and', 'also'],
            'ἐν': ['in', 'on'],
            'εἰς': ['into', 'to'],
            'ἐκ': ['from', 'out of'],
            'πρός': ['with', 'to', 'toward'],
            'μετά': ['with', 'after'],
            'διά': ['through', 'because of'],
            'ὅτι': ['that', 'because'],
            # Most frequent content words
            'θεός': ['God', 'god'],
            'λόγος': ['word', 'Word'],
            'εἰμί': ['am', 'is', 'are', 'was', 'were', 'be'],
            'κόσμος': ['world'],
            'ἄνθρωπος': ['man', 'human', 'person'],
            'ἡμέρα': ['day'],
            'χείρ': ['hand'],
            'ὀφθαλμός': ['eye'],
        }
        
        clean_lemma = self._clean_greek_word(lemma)
        glosses = basic_greek_glosses.get(clean_lemma, basic_greek_glosses.get(lemma, []))
        
        if not glosses and clean_lemma:
            logger.debug(f"No fallback Greek gloss for lemma: {lemma} (cleaned: {clean_lemma}) - comprehensive lexicons should provide this")
        
        return glosses


class MorphologicalAnalyzer:
    """
    Provides detailed morphological analysis including transliteration,
    literal meaning, and grammatical notes.
    """
    
    def __init__(self):
        # Basic transliteration mappings
        self.hebrew_transliteration = {
            'א': "'", 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'w', 'ז': 'z',
            'ח': 'ḥ', 'ט': 'ṭ', 'י': 'y', 'כ': 'k', 'ל': 'l', 'מ': 'm', 'נ': 'n',
            'ס': 's', 'ע': "'", 'פ': 'p', 'צ': 'ṣ', 'ק': 'q', 'ר': 'r', 'ש': 'š',
            'ת': 't', 'ך': 'k', 'ם': 'm', 'ן': 'n', 'ף': 'p', 'ץ': 'ṣ'
        }
        
        # Common morphological patterns
        self.hebrew_patterns = {
            'בְּרֵאשִׁית': {
                'transliteration': 'bə-rēʾšít',
                'literal': 'in (the) beginning',
                'grammar': 'ב preposition "in/at" + רֵאשִׁית "beginning, first part"; feminine sing. construct'
            },
            'בָּרָא': {
                'transliteration': 'bārā\'',
                'literal': 'he created', 
                'grammar': 'Qal perfect, 3rd masc. sing.; used only of divine creation'
            },
            'אֱלֹהִים': {
                'transliteration': "'ĕlōhím",
                'literal': 'God',
                'grammar': 'Morphologically plural but takes singular verbs (plural of majesty)'
            },
            'אֵת': {
                'transliteration': "'ēt",
                'literal': '(object marker)',
                'grammar': 'Untranslated particle marking definite direct object'
            },
            'הַשָּׁמַיִם': {
                'transliteration': 'ha-šāmayim',
                'literal': 'the heavens',
                'grammar': 'ה definite article + plural/dual noun heavens, sky'
            },
            'וְאֵת': {
                'transliteration': 'wə-\'ēt',
                'literal': 'and (object marker)',
                'grammar': 'ו conjunction and + אֵת object marker'
            },
            'הָאָרֶץ': {
                'transliteration': 'hā-\'āreṣ',
                'literal': 'the earth/land',
                'grammar': 'ה definite article + noun earth, land, fem. sing.'
            }
        }
    
    def get_breakdown(self, word: str, language: str, lemma: str = '', glosses: List[str] = None, matched_gloss: str = '', morph_code: str = '', pos: str = '') -> Dict:
        """Get detailed morphological breakdown for a word."""
        if language == 'hebrew':
            return self._get_hebrew_breakdown(word, lemma, glosses or [], matched_gloss, morph_code, pos)
        elif language == 'greek':
            return self._get_greek_breakdown(word, lemma, glosses or [], matched_gloss, morph_code, pos)
        return {}
    
    def _get_hebrew_breakdown(self, word: str, lemma: str = '', glosses: List[str] = None, matched_gloss: str = '', morph_code: str = '', pos: str = '') -> Dict:
        """Get Hebrew morphological breakdown using lexicon data when available."""
        glosses = glosses or []
        
        # Clean word of vowel points for pattern matching
        clean_word = re.sub(r'[\u0591-\u05C7]', '', word)
        
        # Check for known patterns first
        if word in self.hebrew_patterns:
            return self.hebrew_patterns[word]
        elif clean_word in self.hebrew_patterns:
            return self.hebrew_patterns[clean_word]
        
        # Generate proper transliteration with vowel points
        transliteration = self._transliterate_hebrew_with_vowels(word)
        
        # Use matched_gloss as root meaning when available, fall back to first gloss
        root_meaning = ''
        if matched_gloss:
            # Prefer the matched gloss (what was actually aligned)
            root_meaning = matched_gloss
        elif glosses:
            # Fall back to first gloss if no matched gloss
            root_meaning = glosses[0]
        
        # Analyze morphological components
        grammar_notes = []
        prefix_parts = []
        literal_meaning = ''  # Initialize literal_meaning variable
        
        # Enhanced grammatical analysis with detailed explanations
        detailed_analysis = self._get_detailed_grammatical_analysis(word, clean_word, lemma, root_meaning, morph_code, pos)
        grammar_notes = detailed_analysis['grammar_notes']
        prefix_parts = detailed_analysis['prefix_parts']
        literal_meaning = detailed_analysis['literal_meaning'] or literal_meaning
        
        # Construct the final literal meaning combining prefixes with root meaning
        if prefix_parts and root_meaning:
            # Combine prefixes with root meaning (e.g., "in" + "beginning" = "in (the) beginning")
            if len(prefix_parts) == 1:
                # Add parentheses around articles for clarity
                if prefix_parts[0] == 'the':
                    literal_meaning = f"({prefix_parts[0]}) {root_meaning}"
                else:
                    literal_meaning = f"{prefix_parts[0]} (the) {root_meaning}"
            else:
                # Multiple prefixes (e.g., "and the")
                literal_meaning = f"{' '.join(prefix_parts)} {root_meaning}"
        elif prefix_parts and not root_meaning:
            # Only prefixes, no root meaning
            literal_meaning = ' '.join(prefix_parts)
        elif root_meaning:
            # Only root meaning, no prefixes
            literal_meaning = root_meaning
        else:
            # Fallback to transliteration if no lexicon data
            literal_meaning = transliteration
        
        # Add grammatical description - join all detailed notes
        grammar_description = '. '.join(grammar_notes) if grammar_notes else 'Hebrew word'
        
        # Extract structured morphological features
        morphological_features = self._extract_morphological_features(morph_code, pos, lemma, word)
        
        result = {
            'transliteration': transliteration,
            'literal': literal_meaning,
            'grammar': grammar_description
        }
        
        # Add morphological features (only include fields that have values)
        result.update(morphological_features)
        
        return result
    
    def _get_greek_breakdown(self, word: str, lemma: str = '', glosses: List[str] = None, matched_gloss: str = '', morph_code: str = '', pos: str = '') -> Dict:
        """Get Greek morphological breakdown."""
        glosses = glosses or []
        
        # Use matched_gloss as literal meaning when available, fall back to first gloss
        if matched_gloss:
            literal_meaning = matched_gloss
        elif glosses:
            literal_meaning = glosses[0]
        else:
            literal_meaning = word
        
        # Parse morphological code for Greek if available
        grammar_description = 'Greek word'
        if morph_code and pos:
            parsed_morph = self._parse_morphological_code(morph_code, pos)
            if parsed_morph and parsed_morph != "Hebrew word":
                grammar_description = parsed_morph
        
        # Extract structured morphological features for Greek
        morphological_features = self._extract_morphological_features(morph_code, pos, lemma, word)
        
        result = {
            'transliteration': word,  # Simplified for now - could be enhanced
            'literal': literal_meaning,
            'grammar': grammar_description
        }
        
        # Add morphological features (only include fields that have values)
        result.update(morphological_features)
        
        return result
    
    def _transliterate_hebrew(self, word: str) -> str:
        """Basic Hebrew transliteration."""
        result = ''
        for char in word:
            result += self.hebrew_transliteration.get(char, char)
        return result
    
    def _transliterate_hebrew_with_vowels(self, word: str) -> str:
        """Enhanced Hebrew transliteration preserving vowel points."""
        # Enhanced vowel point mappings
        vowel_mappings = {
            # Vowel points
            '\u05B0': 'ə',  # Sheva
            '\u05B1': 'e',  # Hataf Segol  
            '\u05B2': 'a',  # Hataf Patah
            '\u05B3': 'o',  # Hataf Qamats
            '\u05B4': 'i',  # Hiriq
            '\u05B5': 'e',  # Tsere
            '\u05B6': 'e',  # Segol
            '\u05B7': 'a',  # Patah
            '\u05B8': 'ā',  # Qamats
            '\u05B9': 'ō',  # Holam
            '\u05BA': 'o',  # Holam Haser
            '\u05BB': 'u',  # Qubuts
            '\u05BC': '',   # Dagesh (doubling)
            '\u05BD': '',   # Meteg
            '\u05BE': '-',  # Maqaf
            '\u05C0': '|',  # Paseq
            '\u05C3': ':',  # Sof Pasuq
        }
        
        # Check for specific known transliterations first
        known_transliterations = {
            'בְּרֵאשִׁית': 'bə-rēʾšít',
            'בָּרָא': 'bārāʾ',
            'אֱלֹהִים': 'ʾĕlōhím', 
            'אֵת': 'ʾēt',
            'הַשָּׁמַיִם': 'ha-šāmayim',
            'וְאֵת': 'wə-ʾēt',
            'הָאָרֶץ': 'hā-ʾāreṣ'
        }
        
        # Remove slashes and clean the word
        clean_word = word.replace('/', '')
        
        if clean_word in known_transliterations:
            return known_transliterations[clean_word]
        
        # General transliteration with vowel points
        result = ''
        for char in clean_word:
            if char in vowel_mappings:
                result += vowel_mappings[char]
            elif char in self.hebrew_transliteration:
                result += self.hebrew_transliteration[char]
            else:
                result += char
                
        return result
    
    def _get_detailed_grammatical_analysis(self, word: str, clean_word: str, lemma: str, root_meaning: str, morph_code: str = '', pos: str = '') -> Dict:
        """Provide detailed grammatical analysis with comprehensive explanations."""
        grammar_notes = []
        prefix_parts = []
        literal_meaning = ''
        
        # Parse morphological code if available
        morphological_description = ""
        if morph_code and pos:
            morphological_description = self._parse_morphological_code(morph_code, pos)
            if morphological_description and morphological_description != "Hebrew word":
                grammar_notes.append(morphological_description)
        
        # Special cases with detailed explanations
        if lemma in ['853']:
            grammar_notes.append('Untranslated particle marking a definite direct object')
            return {'grammar_notes': grammar_notes, 'prefix_parts': prefix_parts, 'literal_meaning': literal_meaning}
        
        # Handle conjunction + object marker combinations (e.g., וְאֵת)
        if lemma.startswith('c/853') or lemma.startswith('c853'):
            grammar_notes.append('**ו** conjunction + **אֵת** object marker. Repeats the direct-object marker before the second object')
            prefix_parts.append('and')
            if not literal_meaning:
                literal_meaning = 'and (object marker)'
        
        # Handle other conjunction prefixes (with or without slash)
        elif (lemma.startswith('c/') and len(lemma) > 2) or (lemma.startswith('c') and len(lemma) > 1 and not lemma.startswith('c/853')):
            grammar_notes.append('**ו** conjunction prefixed to the next word')
            prefix_parts.append('and')
        
        # Handle definite article prefixes (with or without slash)
        elif lemma.startswith('d/') or (lemma.startswith('d') and len(lemma) > 1):
            root_num = lemma[2:] if lemma.startswith('d/') else lemma[1:]  # Extract number after 'd' or 'd/'
            if root_meaning:
                grammar_notes.append(f'**ה** definite article + noun "{root_meaning}"')
            else:
                grammar_notes.append('**ה** definite article + noun')
            prefix_parts.append('the')
        elif clean_word.startswith('ה') and not lemma.startswith('d'):
            grammar_notes.append('**ה** definite article')
            prefix_parts.append('the')
        
        # Handle preposition prefixes with detailed explanations (with or without slash)
        elif lemma.startswith('b/') or (lemma.startswith('b') and len(lemma) > 1):
            root_num = lemma[2:] if lemma.startswith('b/') else lemma[1:]  # Extract number after 'b' or 'b/'
            if root_meaning:
                grammar_notes.append(f'**ב** preposition "in/at" + **{root_meaning}** "{root_meaning}"; no article in Hebrew, but English needs "the"')
            else:
                grammar_notes.append('**ב** preposition "in/at"')
            prefix_parts.append('in')
        
        # Handle ל preposition (to/for) - new case for Adam
        elif lemma.startswith('l/') or (lemma.startswith('l') and len(lemma) > 1):
            root_num = lemma[2:] if lemma.startswith('l/') else lemma[1:]  # Extract number after 'l' or 'l/'
            if root_meaning:
                grammar_notes.append(f'**ל** preposition "to/for" + **{root_meaning}** "{root_meaning}"')
            else:
                grammar_notes.append('**ל** preposition "to/for"')
            prefix_parts.append('to')
        elif clean_word.startswith('ב') and not root_meaning:
            grammar_notes.append('**ב** preposition "in/at"')
            prefix_parts.append('in')
        
        elif clean_word.startswith('ל'):
            grammar_notes.append('**ל** preposition "to/for"')
            prefix_parts.append('to')
        elif clean_word.startswith('מ'):
            grammar_notes.append('**מ** preposition "from/of"')  
            prefix_parts.append('from')
        elif clean_word.startswith('כ'):
            grammar_notes.append('**כ** preposition "like/as"')
            prefix_parts.append('like')
        
        # Specific word analyses with detailed morphological information including gender/number
        if lemma == '430':  # אֱלֹהִים
            grammar_notes.append('Noun, common, masculine, plural, absolute. Morphologically plural but takes singular verbs when referring to Israel\'s God ("plural of majesty")')
        elif lemma == '1254' or lemma == '1254 a':  # בָּרָא
            grammar_notes.append('Verb, Qal stem, perfect aspect, 3rd person masculine singular; subject is **אֱלֹהִים** that follows. Used only of divine creation in the Bible')
        elif lemma.endswith('8064'):  # שָּׁמַיִם (heavens)
            if lemma.startswith('d'):
                grammar_notes.append('**ה** definite article + noun, common, masculine, plural/dual, absolute "heavens, sky"')
            else:
                grammar_notes.append('Noun, common, masculine, plural/dual, absolute "heavens, sky"')
        elif lemma.endswith('776'):  # אָרֶץ (earth)
            if lemma.startswith('d'):
                grammar_notes.append('**ה** definite article + noun, common, feminine, singular, absolute "earth, land"')
            else:
                grammar_notes.append('Noun, common, feminine, singular, absolute "earth, land"')
        elif lemma.endswith('7225'):  # רֵאשִׁית (beginning)
            if lemma.startswith('b'):
                grammar_notes.append('**ב** preposition "in/at" + **רֵאשִׁית** noun, common, feminine, singular, construct "beginning, first part"; no article in Hebrew, but English needs "the"')
            else:
                grammar_notes.append('**רֵאשִׁית** noun, common, feminine, singular, construct "beginning, first part"')
        
        # If no specific analysis found, provide generic analysis
        if not grammar_notes:
            if prefix_parts:
                grammar_notes.append(f'Hebrew word with {", ".join(prefix_parts)} prefix')
            else:
                grammar_notes.append('Hebrew word')
        
        return {
            'grammar_notes': grammar_notes,
            'prefix_parts': prefix_parts, 
            'literal_meaning': literal_meaning
        }
    
    def _parse_morphological_code(self, morph_code: str, pos: str) -> str:
        """Parse Hebrew morphological codes from OSHB data into readable descriptions."""
        if not morph_code:
            return ""
        
        # Hebrew morphological parsing based on OSHB codes
        # Format: [Part of Speech][Gender][Number][State][Person][Stem][Aspect]
        
        parts = []
        
        # Parse based on POS (Part of Speech)
        if pos.startswith('N'):  # Noun
            parts.append("Noun")
            if 'c' in pos.lower():
                parts.append("common")
            elif 'p' in pos.lower():
                parts.append("proper")
                
            # Gender
            if 'm' in morph_code:
                parts.append("masculine")
            elif 'f' in morph_code:
                parts.append("feminine")
                
            # Number
            if 's' in morph_code:
                parts.append("singular")
            elif 'p' in morph_code:
                parts.append("plural")
            elif 'd' in morph_code:
                parts.append("dual")
                
            # State
            if 'a' in morph_code:
                parts.append("absolute")
            elif 'c' in morph_code:
                parts.append("construct")
            elif 'd' in morph_code:
                parts.append("determined")
                
        elif pos.startswith('V'):  # Verb
            parts.append("Verb")
            
            # Stem
            if 'q' in morph_code.lower():
                parts.append("Qal stem")
            elif 'n' in morph_code.lower():
                parts.append("Niphal stem")
            elif 'p' in morph_code.lower():
                parts.append("Piel stem")
            elif 'h' in morph_code.lower():
                parts.append("Hiphil stem")
                
            # Aspect
            if morph_code and len(morph_code) > 1:
                if morph_code[1] == 'q':
                    parts.append("perfect aspect")
                elif morph_code[1] == 'y':
                    parts.append("imperfect aspect")
                elif morph_code[1] == 'w':
                    parts.append("waw consecutive")
                elif morph_code[1] == 'i':
                    parts.append("imperative")
                elif morph_code[1] == 'p':
                    parts.append("participle")
                    
            # Person, gender, number for verbs
            if len(morph_code) >= 3:
                person_code = morph_code[2:5] if len(morph_code) >= 5 else morph_code[2:]
                if '3' in person_code:
                    parts.append("3rd person")
                elif '2' in person_code:
                    parts.append("2nd person")
                elif '1' in person_code:
                    parts.append("1st person")
                    
                if 'm' in person_code:
                    parts.append("masculine")
                elif 'f' in person_code:
                    parts.append("feminine")
                    
                if 's' in person_code:
                    parts.append("singular")
                elif 'p' in person_code:
                    parts.append("plural")
                    
        elif pos.startswith('P'):  # Pronoun
            parts.append("Pronoun")
            
        elif pos.startswith('A'):  # Adjective
            parts.append("Adjective")
            
        elif pos.startswith('R'):  # Adverb
            parts.append("Adverb")
            
        elif pos.startswith('C'):  # Conjunction
            parts.append("Conjunction")
            
        elif pos.startswith('D'):  # Determiner/Article
            parts.append("Determiner/Article")
            
        elif pos.startswith('T'):  # Particle
            parts.append("Particle")
            
        return ", ".join(parts) if parts else "Hebrew word"
    
    def _extract_morphological_features(self, morph_code: str, pos: str, lemma: str, word: str) -> Dict:
        """Extract individual morphological features into structured fields."""
        features = {}
        
        # Part of speech (handle Hebrew prefix 'H' in OSHB codes)
        pos_clean = pos.lstrip('H')  # Remove Hebrew prefix
        
        # Handle compound POS codes like 'Rd/Ncmsa' where the main word is after the slash
        main_pos = pos_clean
        if '/' in pos_clean:
            # For compound forms, use the part after the slash as the main POS
            main_pos = pos_clean.split('/')[-1]
        
        if main_pos.startswith('N'):
            features['part_of_speech'] = 'noun'
            # Check for proper noun (Np) vs common noun (Nc)
            if main_pos.startswith('Np'):
                features['noun_type'] = 'proper'
                # Add semantic type detection for proper nouns
                semantic_type = self._detect_semantic_type(lemma, word)
                if semantic_type == 'person':
                    features['is_name'] = True
                elif semantic_type == 'location':
                    features['is_place'] = True
            elif main_pos.startswith('Nc'):
                features['noun_type'] = 'common'
                # Check if this common noun is actually a known name (e.g., Adam marked as common)
                semantic_type = self._detect_semantic_type(lemma, word)
                if semantic_type == 'person':
                    features['is_name'] = True
                elif semantic_type == 'location':
                    features['is_place'] = True
        elif pos_clean.startswith('V'):
            features['part_of_speech'] = 'verb'
        elif pos_clean.startswith('A'):
            features['part_of_speech'] = 'adjective'
        elif pos_clean.startswith('P'):
            features['part_of_speech'] = 'pronoun'
        elif pos_clean.startswith('R'):
            features['part_of_speech'] = 'adverb'
        elif pos_clean.startswith('C'):
            features['part_of_speech'] = 'conjunction'
        elif pos_clean.startswith('T'):
            features['part_of_speech'] = 'particle'
        
        # Gender
        if morph_code:
            if 'm' in morph_code:
                features['gender'] = 'masculine'
            elif 'f' in morph_code:
                features['gender'] = 'feminine'
            
            # Number
            if 's' in morph_code:
                features['number'] = 'singular'
            elif 'p' in morph_code:
                features['number'] = 'plural'
            elif 'd' in morph_code:
                features['number'] = 'dual'
            
            # State (for nouns)
            if features.get('part_of_speech') == 'noun':
                if 'a' in morph_code:
                    features['state'] = 'absolute'
                elif 'c' in morph_code:
                    features['state'] = 'construct'
                elif 'd' in morph_code:
                    features['state'] = 'determined'
            
            # Verb features
            if features.get('part_of_speech') == 'verb':
                # Stem
                if 'q' in morph_code.lower():
                    features['stem'] = 'qal'
                elif 'n' in morph_code.lower():
                    features['stem'] = 'niphal'
                elif 'p' in morph_code.lower():
                    features['stem'] = 'piel'
                elif 'h' in morph_code.lower():
                    features['stem'] = 'hiphil'
                
                # Aspect
                if len(morph_code) > 1:
                    if morph_code[1] == 'q':
                        features['aspect'] = 'perfect'
                    elif morph_code[1] == 'y':
                        features['aspect'] = 'imperfect'
                    elif morph_code[1] == 'w':
                        features['aspect'] = 'waw_consecutive'
                    elif morph_code[1] == 'i':
                        features['aspect'] = 'imperative'
                    elif morph_code[1] == 'p':
                        features['aspect'] = 'participle'
                
                # Person (for verbs)
                if len(morph_code) >= 3:
                    person_code = morph_code[2:5] if len(morph_code) >= 5 else morph_code[2:]
                    if '3' in person_code:
                        features['person'] = '3rd'
                    elif '2' in person_code:
                        features['person'] = '2nd'
                    elif '1' in person_code:
                        features['person'] = '1st'
        
        return features
    
    def _detect_semantic_type(self, lemma: str, word: str) -> str:
        """Detect semantic type for proper nouns (person, location, etc.)."""
        # Comprehensive person names from the Hebrew Bible
        person_names = {
            # Divine names
            'יהוה', 'אלהים', 'אדני', 'אל', 'שדי', 
            
            # Patriarchs and Matriarchs
            'אדם', 'חוה', 'נח', 'אברהם', 'שרה', 'יצחק', 'רבקה', 'יעקב', 'רחל', 'לאה',
            'יוסף', 'יהודה', 'בנימין', 'ראובן', 'שמעון', 'לוי', 'דן', 'נפתלי', 'גד', 'אשר', 'יששכר', 'זבולן',
            
            # Moses and Aaron era
            'משה', 'אהרן', 'מרים', 'יתרו', 'ציפורה', 'יהושע', 'כלב', 'פינחס', 'אלעזר', 'איתמר',
            
            # Judges period
            'דבורה', 'ברק', 'גדעון', 'יפתח', 'שמשון', 'דלילה', 'עלי', 'שמואל',
            
            # Kings and royal family
            'שאול', 'דוד', 'שלמה', 'רחבעם', 'יהושפט', 'חזקיהו', 'יאשיהו', 'בת־שבע', 'אבשלום', 'יונתן',
            
            # Prophets
            'נתן', 'אליה', 'אלישע', 'ישעיהו', 'ירמיהו', 'יחזקאל', 'דניאל', 'הושע', 'יואל', 'עמוס', 'עובדיה',
            'יונה', 'מיכה', 'נחום', 'חבקוק', 'צפניה', 'חגי', 'זכריה', 'מלאכי',
            
            # Post-exile leaders
            'עזרא', 'נחמיה', 'זרובבל', 'יהושע', 'מרדכי', 'אסתר', 'המן',
            
            # Women
            'שרה', 'רבקה', 'רחל', 'לאה', 'מרים', 'צפורה', 'רחב', 'רות', 'נעמי', 'חנה', 'אביגיל', 'תמר', 'אסתר',
            
            # Foreign rulers and officials
            'פרעה', 'פוטיפר', 'נבוכדנאצר', 'בלשאצר', 'דריוש', 'כורש', 'ארתחשסתא', 'אחשורוש'
        }
        
        # Comprehensive location names from the Hebrew Bible
        location_names = {
            # Major cities
            'ירושלם', 'חברון', 'בית־לחם', 'שכם', 'צרפת', 'גזר', 'מגדו', 'חצור', 'דן', 'באר־שבע',
            'יריחו', 'עי', 'גבעון', 'רמה', 'בית־אל', 'שילה', 'גלגל', 'מצפה', 'רמות־גלעד', 'תרצה',
            
            # Countries and regions  
            'מצרים', 'כנען', 'אדום', 'מואב', 'עמון', 'ארם', 'פלשת', 'אשור', 'בבל', 'פרס', 'כושׁ', 'פוט',
            'גלעד', 'בשן', 'חרמון', 'לבנון', 'סיני', 'חורב', 'אראראט', 'פדן־ארם', 'מסופוטמיה',
            
            # Geographical features
            'ירדן', 'כנרת', 'המלח', 'כרמל', 'תבור', 'עיבל', 'גריזים', 'ציון', 'מוריה', 'אליונים',
            'נגב', 'ערבה', 'שפלה', 'גלבוע', 'כרמל', 'לבנון', 'אנטילבנון',
            
            # NT locations (Greek names)
            'נצרת', 'כפר־נחום', 'בית־צידה', 'כורזין', 'מגדן', 'קיסריה', 'יפו', 'לוד', 'רמלה'
        }
        
        # Clean word for comparison (remove vowel points and diacritics)
        clean_word = re.sub(r'[\u0591-\u05C7]', '', word)
        
        # Direct lookup in name lists
        if clean_word in person_names:
            return 'person'
        elif clean_word in location_names:
            return 'location'
        
        # Additional pattern-based detection for proper nouns
        # This can catch names not in our explicit lists
        
        # Check if it's a proper noun based on morphological patterns
        # Hebrew proper nouns often have specific patterns
        
        # Names ending in common Hebrew name suffixes
        person_suffixes = ['אל', 'יה', 'יהו']  # God-based theophoric elements
        location_suffixes = ['ון', 'ים', 'ה']  # Common place name endings
        
        for suffix in person_suffixes:
            if clean_word.endswith(suffix) and len(clean_word) > len(suffix):
                return 'person'
        
        # Biblical place names often end with these patterns
        for suffix in location_suffixes:
            if clean_word.endswith(suffix) and len(clean_word) > len(suffix):
                # Additional heuristics to distinguish from common nouns
                if len(clean_word) >= 4:  # Minimum length for place name
                    return 'location'
        
        # Check Strong's numbers for semantic type detection
        # This handles cases where names are marked as common nouns or compound lemmas
        strong_person_numbers = {
            '120': 'person',  # Adam (אדם)
            '85': 'person',   # Abraham (אברהם)
            '3068': 'person', # LORD/Yahweh (יהוה)
            '8283': 'person', # Sarah (שרה)
            '3327': 'person', # Isaac (יצחק)
            '3290': 'person', # Jacob (יעקב)
            '3130': 'person', # Joseph (יוסף)
            '4872': 'person', # Moses (משה)
            '175': 'person',  # Aaron (אהרן)
            '1732': 'person', # David (דוד)
            '8010': 'person', # Solomon (שלמה)
        }
        
        strong_place_numbers = {
            '2275': 'location', # Hebron (חברון)
            '3389': 'location', # Jerusalem (ירושלם)
            '1035': 'location', # Bethlehem (בית־לחם)
            '4714': 'location', # Egypt (מצרים)
            '3667': 'location', # Canaan (כנען)
        }
        
        # Extract Strong's number from lemma (handles both simple and compound forms)
        strong_number = None
        if lemma:
            # Handle compound lemmas like 'l/120' or 'c/853'
            compound_match = re.search(r'[a-z]*/(\d+)', lemma)
            if compound_match:
                strong_number = compound_match.group(1)
            else:
                # Handle simple Strong's numbers like '120' or '430'
                simple_match = re.search(r'(\d+)', lemma)
                if simple_match:
                    strong_number = simple_match.group(1)
        
        if strong_number:
            if strong_number in strong_person_numbers:
                return strong_person_numbers[strong_number]
            elif strong_number in strong_place_numbers:
                return strong_place_numbers[strong_number]
        
        return None


class MorphologicalLexiconAligner:
    """
    Aligner that uses morphological data as lexicon source for semantic alignment.
    """
    
    def __init__(self):
        self.lexicon = MorphologicalLexiconLookup()
        self.confidence_threshold = 0.6  # Reasonable threshold
        self.morphological_analyzer = MorphologicalAnalyzer()
        
    def get_morphological_breakdown(self, word: str, language: str, lemma: str = '', glosses: List[str] = None, matched_gloss: str = '', morph_code: str = '', pos: str = '') -> Dict:
        """Get detailed morphological breakdown for a word."""
        return self.morphological_analyzer.get_breakdown(word, language, lemma, glosses, matched_gloss, morph_code, pos)
        
    def align_verse(self, source_words: List[str], target_words: List[str],
                   source_lang: str = 'hebrew', target_lang: str = 'english', **kwargs) -> List[Dict]:
        """
        Align words using morphological lexicon data.
        """
        logger.debug(f"Morphological lexicon alignment: {len(source_words)} {source_lang} → {len(target_words)} {target_lang}")
        
        alignments = []
        target_words_lower = [w.lower().strip('.,!?;:') for w in target_words]
        
        for src_idx, src_word in enumerate(source_words):
            # Look up lemma information
            lemma_info = self.lexicon.lookup_lemma(src_word, source_lang)
            
            if not lemma_info:
                logger.debug(f"No lemma found for {src_word}")
                continue
            
            lemma = lemma_info.get('lemma', '')
            pos = lemma_info.get('pos', '')
            
            # Get English glosses for this lemma
            glosses = self.lexicon.get_english_glosses(lemma, pos, source_lang)
            
            # Special handling for proper nouns - they don't have glosses but need name mappings
            is_proper_noun = pos.lstrip('H').startswith('Np')
            if not glosses and not is_proper_noun:
                logger.debug(f"No English glosses for lemma {lemma}")
                continue
            elif is_proper_noun and not glosses:
                # For proper nouns, try to get name equivalents
                glosses = self._get_proper_noun_equivalents(src_word, lemma)
            
            # Find matches in target text
            best_matches = self._find_lexicon_matches(
                src_word, src_idx, lemma, glosses, target_words, target_words_lower, pos
            )
            
            # Add alignments above threshold
            for tgt_idx, confidence, matched_gloss in best_matches:
                if confidence >= self.confidence_threshold:
                    # Get morphological breakdown with lexicon data and matched gloss
                    breakdown = self.get_morphological_breakdown(src_word, source_lang, lemma, glosses, matched_gloss, pos, pos)
                    
                    # Handle object markers specially
                    is_object_marker = (lemma in ['853', 'c853'] or 
                                      breakdown.get('literal', '').startswith('(object marker)'))
                    
                    alignment = {
                        'source_index': src_idx,
                        'target_index': tgt_idx,
                        'source_word': src_word,
                        'target_word': target_words[tgt_idx],
                        'confidence': confidence,
                        'method': 'morphological_lexicon',
                        'lexicon_features': {
                            'lemma': lemma,
                            'pos': pos,
                            'matched_gloss': matched_gloss,
                            'all_glosses': glosses,
                            'is_object_marker': is_object_marker,
                            'transliteration': breakdown.get('transliteration', ''),
                            'literal_meaning': breakdown.get('literal', ''),
                            'grammatical_notes': breakdown.get('grammar', ''),
                            # Include all morphological features from breakdown
                            **{k: v for k, v in breakdown.items() if k not in ['transliteration', 'literal', 'grammar']}
                        }
                    }
                    alignments.append(alignment)
            
            # Special handling for object markers - they should be marked even if not aligned
            if lemma in ['853', 'c853'] or pos == 'particle':
                breakdown = self.get_morphological_breakdown(src_word, source_lang, lemma, glosses, '', pos, pos)
                if breakdown.get('literal', '').find('object marker') >= 0:
                    # Add a special object marker alignment
                    object_marker_alignment = {
                        'source_index': src_idx,
                        'target_index': -1,  # No target mapping
                        'source_word': src_word,
                        'target_word': '(untranslated)',
                        'confidence': 1.0,  # High confidence for known object markers
                        'method': 'morphological_lexicon',
                        'lexicon_features': {
                            'lemma': lemma,
                            'pos': 'object_marker',
                            'matched_gloss': 'object marker',
                            'all_glosses': ['object marker'],
                            'is_object_marker': True,
                            'transliteration': breakdown.get('transliteration', ''),
                            'literal_meaning': breakdown.get('literal', ''),
                            'grammatical_notes': breakdown.get('grammar', ''),
                            # Include all morphological features from breakdown
                            **{k: v for k, v in breakdown.items() if k not in ['transliteration', 'literal', 'grammar']}
                        }
                    }
                    alignments.append(object_marker_alignment)
        
        logger.debug(f"Morphological lexicon alignment found {len(alignments)} alignments")
        return alignments
    
    def _find_lexicon_matches(self, src_word: str, src_idx: int, lemma: str, glosses: List[str],
                             target_words: List[str], target_words_lower: List[str], pos: str) -> List[Tuple[int, float, str]]:
        """Find matches between lemma glosses and target words."""
        matches = []
        
        for tgt_idx, (target_word, target_lower) in enumerate(zip(target_words, target_words_lower)):
            best_confidence = 0.0
            best_gloss = ""
            
            for gloss in glosses:
                gloss_lower = gloss.lower()
                confidence = 0.0
                
                # Exact match
                if target_lower == gloss_lower:
                    confidence = 0.9
                # Partial match
                elif gloss_lower in target_lower or target_lower in gloss_lower:
                    if len(gloss_lower) >= 3 and len(target_lower) >= 3:
                        confidence = 0.8
                # Stem match for verbs
                elif pos in ['verb', 'V-'] and self._verb_stem_match(target_lower, gloss_lower):
                    confidence = 0.7
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_gloss = gloss
            
            if best_confidence > 0:
                matches.append((tgt_idx, best_confidence, best_gloss))
        
        # Sort by confidence and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:2]  # Limit to top 2 matches
    
    def _get_proper_noun_equivalents(self, word: str, lemma: str) -> List[str]:
        """Get English name equivalents for Hebrew/Greek proper nouns."""
        # Clean the word for lookup
        clean_word = re.sub(r'[\u0591-\u05C7]', '', word)
        
        # Comprehensive Hebrew-English name mappings
        hebrew_name_mappings = {
            # Divine names
            'יהוה': ['LORD', 'Yahweh', 'Jehovah'],
            'אל': ['El', 'God'],
            'אלהים': ['God', 'Elohim'],
            'אדני': ['Lord', 'Adonai'],
            'שדי': ['Almighty', 'Shaddai'],
            
            # Patriarchs and Matriarchs
            'אברהם': ['Abraham'],
            'שרה': ['Sarah', 'Sarai'],
            'יצחק': ['Isaac'],
            'רבקה': ['Rebecca', 'Rebekah'],
            'יעקב': ['Jacob'],
            'רחל': ['Rachel'],
            'לאה': ['Leah'],
            'יוסף': ['Joseph'],
            'יהודה': ['Judah'],
            'בנימין': ['Benjamin'],
            'ראובן': ['Reuben'],
            'שמעון': ['Simeon'],
            'לוי': ['Levi'],
            'דן': ['Dan'],
            'נפתלי': ['Naphtali'],
            'גד': ['Gad'],
            'אשר': ['Asher'],
            'יששכר': ['Issachar'],
            'זבולן': ['Zebulun'],
            
            # Moses era
            'משה': ['Moses'],
            'אהרן': ['Aaron'],
            'מרים': ['Miriam'],
            'יהושע': ['Joshua'],
            'כלב': ['Caleb'],
            
            # Judges and early leaders
            'שמואל': ['Samuel'],
            'שאול': ['Saul'],
            'דוד': ['David'],
            'שלמה': ['Solomon'],
            'נתן': ['Nathan'],
            
            # Prophets
            'אליה': ['Elijah'],
            'אלישע': ['Elisha'],
            'ישעיהו': ['Isaiah'],
            'ירמיהו': ['Jeremiah'],
            'יחזקאל': ['Ezekiel'],
            'דניאל': ['Daniel'],
            
            # Women
            'חוה': ['Eve'],
            'רות': ['Ruth'],
            'נעמי': ['Naomi'],
            'חנה': ['Hannah'],
            'אסתר': ['Esther'],
            'בת־שבע': ['Bathsheba'],
            
            # Places - Major cities
            'ירושלם': ['Jerusalem'],
            'חברון': ['Hebron'],
            'בית־לחם': ['Bethlehem'],
            'שכם': ['Shechem'],
            'יריחו': ['Jericho'],
            'בית־אל': ['Bethel'],
            'שילה': ['Shiloh'],
            'גבעון': ['Gibeon'],
            'דן': ['Dan'],
            'באר־שבע': ['Beersheba'],
            
            # Places - Countries and regions
            'מצרים': ['Egypt'],
            'כנען': ['Canaan'],
            'אדום': ['Edom'],
            'מואב': ['Moab'],
            'עמון': ['Ammon'],
            'ארם': ['Aram', 'Syria'],
            'פלשת': ['Philistia'],
            'אשור': ['Assyria'],
            'בבל': ['Babylon'],
            'פרס': ['Persia'],
            'גלעד': ['Gilead'],
            'בשן': ['Bashan'],
            
            # Places - Geographical features
            'ירדן': ['Jordan'],
            'כנרת': ['Galilee', 'Chinnereth'],
            'כרמל': ['Carmel'],
            'תבור': ['Tabor'],
            'חרמון': ['Hermon'],
            'לבנון': ['Lebanon'],
            'ציון': ['Zion'],
            'מוריה': ['Moriah'],
            'סיני': ['Sinai'],
            'חורב': ['Horeb']
        }
        
        # Try direct lookup first
        if clean_word in hebrew_name_mappings:
            return hebrew_name_mappings[clean_word]
        
        # Try Strong's number-based lookup for additional mappings
        strong_number_mappings = {
            '85': ['Abraham'],  # אברהם
            '120': ['Adam'],  # אדם (can be common noun "man" or proper noun "Adam")
            '3068': ['LORD', 'Yahweh', 'Jehovah'],  # יהוה
            '8283': ['Sarah'],  # שרה
            '3327': ['Isaac'],  # יצחק
            '3290': ['Jacob'],  # יעקב
            '3130': ['Joseph'],  # יוסף
            '4872': ['Moses'],  # משה
            '175': ['Aaron'],  # אהרן
            '1732': ['David'],  # דוד
            '8010': ['Solomon'],  # שלמה
        }
        
        if lemma in strong_number_mappings:
            return strong_number_mappings[lemma]
        
        # Handle compound lemmas like 'l/120' (preposition + Strong's number)
        # Extract the core Strong's number from compound forms
        compound_match = re.search(r'[a-z]*/(\d+)', lemma)
        if compound_match:
            core_number = compound_match.group(1)
            if core_number in strong_number_mappings:
                return strong_number_mappings[core_number]
        
        # If no direct mapping, return empty list (will be handled by the alignment process)
        return []
    
    def _verb_stem_match(self, word1: str, word2: str) -> bool:
        """Check if words share verb stems (simple heuristic)."""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Simple stem matching
        return word1[:3] == word2[:3] or word1[:4] == word2[:4]