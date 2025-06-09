"""
Biblical phrase detection and analysis.

Identifies multi-word expressions, idioms, and technical terms
that should be aligned as units rather than word-by-word.
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)


class BiblicalPhraseDetector:
    """Detect and analyze biblical phrases and idioms."""
    
    def __init__(self, language: str, min_frequency: int = 3):
        self.language = language
        self.min_frequency = min_frequency
        
        # Known biblical phrases by category
        self.hebrew_phrases = {
            'divine_titles': [
                ('יְהוָה צְבָאוֹת', 'Lord of Hosts'),
                ('אֵל עֶלְיוֹן', 'God Most High'),
                ('אֵל שַׁדַּי', 'God Almighty'),
                ('עַתִּיק יוֹמִין', 'Ancient of Days'),
                ('קְדוֹשׁ יִשְׂרָאֵל', 'Holy One of Israel')
            ],
            'idioms': [
                ('נָשָׂא עֵינָיו', 'lifted up his eyes'),
                ('חָזַק לֵב', 'hardened heart'),
                ('בֵּית אָבִי', 'my father\'s house'),
                ('כִּי טוֹב', 'that it was good'),
                ('וַיְהִי כֵן', 'and it was so')
            ],
            'technical_terms': [
                ('קֹדֶשׁ הַקֳּדָשִׁים', 'Holy of Holies'),
                ('יוֹם כִּפֻּרִים', 'Day of Atonement'),
                ('בְּרִית עוֹלָם', 'everlasting covenant'),
                ('עֹלָה תָמִיד', 'continual burnt offering')
            ],
            'narrative_markers': [
                ('וַיְהִי', 'and it came to pass'),
                ('הִנֵּה', 'behold'),
                ('וְעַתָּה', 'and now'),
                ('לָכֵן', 'therefore')
            ]
        }
        
        self.greek_phrases = {
            'divine_titles': [
                ('κύριος τῶν δυνάμεων', 'Lord of Hosts'),
                ('υἱὸς τοῦ ἀνθρώπου', 'Son of Man'),
                ('υἱὸς τοῦ θεοῦ', 'Son of God'),
                ('ἅγιος τοῦ θεοῦ', 'Holy One of God')
            ],
            'kingdom_terms': [
                ('βασιλεία τοῦ θεοῦ', 'kingdom of God'),
                ('βασιλεία τῶν οὐρανῶν', 'kingdom of heaven'),
                ('ζωὴ αἰώνιος', 'eternal life')
            ],
            'discourse_markers': [
                ('ἀμὴν λέγω ὑμῖν', 'truly I say to you'),
                ('διὰ τοῦτο', 'because of this'),
                ('ἐν ἐκείναις ταῖς ἡμέραις', 'in those days')
            ]
        }
        
        self.english_phrases = {
            'archaic_forms': [
                ('it came to pass', 'narrative_marker'),
                ('thus saith the Lord', 'prophetic_formula'),
                ('verily I say unto you', 'emphasis_formula')
            ],
            'theological_terms': [
                ('born again', 'spiritual_rebirth'),
                ('kingdom of God', 'divine_rule'),
                ('Son of Man', 'messianic_title'),
                ('Holy Spirit', 'divine_person')
            ]
        }
        
        # Learned phrases
        self.discovered_phrases = defaultdict(Counter)
        self.phrase_contexts = defaultdict(list)
        
    def extract_phrases(self, corpus: List[Dict], 
                       min_length: int = 2, 
                       max_length: int = 5) -> List[Tuple[str, int]]:
        """Extract recurring phrases from corpus."""
        phrase_counts = Counter()
        
        for verse_data in corpus:
            # Get text based on language
            if self.language == 'hebrew':
                words = verse_data.get('hebrew_words', [])
                text_words = [w.get('text', '') for w in words]
            elif self.language == 'greek':
                words = verse_data.get('greek_words', [])
                text_words = [w.get('text', '') for w in words]
            else:  # English or other
                text = self._get_translation_text(verse_data)
                text_words = text.lower().split() if text else []
            
            # Extract n-grams
            for n in range(min_length, min(max_length + 1, len(text_words) + 1)):
                for i in range(len(text_words) - n + 1):
                    phrase = ' '.join(text_words[i:i+n])
                    phrase_counts[phrase] += 1
                    
                    # Store context
                    context = {
                        'verse_id': verse_data.get('verse_id', ''),
                        'before': ' '.join(text_words[max(0, i-3):i]),
                        'after': ' '.join(text_words[i+n:min(len(text_words), i+n+3)])
                    }
                    self.phrase_contexts[phrase].append(context)
        
        # Filter by frequency
        frequent_phrases = [(phrase, count) for phrase, count in phrase_counts.items()
                           if count >= self.min_frequency]
        
        # Remove subphrases of longer phrases
        filtered_phrases = self._filter_subphrases(frequent_phrases)
        
        return filtered_phrases
    
    def _filter_subphrases(self, phrases: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Remove phrases that are subparts of longer frequent phrases."""
        # Sort by length (longest first)
        sorted_phrases = sorted(phrases, key=lambda x: len(x[0].split()), reverse=True)
        
        kept_phrases = []
        removed = set()
        
        for phrase, count in sorted_phrases:
            if phrase not in removed:
                kept_phrases.append((phrase, count))
                
                # Mark subphrases for removal
                words = phrase.split()
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        subphrase = ' '.join(words[i:j])
                        if subphrase != phrase:
                            removed.add(subphrase)
        
        return kept_phrases
    
    def classify_phrase(self, phrase: str, contexts: List[Dict]) -> Dict[str, any]:
        """Classify a phrase based on its usage patterns."""
        classification = {
            'phrase': phrase,
            'frequency': len(contexts),
            'type': 'unknown',
            'confidence': 0.0,
            'semantic_role': None
        }
        
        # Check known phrase categories
        if self.language == 'hebrew':
            for category, known_phrases in self.hebrew_phrases.items():
                for known_phrase, translation in known_phrases:
                    if phrase == known_phrase or phrase in known_phrase:
                        classification['type'] = category
                        classification['translation'] = translation
                        classification['confidence'] = 1.0
                        return classification
                        
        elif self.language == 'greek':
            for category, known_phrases in self.greek_phrases.items():
                for known_phrase, translation in known_phrases:
                    if phrase == known_phrase or phrase in known_phrase:
                        classification['type'] = category
                        classification['translation'] = translation
                        classification['confidence'] = 1.0
                        return classification
        
        # Analyze usage patterns for unknown phrases
        classification.update(self._analyze_phrase_pattern(phrase, contexts))
        
        return classification
    
    def _analyze_phrase_pattern(self, phrase: str, contexts: List[Dict]) -> Dict[str, any]:
        """Analyze usage pattern of a phrase."""
        pattern_info = {
            'positional_preference': self._analyze_position(contexts),
            'co_occurrence': self._analyze_cooccurrence(contexts),
            'syntactic_pattern': self._analyze_syntax(phrase)
        }
        
        # Heuristic classification based on patterns
        if pattern_info['positional_preference'] == 'initial':
            pattern_info['type'] = 'discourse_marker'
            pattern_info['confidence'] = 0.7
        elif self._is_construct_chain(phrase):
            pattern_info['type'] = 'construct_phrase'
            pattern_info['confidence'] = 0.8
        elif self._contains_divine_name(phrase):
            pattern_info['type'] = 'theological_phrase'
            pattern_info['confidence'] = 0.8
        
        return pattern_info
    
    def _analyze_position(self, contexts: List[Dict]) -> str:
        """Analyze positional preferences of phrase."""
        positions = Counter()
        
        for ctx in contexts:
            if not ctx['before']:
                positions['initial'] += 1
            elif not ctx['after']:
                positions['final'] += 1
            else:
                positions['medial'] += 1
        
        if positions['initial'] > len(contexts) * 0.6:
            return 'initial'
        elif positions['final'] > len(contexts) * 0.6:
            return 'final'
        else:
            return 'medial'
    
    def _analyze_cooccurrence(self, contexts: List[Dict]) -> Dict[str, List[str]]:
        """Analyze what words commonly appear near the phrase."""
        before_words = Counter()
        after_words = Counter()
        
        for ctx in contexts:
            if ctx['before']:
                before_words.update(ctx['before'].split())
            if ctx['after']:
                after_words.update(ctx['after'].split())
        
        return {
            'common_before': [w for w, _ in before_words.most_common(5)],
            'common_after': [w for w, _ in after_words.most_common(5)]
        }
    
    def _analyze_syntax(self, phrase: str) -> str:
        """Basic syntactic analysis of phrase structure."""
        words = phrase.split()
        
        if self.language == 'hebrew':
            # Check for construct chains (סמיכות)
            if len(words) == 2:
                # Simplified check - real implementation would use morphology
                return 'construct_chain'
            elif any(w in ['יְהוָה', 'אֱלֹהִים', 'אֵל'] for w in words):
                return 'divine_name_phrase'
                
        elif self.language == 'greek':
            # Check for genitive constructions
            if 'τοῦ' in words or 'τῆς' in words or 'τῶν' in words:
                return 'genitive_phrase'
                
        return 'simple_phrase'
    
    def _is_construct_chain(self, phrase: str) -> bool:
        """Check if phrase is a Hebrew construct chain."""
        if self.language != 'hebrew':
            return False
            
        words = phrase.split()
        if len(words) == 2:
            # Simplified heuristic - would need morphological analysis
            return True
        return False
    
    def _contains_divine_name(self, phrase: str) -> bool:
        """Check if phrase contains a divine name."""
        divine_names = {
            'hebrew': ['יְהוָה', 'אֱלֹהִים', 'אֵל', 'אֲדֹנָי'],
            'greek': ['θεός', 'κύριος', 'χριστός'],
            'english': ['god', 'lord', 'christ', 'jesus']
        }
        
        names = divine_names.get(self.language, [])
        phrase_lower = phrase.lower()
        
        return any(name in phrase_lower for name in names)
    
    def align_phrases_for_translation(self, source_phrases: List[Dict], 
                                    target_text: str) -> List[Dict]:
        """Align discovered phrases with their translations."""
        alignments = []
        
        for phrase_info in source_phrases:
            phrase = phrase_info['phrase']
            
            # Look for known translations
            if 'translation' in phrase_info:
                if phrase_info['translation'].lower() in target_text.lower():
                    alignments.append({
                        'source_phrase': phrase,
                        'target_phrase': phrase_info['translation'],
                        'confidence': phrase_info['confidence'],
                        'type': phrase_info['type']
                    })
            else:
                # Try to find alignment using context and patterns
                potential_alignment = self._find_phrase_alignment(
                    phrase, phrase_info, target_text
                )
                if potential_alignment:
                    alignments.append(potential_alignment)
        
        return alignments
    
    def _find_phrase_alignment(self, source_phrase: str, 
                              phrase_info: Dict, 
                              target_text: str) -> Optional[Dict]:
        """Attempt to find phrase alignment using patterns."""
        # This is a simplified version - real implementation would use
        # statistical methods and learned patterns
        
        # For demonstration, return None
        # Real implementation would analyze parallel texts
        return None
    
    def _get_translation_text(self, verse_data: Dict) -> Optional[str]:
        """Get translation text from verse data."""
        translations = verse_data.get('translations', {})
        # Try common translation codes
        for code in ['ENG_KJV', 'eng_kjv', 'ENG_NIV', 'eng_niv']:
            if code in translations:
                return translations[code]
        # Return first available
        if translations:
            return next(iter(translations.values()))
        return None
    
    def generate_phrase_report(self, phrases: List[Tuple[str, int]], 
                             classifications: List[Dict]) -> str:
        """Generate human-readable phrase analysis report."""
        report = []
        report.append(f"Biblical Phrase Analysis - {self.language.title()}")
        report.append("=" * 60)
        report.append(f"Total phrases discovered: {len(phrases)}")
        report.append(f"Minimum frequency threshold: {self.min_frequency}")
        report.append("")
        
        # Group by type
        by_type = defaultdict(list)
        for classification in classifications:
            by_type[classification['type']].append(classification)
        
        for phrase_type, phrases_of_type in by_type.items():
            report.append(f"\n{phrase_type.replace('_', ' ').title()} ({len(phrases_of_type)} phrases):")
            report.append("-" * 40)
            
            # Sort by frequency
            sorted_phrases = sorted(phrases_of_type, 
                                  key=lambda x: x['frequency'], 
                                  reverse=True)
            
            for phrase_info in sorted_phrases[:10]:  # Top 10 per category
                report.append(f"  {phrase_info['phrase']}")
                report.append(f"    Frequency: {phrase_info['frequency']}")
                if 'translation' in phrase_info:
                    report.append(f"    Translation: {phrase_info['translation']}")
                report.append(f"    Confidence: {phrase_info['confidence']:.2f}")
                report.append("")
        
        return "\n".join(report)