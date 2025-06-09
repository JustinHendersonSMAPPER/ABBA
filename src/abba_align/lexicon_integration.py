"""
Strong's Concordance and lexicon integration for enhanced alignment.

Strong's provides:
- Word definitions and glosses
- Semantic ranges
- Usage contexts
- Root word connections
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class StrongsLexiconIntegration:
    """Integrate Strong's Concordance for semantic-aware alignment."""
    
    def __init__(self):
        self.hebrew_strongs = {}  # H#### -> definition data
        self.greek_strongs = {}   # G#### -> definition data
        self.strongs_to_english = defaultdict(Counter)  # Strong's -> English words
        self.semantic_clusters = defaultdict(set)  # Group related Strong's numbers
        
    def load_strongs_lexicon(self, lexicon_path: Path, language: str = 'hebrew'):
        """Load Strong's lexicon data."""
        logger.info(f"Loading Strong's {language} lexicon from {lexicon_path}")
        
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            lexicon_data = json.load(f)
            
        for strongs_num, entry in lexicon_data.items():
            # Parse Strong's entry
            parsed_entry = {
                'number': strongs_num,
                'original': entry.get('original', ''),
                'transliteration': entry.get('translit', ''),
                'pronunciation': entry.get('pronounce', ''),
                'definition': entry.get('definition', ''),
                'kjv_usage': entry.get('kjv_usage', ''),
                'outline': entry.get('outline', ''),
                'short_def': self._extract_short_definition(entry),
                'glosses': self._extract_glosses(entry),
                'semantic_field': self._determine_semantic_field(entry)
            }
            
            if language == 'hebrew':
                self.hebrew_strongs[strongs_num] = parsed_entry
            else:
                self.greek_strongs[strongs_num] = parsed_entry
                
            # Build Strong's to English mapping
            self._build_translation_mapping(strongs_num, parsed_entry)
            
    def _extract_short_definition(self, entry: Dict) -> str:
        """Extract concise definition from Strong's entry."""
        definition = entry.get('definition', '')
        
        # Take first sentence or phrase
        if ';' in definition:
            return definition.split(';')[0].strip()
        elif ',' in definition:
            parts = definition.split(',')
            return parts[0].strip() if len(parts[0]) < 50 else definition[:50]
        else:
            return definition[:100] if len(definition) > 100 else definition
            
    def _extract_glosses(self, entry: Dict) -> List[str]:
        """Extract all possible glosses/translations."""
        glosses = []
        
        # From KJV usage
        kjv_usage = entry.get('kjv_usage', '')
        if kjv_usage:
            # Parse format: "word (5x), another (3x), ..."
            matches = re.findall(r'(\w+)\s*\(\d+x?\)', kjv_usage)
            glosses.extend(matches)
            
        # From definition
        definition = entry.get('definition', '')
        if definition:
            # Extract quoted words as potential glosses
            quoted = re.findall(r'"([^"]+)"', definition)
            glosses.extend(quoted)
            
            # Extract words after "meaning" or "denotes"
            meaning_match = re.search(r'(?:meaning|denotes?|signifies?)\s+["\']?(\w+)', 
                                    definition, re.I)
            if meaning_match:
                glosses.append(meaning_match.group(1))
                
        # From outline of Biblical usage
        outline = entry.get('outline', '')
        if outline:
            # Extract main points (often single words)
            lines = outline.split('\n')
            for line in lines:
                if re.match(r'^\s*\d+\.\s*(\w+)', line):
                    match = re.match(r'^\s*\d+\.\s*(\w+)', line)
                    if match:
                        glosses.append(match.group(1))
                        
        # Clean and deduplicate
        glosses = [g.lower().strip() for g in glosses]
        return list(dict.fromkeys(glosses))  # Preserve order, remove duplicates
        
    def _determine_semantic_field(self, entry: Dict) -> str:
        """Categorize word into semantic field."""
        definition = entry.get('definition', '').lower()
        outline = entry.get('outline', '').lower()
        full_text = f"{definition} {outline}"
        
        # Theological terms
        if any(term in full_text for term in ['god', 'lord', 'divine', 'holy', 'worship']):
            return 'theological'
            
        # Action verbs
        if any(term in full_text for term in ['to go', 'to come', 'to do', 'to make', 
                                              'to speak', 'to say']):
            return 'action'
            
        # Emotions/states
        if any(term in full_text for term in ['love', 'hate', 'fear', 'joy', 'anger',
                                              'happy', 'sad']):
            return 'emotion'
            
        # Relationships
        if any(term in full_text for term in ['father', 'mother', 'son', 'daughter',
                                              'brother', 'sister', 'family']):
            return 'kinship'
            
        # Time/space
        if any(term in full_text for term in ['day', 'night', 'year', 'time', 'place',
                                              'heaven', 'earth']):
            return 'cosmological'
            
        return 'general'
        
    def _build_translation_mapping(self, strongs_num: str, entry: Dict):
        """Build mapping from Strong's numbers to English translations."""
        # Add all glosses
        for gloss in entry['glosses']:
            self.strongs_to_english[strongs_num][gloss] += 1
            
        # Add words from KJV usage with their frequencies
        kjv_usage = entry.get('kjv_usage', '')
        if kjv_usage:
            matches = re.findall(r'(\w+)\s*\((\d+)x?\)', kjv_usage)
            for word, count in matches:
                self.strongs_to_english[strongs_num][word.lower()] += int(count)
                
        # Group semantically related Strong's numbers
        semantic_field = entry['semantic_field']
        self.semantic_clusters[semantic_field].add(strongs_num)
        
    def enhance_alignment_with_strongs(self, source_word_data: Dict, 
                                     target_words: List[str],
                                     base_scores: Dict[str, float]) -> Dict[str, float]:
        """Enhance alignment scores using Strong's concordance data."""
        enhanced_scores = base_scores.copy()
        
        # Get Strong's number from source word
        strongs_ref = source_word_data.get('lemma', '').split('/')[-1]
        if not strongs_ref:
            return enhanced_scores
            
        # Add 'H' or 'G' prefix if needed
        if strongs_ref.isdigit():
            lang = 'H' if source_word_data.get('language') == 'hebrew' else 'G'
            strongs_ref = f"{lang}{strongs_ref}"
            
        # Look up in appropriate lexicon
        strongs_entry = None
        if strongs_ref.startswith('H'):
            strongs_entry = self.hebrew_strongs.get(strongs_ref)
        elif strongs_ref.startswith('G'):
            strongs_entry = self.greek_strongs.get(strongs_ref)
            
        if not strongs_entry:
            return enhanced_scores
            
        # Enhance scores based on Strong's data
        source_word = source_word_data.get('text', '')
        
        for i, target_word in enumerate(target_words):
            target_lower = target_word.lower()
            
            # 1. Direct gloss match (highest confidence)
            if target_lower in strongs_entry['glosses']:
                boost = 0.4  # Strong boost for direct gloss match
                enhanced_scores[f"{source_word}→{target_word}"] = \
                    enhanced_scores.get(f"{source_word}→{target_word}", 0.0) + boost
                    
            # 2. KJV usage frequency (weighted by occurrence count)
            if strongs_ref in self.strongs_to_english:
                usage_count = self.strongs_to_english[strongs_ref].get(target_lower, 0)
                if usage_count > 0:
                    # Normalize by total usage
                    total_usage = sum(self.strongs_to_english[strongs_ref].values())
                    usage_prob = usage_count / total_usage
                    boost = 0.3 * usage_prob  # Weighted by frequency
                    enhanced_scores[f"{source_word}→{target_word}"] = \
                        enhanced_scores.get(f"{source_word}→{target_word}", 0.0) + boost
                        
            # 3. Semantic field match (lower confidence)
            # Check if target word appears in definitions of same semantic field
            semantic_field = strongs_entry['semantic_field']
            related_strongs = self.semantic_clusters[semantic_field]
            
            for related_num in related_strongs:
                if related_num != strongs_ref:
                    related_entry = (self.hebrew_strongs.get(related_num) or 
                                   self.greek_strongs.get(related_num))
                    if related_entry and target_lower in related_entry['glosses']:
                        boost = 0.1  # Small boost for semantic field match
                        enhanced_scores[f"{source_word}→{target_word}"] = \
                            enhanced_scores.get(f"{source_word}→{target_word}", 0.0) + boost
                        break
                        
        return enhanced_scores
        
    def get_translation_candidates(self, strongs_num: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most likely English translations for a Strong's number."""
        if strongs_num not in self.strongs_to_english:
            return []
            
        translations = self.strongs_to_english[strongs_num]
        total = sum(translations.values())
        
        # Calculate probabilities
        candidates = []
        for word, count in translations.most_common(top_k):
            prob = count / total if total > 0 else 0.0
            candidates.append((word, prob))
            
        return candidates
        
    def explain_alignment(self, source_word_data: Dict, target_word: str) -> Dict[str, any]:
        """Provide detailed explanation for why words align."""
        explanation = {
            'source': source_word_data.get('text', ''),
            'target': target_word,
            'strongs': source_word_data.get('lemma', '').split('/')[-1],
            'confidence_factors': []
        }
        
        strongs_ref = explanation['strongs']
        if strongs_ref.isdigit():
            lang = 'H' if source_word_data.get('language') == 'hebrew' else 'G'
            strongs_ref = f"{lang}{strongs_ref}"
            
        strongs_entry = (self.hebrew_strongs.get(strongs_ref) or 
                        self.greek_strongs.get(strongs_ref))
        
        if strongs_entry:
            explanation['definition'] = strongs_entry['short_def']
            explanation['glosses'] = strongs_entry['glosses']
            
            # Check why this alignment makes sense
            target_lower = target_word.lower()
            
            if target_lower in strongs_entry['glosses']:
                explanation['confidence_factors'].append({
                    'factor': 'direct_gloss',
                    'description': f"'{target_word}' is a primary gloss in Strong's",
                    'confidence': 0.9
                })
                
            usage_count = self.strongs_to_english[strongs_ref].get(target_lower, 0)
            if usage_count > 0:
                explanation['confidence_factors'].append({
                    'factor': 'kjv_usage',
                    'description': f"Used {usage_count}x in KJV for this Strong's number",
                    'confidence': 0.7
                })
                
            if strongs_entry['semantic_field'] == 'theological' and \
               any(term in target_lower for term in ['god', 'lord', 'holy']):
                explanation['confidence_factors'].append({
                    'factor': 'semantic_field',
                    'description': 'Both in theological semantic field',
                    'confidence': 0.5
                })
                
        return explanation
        
    def generate_strongs_report(self) -> str:
        """Generate report on Strong's integration."""
        report = []
        report.append("Strong's Concordance Integration Report")
        report.append("=" * 50)
        report.append(f"Hebrew entries loaded: {len(self.hebrew_strongs)}")
        report.append(f"Greek entries loaded: {len(self.greek_strongs)}")
        report.append(f"Unique English glosses: {len(set(w for counts in self.strongs_to_english.values() for w in counts))}")
        report.append("")
        
        # Semantic field distribution
        report.append("Semantic Field Distribution:")
        field_counts = Counter()
        for entry in list(self.hebrew_strongs.values()) + list(self.greek_strongs.values()):
            field_counts[entry['semantic_field']] += 1
            
        for field, count in field_counts.most_common():
            report.append(f"  {field}: {count} entries")
            
        report.append("")
        
        # Most frequent translations
        report.append("Most Frequent Strong's → English Mappings:")
        all_mappings = []
        for strongs, translations in self.strongs_to_english.items():
            for word, count in translations.items():
                all_mappings.append((strongs, word, count))
                
        all_mappings.sort(key=lambda x: x[2], reverse=True)
        
        for strongs, word, count in all_mappings[:20]:
            entry = self.hebrew_strongs.get(strongs) or self.greek_strongs.get(strongs)
            if entry:
                report.append(f"  {strongs} ({entry['transliteration']}) → '{word}' ({count}x)")
                
        return "\n".join(report)