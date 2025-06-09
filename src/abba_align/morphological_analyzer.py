"""
Advanced morphological analysis for biblical languages.

This module provides deep morphological analysis including:
- Prefix/suffix decomposition
- Root extraction
- Morphological pattern analysis
- Statistical morpheme alignment
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class MorphologicalAnalyzer:
    """Advanced morphological analyzer for biblical texts."""
    
    def __init__(self, language: str):
        self.language = language
        self.morpheme_alignments = defaultdict(Counter)
        
        # Hebrew morphological patterns
        self.hebrew_prefixes = {
            'ב': 'in/with',
            'כ': 'like/as', 
            'ל': 'to/for',
            'מ': 'from',
            'ה': 'the',
            'ו': 'and',
            'ש': 'that/which'
        }
        
        self.hebrew_suffixes = {
            # Pronominal suffixes
            'י': '1cs_my',
            'ך': '2ms_your',
            'ך': '2fs_your',
            'ו': '3ms_his',
            'ה': '3fs_her',
            'נו': '1cp_our',
            'כם': '2mp_your',
            'כן': '2fp_your',
            'ם': '3mp_their',
            'ן': '3fp_their',
            # Plural endings
            'ים': 'mp',
            'ות': 'fp'
        }
        
        # Greek morphological patterns
        self.greek_cases = {
            'ος': 'nom_sg_m',
            'ον': 'acc_sg_m/n',
            'ου': 'gen_sg_m/n',
            'ῳ': 'dat_sg_m/n',
            'οι': 'nom_pl_m',
            'ους': 'acc_pl_m',
            'ων': 'gen_pl',
            'οις': 'dat_pl_m/n'
        }
        
    def decompose_hebrew_word(self, word: str, morphology: Optional[str] = None) -> Dict[str, any]:
        """Decompose a Hebrew word into morphemes."""
        result = {
            'word': word,
            'prefixes': [],
            'root': '',
            'suffixes': [],
            'pattern': '',
            'morphology': morphology
        }
        
        current = word
        
        # Extract prefixes
        prefix_found = True
        while prefix_found and current:
            prefix_found = False
            for prefix, meaning in self.hebrew_prefixes.items():
                if current.startswith(prefix):
                    result['prefixes'].append({
                        'morpheme': prefix,
                        'meaning': meaning
                    })
                    current = current[len(prefix):]
                    prefix_found = True
                    break
        
        # Extract suffixes (from end)
        suffix_found = True
        while suffix_found and current:
            suffix_found = False
            for suffix, meaning in sorted(self.hebrew_suffixes.items(), 
                                        key=lambda x: len(x[0]), reverse=True):
                if current.endswith(suffix):
                    result['suffixes'].insert(0, {
                        'morpheme': suffix,
                        'meaning': meaning
                    })
                    current = current[:-len(suffix)]
                    suffix_found = True
                    break
        
        # What remains is likely the root/stem
        result['root'] = current
        
        # Detect pattern (simplified)
        if morphology:
            result['pattern'] = self._extract_pattern_from_morphology(morphology)
        
        return result
    
    def decompose_greek_word(self, word: str, morphology: Optional[str] = None) -> Dict[str, any]:
        """Decompose a Greek word into morphemes."""
        result = {
            'word': word,
            'stem': '',
            'ending': '',
            'case_info': '',
            'morphology': morphology
        }
        
        # Find case ending
        for ending, case_info in sorted(self.greek_cases.items(), 
                                      key=lambda x: len(x[0]), reverse=True):
            if word.endswith(ending):
                result['ending'] = ending
                result['case_info'] = case_info
                result['stem'] = word[:-len(ending)]
                break
        
        if not result['stem']:
            result['stem'] = word
            
        return result
    
    def _extract_pattern_from_morphology(self, morph_code: str) -> str:
        """Extract morphological pattern from code."""
        # Parse codes like "HVqp3ms" (Hebrew Verb qal perfect 3rd masc sing)
        if morph_code.startswith('H'):
            if 'V' in morph_code:
                # Verb patterns
                if 'q' in morph_code:
                    return 'Qal'
                elif 'n' in morph_code:
                    return 'Niphal'
                elif 'p' in morph_code and morph_code.count('p') > 1:
                    return 'Piel'
                elif 'h' in morph_code:
                    return 'Hiphil'
            elif 'N' in morph_code:
                return 'Noun'
        return 'Unknown'
    
    def learn_morpheme_alignments(self, aligned_pairs: List[Tuple[Dict, str]]):
        """Learn morpheme-to-translation alignments from parallel data."""
        for source_word_data, translation in aligned_pairs:
            if self.language == 'hebrew':
                decomp = self.decompose_hebrew_word(
                    source_word_data.get('text', ''),
                    source_word_data.get('morph', '')
                )
                
                # Learn prefix alignments
                for prefix_info in decomp['prefixes']:
                    prefix = prefix_info['morpheme']
                    # Look for corresponding prepositions in translation
                    for prep in ['in', 'with', 'to', 'for', 'from', 'and']:
                        if prep in translation.lower():
                            self.morpheme_alignments[prefix][prep] += 1
                
                # Learn suffix alignments
                for suffix_info in decomp['suffixes']:
                    suffix = suffix_info['morpheme']
                    meaning = suffix_info['meaning']
                    
                    # Look for pronouns
                    if 'my' in meaning and 'my' in translation:
                        self.morpheme_alignments[suffix]['my'] += 1
                    elif 'his' in meaning and 'his' in translation:
                        self.morpheme_alignments[suffix]['his'] += 1
                    elif 'their' in meaning and 'their' in translation:
                        self.morpheme_alignments[suffix]['their'] += 1
    
    def analyze_morphological_patterns(self, corpus: List[Dict]) -> Dict[str, any]:
        """Analyze morphological patterns in a corpus."""
        patterns = defaultdict(Counter)
        morpheme_freq = defaultdict(Counter)
        decompositions = []
        
        for verse_data in corpus:
            words = (verse_data.get('hebrew_words', []) if self.language == 'hebrew' 
                    else verse_data.get('greek_words', []))
            
            for word_data in words:
                word = word_data.get('text', '')
                morph = word_data.get('morph', '')
                
                if self.language == 'hebrew':
                    decomp = self.decompose_hebrew_word(word, morph)
                    
                    # Track prefix combinations
                    if decomp['prefixes']:
                        prefix_combo = '+'.join(p['morpheme'] for p in decomp['prefixes'])
                        patterns['prefix_combinations'][prefix_combo] += 1
                        
                    # Track suffix combinations
                    if decomp['suffixes']:
                        suffix_combo = '+'.join(s['morpheme'] for s in decomp['suffixes'])
                        patterns['suffix_combinations'][suffix_combo] += 1
                    
                    # Track morphological patterns
                    if decomp['pattern']:
                        patterns['verbal_patterns'][decomp['pattern']] += 1
                        
                else:  # Greek
                    decomp = self.decompose_greek_word(word, morph)
                    
                    if decomp['case_info']:
                        patterns['case_patterns'][decomp['case_info']] += 1
                        
                decompositions.append(decomp)
        
        return {
            'language': self.language,
            'total_words': len(decompositions),
            'patterns': dict(patterns),
            'sample_decompositions': decompositions[:10],
            'morpheme_alignments': dict(self.morpheme_alignments)
        }
    
    def apply_morphological_constraints(self, alignment_scores: Dict[str, Dict[str, float]],
                                      source_word_data: Dict, 
                                      target_words: List[str]) -> Dict[str, Dict[str, float]]:
        """Apply morphological constraints to alignment scores."""
        word = source_word_data.get('text', '')
        morph = source_word_data.get('morph', '')
        
        if self.language == 'hebrew':
            decomp = self.decompose_hebrew_word(word, morph)
            
            # Boost scores for morphologically motivated alignments
            for prefix_info in decomp['prefixes']:
                prefix = prefix_info['morpheme']
                expected_translations = self.morpheme_alignments.get(prefix, {})
                
                for target_word in target_words:
                    for expected, count in expected_translations.most_common(3):
                        if expected in target_word.lower():
                            # Boost alignment score
                            if word not in alignment_scores:
                                alignment_scores[word] = {}
                            current_score = alignment_scores[word].get(target_word, 0.0)
                            boost = min(0.2, count / 100.0)  # Confidence boost
                            alignment_scores[word][target_word] = current_score + boost
        
        return alignment_scores
    
    def generate_morphological_report(self, analysis: Dict) -> str:
        """Generate human-readable morphological analysis report."""
        report = []
        report.append(f"Morphological Analysis Report - {analysis['language'].title()}")
        report.append("=" * 50)
        report.append(f"Total words analyzed: {analysis['total_words']}")
        report.append("")
        
        if 'patterns' in analysis:
            patterns = analysis['patterns']
            
            if 'prefix_combinations' in patterns:
                report.append("Most Common Prefix Combinations:")
                for combo, count in patterns['prefix_combinations'].most_common(10):
                    report.append(f"  {combo}: {count}")
                report.append("")
                
            if 'suffix_combinations' in patterns:
                report.append("Most Common Suffix Combinations:")
                for combo, count in patterns['suffix_combinations'].most_common(10):
                    report.append(f"  {combo}: {count}")
                report.append("")
                
            if 'verbal_patterns' in patterns:
                report.append("Verbal Patterns:")
                for pattern, count in patterns['verbal_patterns'].most_common():
                    report.append(f"  {pattern}: {count}")
                report.append("")
        
        if 'sample_decompositions' in analysis:
            report.append("Sample Decompositions:")
            for decomp in analysis['sample_decompositions'][:5]:
                report.append(f"  {decomp['word']}:")
                if decomp.get('prefixes'):
                    prefixes = [p['morpheme'] + f" ({p['meaning']})" 
                               for p in decomp['prefixes']]
                    report.append(f"    Prefixes: {', '.join(prefixes)}")
                if decomp.get('root'):
                    report.append(f"    Root: {decomp['root']}")
                if decomp.get('suffixes'):
                    suffixes = [s['morpheme'] + f" ({s['meaning']})" 
                               for s in decomp['suffixes']]
                    report.append(f"    Suffixes: {', '.join(suffixes)}")
                report.append("")
        
        return "\n".join(report)