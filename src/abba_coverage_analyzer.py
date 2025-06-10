"""
ABBA Coverage Analyzer Module

Analyzes translation coverage using alignment models and extracted JSON translations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger('ABBA.CoverageAnalyzer')


class ABBACoverageAnalyzer:
    """Analyzes biblical translation coverage using alignment models."""
    
    def __init__(self, aligner=None, production_mode=True, confidence_threshold=0.0, collect_alignments=False):
        self.translations_dir = Path('data/sources/translations')
        self.morphology_dir = Path('data/sources/morphology')
        self.aligner = aligner
        self.confidence_threshold = confidence_threshold
        self.collect_alignments = collect_alignments
        
        # Production vs test mode configuration
        if production_mode:
            self.verses_per_book = None  # Process ALL verses for complete accuracy
            self.max_translations = None  # Process ALL translations (production mode)
        else:
            self.verses_per_book = 10   # Quick testing
            self.max_translations = 5   # Quick testing
        
    def get_available_translations(self, translation_filter: str = None) -> List[Path]:
        """Get list of available translation JSON files."""
        all_translations = list(self.translations_dir.glob('*.json'))
        
        if translation_filter:
            # Filter to specific translation
            filtered = [t for t in all_translations if t.stem == translation_filter]
            if filtered:
                return filtered
            else:
                logger.warning(f"Translation '{translation_filter}' not found. Available translations: {[t.stem for t in all_translations[:5]]}...")
                return []
        
        return all_translations
    
    def analyze_translation(self, translation_path: Path) -> Optional[Dict]:
        """Analyze coverage for a single translation."""
        if not self.aligner:
            logger.warning("No aligner provided - cannot analyze coverage")
            return None
            
        translation_id = translation_path.stem
        logger.info(f"Analyzing {translation_id}")
        
        # Load translation data
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {translation_path}: {e}")
            return None
        
        # Extract translation metadata
        translation_metadata = {}
        if 'books' in translation_data:
            # New format: extract metadata fields
            for field in ['version', 'name', 'language', 'copyright', 'source', 'website', 'license_url']:
                if field in translation_data:
                    translation_metadata[field] = translation_data[field]
        
        # Extract books data
        books_data = []
        if 'books' in translation_data:
            # New format: {version, name, language, books: {bookName: bookData}}
            for book_code, book_data in translation_data['books'].items():
                book_data['book_code'] = book_code
                books_data.append(book_data)
        elif isinstance(translation_data, list):
            # Old format: array of books
            books_data = translation_data
        else:
            logger.warning(f"Unknown translation format for {translation_id}")
            return None
        
        # Count books and verses
        book_count = len(books_data)
        verse_count = 0
        
        # Separate OT and NT books
        ot_books = []
        nt_books = []
        
        # Book codes for testament determination
        ot_book_codes = {
            'Gen', 'Exod', 'Lev', 'Num', 'Deut', 'Josh', 'Judg', 'Ruth', '1Sam', '2Sam',
            '1Kgs', '2Kgs', '1Chr', '2Chr', 'Ezra', 'Neh', 'Esth', 'Job', 'Ps', 'Prov',
            'Eccl', 'Song', 'Isa', 'Jer', 'Lam', 'Ezek', 'Dan', 'Hos', 'Joel', 'Amos',
            'Obad', 'Jonah', 'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal'
        }
        
        # Full English NT book names (as used in translation files)
        nt_book_names = {
            'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
            '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians', 
            'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
            '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews',
            'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John',
            'Jude', 'Revelation'
        }
        
        # Standard NT book codes (abbreviations)
        nt_book_codes = {
            'Matt', 'Mark', 'Luke', 'John', 'Acts', 'Rom', '1Cor', '2Cor',
            'Gal', 'Eph', 'Phil', 'Col', '1Thess', '2Thess', '1Tim', '2Tim',
            'Titus', 'Phlm', 'Heb', 'Jas', '1Pet', '2Pet', '1John', '2John',
            '3John', 'Jude', 'Rev'
        }
        
        # Mapping from full English names to standard abbreviations
        english_to_abbrev = {
            'Matthew': 'Matt', 'Mark': 'Mark', 'Luke': 'Luke', 'John': 'John',
            'Acts': 'Acts', 'Romans': 'Rom', '1 Corinthians': '1Cor', 
            '2 Corinthians': '2Cor', 'Galatians': 'Gal', 'Ephesians': 'Eph',
            'Philippians': 'Phil', 'Colossians': 'Col', '1 Thessalonians': '1Thess',
            '2 Thessalonians': '2Thess', '1 Timothy': '1Tim', '2 Timothy': '2Tim',
            'Titus': 'Titus', 'Philemon': 'Phlm', 'Hebrews': 'Heb',
            'James': 'Jas', '1 Peter': '1Pet', '2 Peter': '2Pet',
            '1 John': '1John', '2 John': '2John', '3 John': '3John',
            'Jude': 'Jude', 'Revelation': 'Rev'
        }
        
        for book in books_data:
            # Get book code from either field
            book_name = book.get('book_code') or book.get('abbr') or book.get('book', '')
            
            # Count verses in this book
            for chapter in book.get('chapters', []):
                verse_count += len(chapter.get('verses', []))
            
            # Convert English name to standard abbreviation if needed
            book_code = english_to_abbrev.get(book_name, book_name)
            
            # Store the standardized book code for later use
            book['book_code'] = book_code
            
            # Categorize by testament using book code
            if book_code in ot_book_codes:
                ot_books.append(book)
            elif book_name in nt_book_names or book_code in nt_book_codes:
                nt_books.append(book)
            else:
                logger.warning(f"Book {book_code} (name: {book_name}) not categorized as OT or NT")
        
        # Analyze Hebrew (OT) coverage
        hebrew_stats = self._analyze_books_coverage(ot_books, 'hebrew')
        
        # Analyze Greek (NT) coverage  
        greek_stats = self._analyze_books_coverage(nt_books, 'greek')
        
        # Calculate overall coverage
        total_tokens = hebrew_stats['total_tokens'] + greek_stats['total_tokens']
        aligned_tokens = hebrew_stats['aligned_tokens'] + greek_stats['aligned_tokens']
        
        overall_coverage = (aligned_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        result = {
            'translation_id': translation_id,
            'book_count': book_count,
            'verse_count': verse_count,
            'overall_coverage': overall_coverage,
            'hebrew_coverage': hebrew_stats['alignment_coverage'] if hebrew_stats['total_tokens'] > 0 else None,
            'greek_coverage': greek_stats['alignment_coverage'] if greek_stats['total_tokens'] > 0 else None,
            'hebrew_stats': hebrew_stats,
            'greek_stats': greek_stats,
            'has_hebrew': len(ot_books) > 0,
            'has_greek': len(nt_books) > 0,
            'translation_metadata': translation_metadata  # Include source metadata
        }
        
        return result
    
    def _get_morphology_filename(self, book_code: str, language: str) -> str:
        """Get the correct morphology filename for a book code and language."""
        if language == 'hebrew':
            return f"{book_code}.json"
        elif language == 'greek':
            # Greek morphology files use lowercase full names
            nt_book_mapping = {
                'Matt': 'matthew',
                'Mark': 'mark', 
                'Luke': 'luke',
                'John': 'john',
                'Acts': 'acts',
                'Rom': 'romans',
                '1Cor': '1corinthians',
                '2Cor': '2corinthians',
                'Gal': 'galatians',
                'Eph': 'ephesians',
                'Phil': 'philippians',
                'Col': 'colossians',
                '1Thess': '1thessalonians',
                '2Thess': '2thessalonians',
                '1Tim': '1timothy',
                '2Tim': '2timothy',
                'Titus': 'titus',
                'Phlm': 'philemon',
                'Heb': 'hebrews',
                'Jas': 'james',
                '1Pet': '1peter',
                '2Pet': '2peter',
                '1John': '1john',
                '2John': '2john',
                '3John': '3john',
                'Jude': 'jude',
                'Rev': 'revelation'
            }
            return f"{nt_book_mapping.get(book_code, book_code.lower())}.json"
        return f"{book_code}.json"
    
    def _calculate_confidence_stats(self, confidence_scores: List[float]) -> Dict:
        """Calculate confidence distribution statistics."""
        if not confidence_scores:
            return {
                'avg_confidence': 0.0,
                'median_confidence': 0.0,
                'high_confidence_pct': 0.0,
                'medium_confidence_pct': 0.0,
                'low_confidence_pct': 0.0,
                'total_alignments': 0
            }
        
        import statistics
        
        avg_confidence = statistics.mean(confidence_scores)
        median_confidence = statistics.median(confidence_scores)
        
        # Categorize confidence levels
        high_conf = len([c for c in confidence_scores if c >= 0.7])  # High confidence
        medium_conf = len([c for c in confidence_scores if 0.4 <= c < 0.7])  # Medium confidence
        low_conf = len([c for c in confidence_scores if c < 0.4])  # Low confidence
        
        total = len(confidence_scores)
        
        return {
            'avg_confidence': avg_confidence,
            'median_confidence': median_confidence,
            'high_confidence_pct': (high_conf / total * 100) if total > 0 else 0,
            'medium_confidence_pct': (medium_conf / total * 100) if total > 0 else 0,
            'low_confidence_pct': (low_conf / total * 100) if total > 0 else 0,
            'total_alignments': total
        }

    def _analyze_books_coverage(self, books: List[Dict], language: str) -> Dict:
        """Analyze coverage for a set of books using the aligner."""
        if not books:
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        total_tokens = 0
        aligned_tokens = 0
        total_verses = 0
        aligned_verses = 0
        confidence_scores = []  # Track all confidence scores for analysis
        detailed_verses = []  # Track detailed verse alignments if requested
        
        # Sample first few verses from first book for efficiency
        sample_book = books[0]
        book_code = sample_book.get('book_code', '')
        
        # Get correct morphology filename for this language
        morph_filename = self._get_morphology_filename(book_code, language)
        morph_path = self.morphology_dir / language / morph_filename
        
        if not morph_path.exists():
            logger.info(f"No morphological data for {book_code} in {language} (tried {morph_filename})")
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        try:
            with open(morph_path, 'r', encoding='utf-8') as f:
                morph_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load morphological data: {e}")
            return {
                'total_tokens': 0,
                'aligned_tokens': 0,
                'total_verses': 0,
                'aligned_verses': 0,
                'alignment_coverage': 0,
                'verse_coverage': 0
            }
        
        # Process verses (configurable sample size for performance)
        verses_per_book = getattr(self, 'verses_per_book', 50)  # Default: 50 verses per book
        
        if 'chapters' in sample_book and sample_book['chapters']:
            # Process multiple chapters if needed
            all_verses = []
            for chapter in sample_book['chapters']:
                all_verses.extend(chapter.get('verses', []))
            
            # In production mode (verses_per_book=None), process ALL verses
            if verses_per_book is None:
                sample_size = len(all_verses)  # Process ALL verses in production
            else:
                sample_size = min(verses_per_book, len(all_verses))  # Limit for testing
            
            for i in range(sample_size):
                verse = all_verses[i]
                verse_text = verse.get('text', '')
                verse_num = verse.get('verse', i + 1)
                
                # Try to find matching morphology verse
                source_verse = None
                
                # Try index-based matching first (works for most cases)
                if i < len(morph_data.get('verses', [])):
                    potential_verse = morph_data['verses'][i]
                    # For Hebrew, verify verse number matches
                    if language == 'hebrew':
                        osis_id = potential_verse.get('osisID', '')
                        if '.' in osis_id:
                            parts = osis_id.split('.')
                            if len(parts) >= 3:
                                try:
                                    morph_verse_num = int(parts[2])
                                    if morph_verse_num == verse_num:
                                        source_verse = potential_verse
                                except ValueError:
                                    pass
                    else:
                        # For Greek, use index-based matching
                        source_verse = potential_verse
                
                # If Hebrew index-based failed, try verse number lookup
                if not source_verse and language == 'hebrew':
                    for morph_verse in morph_data.get('verses', []):
                        osis_id = morph_verse.get('osisID', '')
                        if '.' in osis_id:
                            parts = osis_id.split('.')
                            if len(parts) >= 3:
                                try:
                                    morph_verse_num = int(parts[2])
                                    if morph_verse_num == verse_num:
                                        source_verse = morph_verse
                                        break
                                except ValueError:
                                    continue
                
                if source_verse:
                    source_words = [w['text'] for w in source_verse.get('words', [])]
                    
                    # Enhanced word tokenization for target
                    import re
                    # Remove punctuation and split on whitespace
                    cleaned_text = re.sub(r'[^\w\s]', ' ', verse_text.strip())
                    target_words = [w for w in cleaned_text.split() if w]
                    
                    if source_words and target_words:
                        # Perform alignment
                        try:
                            alignments = self.aligner.align_verse(
                                source_words, target_words, language, 'english'
                            )
                        except Exception as e:
                            logger.error(f"Alignment failed for {book_code} verse {verse_num}: {e}")
                            alignments = []
                        
                        total_verses += 1
                        total_tokens += len(source_words)
                        
                        # Count aligned tokens and collect confidence scores
                        above_threshold = [a for a in alignments if a['confidence'] > self.confidence_threshold]
                        
                        # Count unique source words that have alignments (not total alignments)
                        unique_aligned_sources = set()
                        for alignment in above_threshold:
                            unique_aligned_sources.add(alignment['source_index'])
                        
                        aligned_in_verse = len(unique_aligned_sources)  # Unique source words with alignments
                        aligned_tokens += aligned_in_verse
                        
                        # Collect confidence scores for analysis
                        for alignment in alignments:
                            confidence_scores.append(alignment['confidence'])
                        
                        if aligned_in_verse > 0:
                            aligned_verses += 1
                        
                        # Collect detailed alignment data if requested
                        if self.collect_alignments:
                            # Get chapter and verse numbers
                            chapter_num = 1  # Default for now, could be extracted from context
                            for ch_idx, chapter in enumerate(sample_book.get('chapters', [])):
                                if i < sum(len(ch.get('verses', [])) for ch in sample_book['chapters'][:ch_idx + 1]):
                                    chapter_num = ch_idx + 1
                                    break
                            
                            # Create enhanced word mappings supporting span-based alignments
                            word_mappings, _ = self._create_span_aware_mappings(
                                source_words, target_words, above_threshold
                            )
                            
                            # Get original text from source words
                            original_text = ' '.join(source_words) if source_words else ''
                            
                            # Extract morphological breakdown from alignments
                            morphological_analysis = self._extract_morphological_breakdown(
                                source_words, alignments, language
                            )
                            
                            verse_detail = {
                                'book': book_code,
                                'chapter': chapter_num,
                                'verse': verse_num,
                                'language': language,
                                'translation_text': verse_text,
                                'original_text': original_text,
                                'morphological_analysis': morphological_analysis,
                                'word_mappings': word_mappings,  # Original → Translation mappings only
                                'total_alignments': len(alignments),
                                'verse_coverage': (aligned_in_verse / len(source_words) * 100) if source_words else 0
                            }
                            detailed_verses.append(verse_detail)
                            
                    else:
                        logger.info(f"Empty words for verse {verse_num} ({book_code} {language}): source={len(source_words)}, target={len(target_words)}")
                else:
                    logger.info(f"No source verse found for verse {verse_num} ({book_code} {language})")
        
        # Collect confidence statistics
        confidence_stats = self._calculate_confidence_stats(confidence_scores)
        
        logger.info(f"Coverage analysis for {book_code} ({language}): {total_verses} verses processed, {aligned_tokens}/{total_tokens} tokens aligned, avg confidence: {confidence_stats['avg_confidence']:.3f}")
        
        # Extrapolate from sample to full book set (production-aware)
        if sample_size > 0:
            # Count total verses across all books
            total_verses_all = sum(
                len(chapter.get('verses', []))
                for book in books
                for chapter in book.get('chapters', [])
            )
            
            # Only extrapolate if we sampled a subset
            if sample_size < total_verses_all:
                scale_factor = total_verses_all / sample_size
                total_tokens = int(total_tokens * scale_factor)
                aligned_tokens = int(aligned_tokens * scale_factor)
                total_verses = int(total_verses * scale_factor)
                aligned_verses = int(aligned_verses * scale_factor)
                logger.debug(f"Extrapolated from {sample_size} to {total_verses_all} verses (factor: {scale_factor:.2f})")
            else:
                logger.debug(f"Processed all {total_verses_all} verses (no extrapolation needed)")
        
        alignment_coverage = (aligned_tokens / total_tokens * 100) if total_tokens > 0 else 0
        verse_coverage = (aligned_verses / total_verses * 100) if total_verses > 0 else 0
        
        result = {
            'total_tokens': total_tokens,
            'aligned_tokens': aligned_tokens,
            'total_verses': total_verses,
            'aligned_verses': aligned_verses,
            'alignment_coverage': alignment_coverage,
            'verse_coverage': verse_coverage,
            'confidence_stats': confidence_stats
        }
        
        # Add detailed verses if requested
        if self.collect_alignments:
            result['detailed_verses'] = detailed_verses
            
        return result
    
    def analyze_all_translations(self, max_translations: int = None, translation_filter: str = None) -> List[Dict]:
        """Analyze coverage for available translations."""
        results = []
        translations = self.get_available_translations(translation_filter)
        
        # Use provided limit or instance default
        limit = max_translations or self.max_translations
        if limit:
            translations = translations[:limit]
            
        # Determine actual mode based on configuration
        actual_mode = "production" if self.verses_per_book is None and self.max_translations is None else "test"
        logger.info(f"Analyzing {len(translations)} translations in {actual_mode} mode")
        
        for translation_path in translations:
            result = self.analyze_translation(translation_path)
            if result:
                results.append(result)
        
        return results
    
    def generate_coverage_report(self, results: List[Dict]) -> str:
        """Generate a coverage report from analysis results."""
        if not results:
            return "No results to report"
        
        report_lines = [
            "Translation Coverage Analysis Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Translations analyzed: {len(results)}",
            f"Aligner: {self.aligner.__class__.__name__ if self.aligner else 'None'}",
            "",
        ]
        
        # Sort by overall coverage
        sorted_results = sorted(results, key=lambda x: x['overall_coverage'], reverse=True)
        
        # Summary statistics
        avg_coverage = sum(r['overall_coverage'] for r in results) / len(results)
        
        # Calculate Hebrew average only for translations that have Hebrew
        hebrew_results = [r for r in results if r['hebrew_coverage'] is not None]
        avg_hebrew = sum(r['hebrew_coverage'] for r in hebrew_results) / len(hebrew_results) if hebrew_results else 0
        
        # Calculate Greek average only for translations that have Greek  
        greek_results = [r for r in results if r['greek_coverage'] is not None]
        avg_greek = sum(r['greek_coverage'] for r in greek_results) / len(greek_results) if greek_results else 0
        
        # Calculate overall confidence statistics
        all_hebrew_stats = [r['hebrew_stats']['confidence_stats'] for r in results if 'confidence_stats' in r.get('hebrew_stats', {})]
        all_greek_stats = [r['greek_stats']['confidence_stats'] for r in results if 'confidence_stats' in r.get('greek_stats', {})]
        
        # Aggregate confidence scores
        all_confidences = []
        for stats in all_hebrew_stats + all_greek_stats:
            if stats['total_alignments'] > 0:
                all_confidences.append(stats['avg_confidence'])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        report_lines.extend([
            "Summary Statistics",
            "-" * 30,
            f"Average overall coverage: {avg_coverage:.1f}%",
            f"Average Hebrew coverage: {avg_hebrew:.1f}% ({len(hebrew_results)} translations with OT)",
            f"Average Greek coverage: {avg_greek:.1f}% ({len(greek_results)} translations with NT)",
            f"Average confidence score: {avg_confidence:.3f}",
            "",
            "Top 10 Translations by Coverage",
            "-" * 30,
        ])
        
        # Top translations
        for result in sorted_results[:10]:
            hebrew_str = f"{result['hebrew_coverage']:5.1f}%" if result['hebrew_coverage'] is not None else "  N/A"
            greek_str = f"{result['greek_coverage']:5.1f}%" if result['greek_coverage'] is not None else "  N/A"
            report_lines.append(
                f"{result['translation_id']:20s} - Overall: {result['overall_coverage']:5.1f}% "
                f"(Hebrew: {hebrew_str}, Greek: {greek_str})"
            )
        
        return "\n".join(report_lines)
    
    def _create_span_aware_mappings(self, source_words: List[str], target_words: List[str], 
                                   alignments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Create proper 1:1 word mappings with quality constraints.
        Fixes the broken span logic that was creating nonsensical multi-word mappings.
        """
        word_mappings = []
        
        # Initialize mappings for each original word
        for orig_idx, orig_word in enumerate(source_words):
            word_mappings.append({
                'original_word': orig_word,
                'original_index': orig_idx,
                'mapped_words': [],
                'has_alignment': False
            })
        
        # Track which target words have been used to prevent over-mapping
        target_word_usage = {}  # target_index -> list of (source_index, confidence)
        
        # Sort alignments by confidence (highest first) for greedy assignment
        sorted_alignments = sorted(alignments, key=lambda x: x['confidence'], reverse=True)
        
        for align in sorted_alignments:
            source_idx = align['source_index']
            target_idx = align['target_index']
            confidence = align['confidence']
            
            # Skip invalid indices
            if source_idx >= len(source_words) or target_idx >= len(target_words):
                continue
            
            # Quality constraints
            if confidence < 0.5:  # Minimum quality threshold
                continue
                
            # Check if this source word already has a high-confidence mapping
            current_mappings = word_mappings[source_idx]['mapped_words']
            if current_mappings:
                # If we already have a good mapping, only replace with significantly better one
                best_existing = max(current_mappings, key=lambda x: x['confidence'])
                if best_existing['confidence'] > confidence * 1.2:  # Require 20% improvement
                    continue
            
            # Check target word usage to prevent excessive many-to-one mappings
            if target_idx in target_word_usage:
                existing_mappings = target_word_usage[target_idx]
                # Allow max 2 source words per target word, and only if confidence is high
                if len(existing_mappings) >= 2:
                    continue
                # If target already mapped, require significantly higher confidence
                if any(existing_conf > confidence * 1.1 for _, existing_conf in existing_mappings):
                    continue
            
            # Create the mapping - single word only
            target_word = target_words[target_idx]
            translation_info = {
                'translation_word': target_word,  # Single word, not phrase
                'translation_start_index': target_idx,
                'translation_span': 1,  # Always 1 for proper word alignment
                'confidence': confidence,
                'method': align.get('method', 'unknown'),
                'alignment_type': 'word'  # Always word-level
            }
            
            # Clear any existing lower-confidence mappings for this source word
            word_mappings[source_idx]['mapped_words'] = [
                m for m in current_mappings if m['confidence'] > confidence
            ]
            
            # Add the new mapping
            word_mappings[source_idx]['mapped_words'].append(translation_info)
            word_mappings[source_idx]['has_alignment'] = True
            
            # Track target usage
            if target_idx not in target_word_usage:
                target_word_usage[target_idx] = []
            target_word_usage[target_idx].append((source_idx, confidence))
        
        # Final cleanup: ensure each source word has at most 1 mapping
        for mapping in word_mappings:
            if len(mapping['mapped_words']) > 1:
                # Keep only the highest confidence mapping
                best_mapping = max(mapping['mapped_words'], key=lambda x: x['confidence'])
                mapping['mapped_words'] = [best_mapping]
        
        return word_mappings, []
    
    def _extract_morphological_breakdown(self, source_words: List[str], alignments: List[Dict], language: str) -> List[Dict]:
        """
        Extract morphological breakdown for each word from alignment data.
        Creates a table-like structure with Hebrew/Greek, transliteration, literal meaning, and grammar.
        """
        morphological_analysis = []
        
        # Create a lookup for alignment data by source index
        alignment_lookup = {}
        for alignment in alignments:
            src_idx = alignment.get('source_index', -1)
            if src_idx >= 0:
                if src_idx not in alignment_lookup:
                    alignment_lookup[src_idx] = []
                alignment_lookup[src_idx].append(alignment)
        
        # Process each source word
        for i, word in enumerate(source_words):
            word_analysis = {
                'word_index': i,
                'original_word': word,
                'transliteration': '',
                'literal_meaning': '',
                'grammatical_notes': '',
                'pos': '',
                'lemma': '',
                'is_object_marker': False,
                'aligned': i in alignment_lookup
            }
            
            # Extract morphological data from alignments if available
            if i in alignment_lookup:
                # Use the first alignment with lexicon features
                for alignment in alignment_lookup[i]:
                    lexicon_features = alignment.get('lexicon_features', {})
                    if lexicon_features:
                        # Use matched_gloss as literal_meaning (the gloss that was actually matched)
                        # rather than the first gloss from all_glosses
                        matched_gloss = lexicon_features.get('matched_gloss', '')
                        is_object_marker = lexicon_features.get('is_object_marker', False)
                        
                        # Special handling for object markers
                        if is_object_marker:
                            literal_meaning = "(object marker)"
                        else:
                            # Use the morphological breakdown which combines prefixes with matched glosses
                            morphological_literal = lexicon_features.get('literal_meaning', '')
                            # If morphological breakdown has prefix info, use it; otherwise use matched_gloss
                            if morphological_literal and ('(' in morphological_literal or 'the' in morphological_literal.split()[:2]):
                                literal_meaning = morphological_literal
                            else:
                                literal_meaning = matched_gloss or morphological_literal
                        
                        # Basic lexicon features
                        word_analysis.update({
                            'transliteration': lexicon_features.get('transliteration', ''),
                            'literal_meaning': literal_meaning,
                            'grammatical_notes': lexicon_features.get('grammatical_notes', ''),
                            'pos': lexicon_features.get('pos', ''),
                            'lemma': lexicon_features.get('lemma', ''),
                            'is_object_marker': lexicon_features.get('is_object_marker', False),
                            'matched_gloss': matched_gloss,
                            'all_glosses': lexicon_features.get('all_glosses', [])
                        })
                        
                        # Extract structured morphological features (only include fields with values)
                        morphological_fields = [
                            'part_of_speech', 'noun_type', 'gender', 'number', 'state', 
                            'stem', 'aspect', 'person', 'is_name', 'is_place'
                        ]
                        for field in morphological_fields:
                            if field in lexicon_features and lexicon_features[field]:
                                word_analysis[field] = lexicon_features[field]
                        break  # Use the first alignment with morphological data
            
            # If no morphological data found, try to get basic analysis
            if not word_analysis['transliteration']:
                try:
                    # Try to get morphological breakdown from the ensemble's morphological aligner
                    morphological_aligner = self._get_morphological_aligner()
                    if morphological_aligner:
                        breakdown = morphological_aligner.get_morphological_breakdown(word, language)
                        if breakdown:
                            word_analysis.update({
                                'transliteration': breakdown.get('transliteration', ''),
                                'literal_meaning': breakdown.get('literal', ''),
                                'grammatical_notes': breakdown.get('grammar', '')
                            })
                except Exception as e:
                    logger.debug(f"Could not get morphological breakdown for {word}: {e}")
            
            morphological_analysis.append(word_analysis)
        
        return morphological_analysis
    
    def _get_morphological_aligner(self):
        """Get the morphological lexicon aligner from the ensemble aligner."""
        if hasattr(self.aligner, 'aligners'):
            # This is an ensemble aligner, find the morphological lexicon aligner
            for aligner, weight in self.aligner.aligners:
                if aligner.__class__.__name__ == 'MorphologicalLexiconAligner':
                    return aligner
        elif hasattr(self.aligner, 'get_morphological_breakdown'):
            # This is already a morphological aligner
            return self.aligner
        return None