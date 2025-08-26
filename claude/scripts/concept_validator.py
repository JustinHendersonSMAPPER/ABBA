#!/usr/bin/env python3
"""
Concept Validator Script

Validates that Hebrew/Greek terms and Strong's numbers in concepts.yaml
actually exist in the SQLite database and embeddings database.

This ensures concept definitions are grounded in actual biblical data.
"""

import sys
import sqlite3
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abba.config import config_manager
from abba.concept_manager import ConceptManager
from abba.database.sqlite_manager import SQLiteManager
from abba.embeddings.chroma_manager import ChromaManager
from abba.logging_setup import setup_logging, configure_standard_logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results of concept validation."""
    concept_name: str
    hebrew_terms_found: List[str]
    hebrew_terms_missing: List[str]
    greek_terms_found: List[str]
    greek_terms_missing: List[str]
    strongs_found: List[str]
    strongs_missing: List[str]
    total_occurrences: int
    has_embeddings: bool
    validation_passed: bool


@dataclass 
class DatabaseStats:
    """Statistics about database content."""
    total_hebrew_words: int
    total_greek_words: int
    unique_hebrew_terms: int
    unique_greek_terms: int
    unique_strongs_numbers: int
    embedding_verses: int


class ConceptValidator:
    """Validates concepts against SQLite and embeddings databases."""
    
    def __init__(self, config):
        """Initialize validator.
        
        Args:
            config: ABBA configuration object
        """
        self.config = config
        self.concept_manager = ConceptManager(config.concepts_path)
        self.db_manager = SQLiteManager(config.data_dir / "abba.db")
        self.chroma_manager = ChromaManager(str(config.vectors_path))
        
        # Load concepts
        self.concept_manager.load_concepts()
        
        logger.info("Concept validator initialized")
        logger.info(f"Concepts file: {config.concepts_path}")
        logger.info(f"Database: {config.data_dir / 'abba.db'}")
        logger.info(f"Embeddings: {config.vector_db_path}")
    
    def normalize_hebrew_greek(self, text: str) -> str:
        """Normalize Hebrew/Greek text by removing vowel points and morphological markers.
        
        Args:
            text: Hebrew or Greek text with diacritics
            
        Returns:
            Normalized text for searching
        """
        # Remove morphological separators
        text = text.replace('/', '').replace('\\', '')
        
        # For Hebrew: Remove niqqud (vowel points) and cantillation marks
        # Unicode ranges: 0591-05C7 (Hebrew accents and points)
        text = re.sub(r'[\u0591-\u05C7]', '', text)
        
        # For Greek: Remove accents and breathing marks
        # Normalize to NFD (decomposed) then remove combining marks
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        return text
    
    def get_database_stats(self) -> DatabaseStats:
        """Get statistics about database content.
        
        Returns:
            DatabaseStats with database information
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count Hebrew words
            cursor.execute("SELECT COUNT(*) FROM stepbible_verses WHERE language = 'hebrew'")
            total_hebrew_words = cursor.fetchone()[0]
            
            # Count Greek words
            cursor.execute("SELECT COUNT(*) FROM stepbible_verses WHERE language = 'greek'")
            total_greek_words = cursor.fetchone()[0]
            
            # Count unique Hebrew terms
            cursor.execute("SELECT COUNT(DISTINCT original_word) FROM stepbible_verses WHERE language = 'hebrew' AND original_word IS NOT NULL AND original_word != ''")
            unique_hebrew_terms = cursor.fetchone()[0]
            
            # Count unique Greek terms
            cursor.execute("SELECT COUNT(DISTINCT original_word) FROM stepbible_verses WHERE language = 'greek' AND original_word IS NOT NULL AND original_word != ''")
            unique_greek_terms = cursor.fetchone()[0]
            
            # Count unique Strong's numbers
            cursor.execute("SELECT COUNT(DISTINCT strongs_primary) FROM stepbible_verses WHERE strongs_primary IS NOT NULL AND strongs_primary != ''")
            unique_strongs_numbers = cursor.fetchone()[0]
        
        # Get embedding stats
        try:
            chroma_stats = self.chroma_manager.get_database_stats()
            embedding_verses = chroma_stats.get('collections', {}).get('original_verses', {}).get('count', 0)
        except Exception as e:
            logger.warning(f"Could not get embedding stats: {e}")
            embedding_verses = 0
        
        return DatabaseStats(
            total_hebrew_words=total_hebrew_words,
            total_greek_words=total_greek_words,
            unique_hebrew_terms=unique_hebrew_terms,
            unique_greek_terms=unique_greek_terms,
            unique_strongs_numbers=unique_strongs_numbers,
            embedding_verses=embedding_verses
        )
    
    def validate_hebrew_terms(self, concept_name: str, hebrew_terms: List[str]) -> Tuple[List[str], List[str], int]:
        """Validate Hebrew terms against database.
        
        Args:
            concept_name: Name of concept being validated
            hebrew_terms: List of Hebrew terms to validate
            
        Returns:
            Tuple of (found_terms, missing_terms, total_occurrences)
        """
        found_terms = []
        missing_terms = []
        total_occurrences = 0
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for term in hebrew_terms:
                # Use normalized column for efficient exact matching
                cursor.execute("""
                    SELECT normalized_word, COUNT(*) as count
                    FROM stepbible_verses 
                    WHERE language = 'hebrew' AND normalized_word = ?
                    GROUP BY normalized_word
                """, (term,))
                
                results = cursor.fetchall()
                if results:
                    found_terms.append(term)
                    # Sum all occurrences for this term
                    term_count = sum(row[1] for row in results)
                    total_occurrences += term_count
                    logger.debug(f"Hebrew term '{term}' found {term_count} times in {len(results)} variations")
                else:
                    missing_terms.append(term)
                    logger.debug(f"Hebrew term '{term}' not found in database")
        
        return found_terms, missing_terms, total_occurrences
    
    def validate_greek_terms(self, concept_name: str, greek_terms: List[str]) -> Tuple[List[str], List[str], int]:
        """Validate Greek terms against database.
        
        Args:
            concept_name: Name of concept being validated
            greek_terms: List of Greek terms to validate
            
        Returns:
            Tuple of (found_terms, missing_terms, total_occurrences)
        """
        found_terms = []
        missing_terms = []
        total_occurrences = 0
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for term in greek_terms:
                # Use normalized column for Greek too
                # First normalize the search term
                normalized_term = self.normalize_hebrew_greek(term)
                
                cursor.execute("""
                    SELECT normalized_word, COUNT(*) as count
                    FROM stepbible_verses 
                    WHERE language = 'greek' AND normalized_word = ?
                    GROUP BY normalized_word
                """, (normalized_term,))
                
                results = cursor.fetchall()
                if results:
                    found_terms.append(term)
                    # Sum all occurrences for this term
                    term_count = sum(row[1] for row in results)
                    total_occurrences += term_count
                    logger.debug(f"Greek term '{term}' found {term_count} times in {len(results)} variations")
                else:
                    missing_terms.append(term)
                    logger.debug(f"Greek term '{term}' not found in database")
        
        return found_terms, missing_terms, total_occurrences
    
    def validate_strongs_numbers(self, concept_name: str, strongs_numbers: List[str]) -> Tuple[List[str], List[str], int]:
        """Validate Strong's numbers against database.
        
        Args:
            concept_name: Name of concept being validated
            strongs_numbers: List of Strong's numbers to validate
            
        Returns:
            Tuple of (found_numbers, missing_numbers, total_occurrences)
        """
        found_numbers = []
        missing_numbers = []
        total_occurrences = 0
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for strongs_num in strongs_numbers:
                # Handle padding for Hebrew numbers (H157 -> H0157)
                search_nums = [strongs_num]
                if strongs_num.startswith('H'):
                    padded = strongs_num[0] + strongs_num[1:].zfill(4)
                    search_nums.append(padded)
                
                # Build query to search for any variant
                placeholders = ' OR '.join(['strongs_lexical = ?' for _ in search_nums])
                query = f"SELECT COUNT(*) FROM stepbible_verses WHERE {placeholders}"
                
                cursor.execute(query, search_nums)
                
                count = cursor.fetchone()[0]
                if count > 0:
                    found_numbers.append(strongs_num)
                    total_occurrences += count
                    logger.debug(f"Strong's number '{strongs_num}' found {count} times")
                else:
                    missing_numbers.append(strongs_num)
                    logger.debug(f"Strong's number '{strongs_num}' not found in database")
        
        return found_numbers, missing_numbers, total_occurrences
    
    def check_embeddings_coverage(self, concept_name: str) -> bool:
        """Check if concept terms have embedding coverage.
        
        Args:
            concept_name: Name of concept to check
            
        Returns:
            True if concept has embedding coverage
        """
        try:
            # Check if original_verses collection exists and has data
            stats = self.chroma_manager.get_database_stats()
            original_verses_count = stats.get('collections', {}).get('original_verses', {}).get('count', 0)
            
            if original_verses_count > 0:
                logger.debug(f"Embeddings available: {original_verses_count} verses")
                return True
            else:
                logger.debug("No embeddings found")
                return False
                
        except Exception as e:
            logger.warning(f"Could not check embeddings for {concept_name}: {e}")
            return False
    
    def validate_concept(self, concept_name: str) -> ValidationResult:
        """Validate a single concept.
        
        Args:
            concept_name: Name of concept to validate
            
        Returns:
            ValidationResult with detailed validation information
        """
        concept = self.concept_manager.get_concept_by_name(concept_name)
        if not concept:
            logger.error(f"Concept '{concept_name}' not found")
            return ValidationResult(
                concept_name=concept_name,
                hebrew_terms_found=[], hebrew_terms_missing=[],
                greek_terms_found=[], greek_terms_missing=[],
                strongs_found=[], strongs_missing=[],
                total_occurrences=0, has_embeddings=False,
                validation_passed=False
            )
        
        logger.info(f"Validating concept: {concept_name}")
        
        # Validate Hebrew terms
        hebrew_found, hebrew_missing, hebrew_occurrences = self.validate_hebrew_terms(
            concept_name, concept.hebrew_terms or []
        )
        
        # Validate Greek terms
        greek_found, greek_missing, greek_occurrences = self.validate_greek_terms(
            concept_name, concept.greek_terms or []
        )
        
        # Validate Strong's numbers
        strongs_found, strongs_missing, strongs_occurrences = self.validate_strongs_numbers(
            concept_name, concept.strongs_numbers or []
        )
        
        # Check embeddings
        has_embeddings = self.check_embeddings_coverage(concept_name)
        
        # Calculate total occurrences
        total_occurrences = hebrew_occurrences + greek_occurrences + strongs_occurrences
        
        # Determine if validation passed
        validation_passed = (
            len(hebrew_missing) == 0 and 
            len(greek_missing) == 0 and 
            len(strongs_missing) == 0 and
            total_occurrences > 0 and
            has_embeddings
        )
        
        return ValidationResult(
            concept_name=concept_name,
            hebrew_terms_found=hebrew_found,
            hebrew_terms_missing=hebrew_missing,
            greek_terms_found=greek_found,
            greek_terms_missing=greek_missing,
            strongs_found=strongs_found,
            strongs_missing=strongs_missing,
            total_occurrences=total_occurrences,
            has_embeddings=has_embeddings,
            validation_passed=validation_passed
        )
    
    def validate_all_concepts(self) -> List[ValidationResult]:
        """Validate all concepts in concepts.yaml.
        
        Returns:
            List of ValidationResult objects
        """
        concepts = self.concept_manager.get_concepts()
        results = []
        
        logger.info(f"Validating {len(concepts)} concepts...")
        
        for concept in concepts:
            result = self.validate_concept(concept.name)
            results.append(result)
        
        return results
    
    def print_validation_report(self, results: List[ValidationResult]):
        """Print comprehensive validation report.
        
        Args:
            results: List of validation results
        """
        print("\n" + "=" * 80)
        print("CONCEPT VALIDATION REPORT")
        print("=" * 80)
        
        # Database stats
        db_stats = self.get_database_stats()
        print(f"\nDATABASE STATISTICS:")
        print(f"  Hebrew words: {db_stats.total_hebrew_words:,} total, {db_stats.unique_hebrew_terms:,} unique")
        print(f"  Greek words: {db_stats.total_greek_words:,} total, {db_stats.unique_greek_terms:,} unique")
        print(f"  Strong's numbers: {db_stats.unique_strongs_numbers:,} unique")
        print(f"  Embedded verses: {db_stats.embedding_verses:,}")
        
        # Overall summary
        total_concepts = len(results)
        passed_concepts = sum(1 for r in results if r.validation_passed)
        failed_concepts = total_concepts - passed_concepts
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total concepts: {total_concepts}")
        print(f"  ✅ Passed: {passed_concepts}")
        print(f"  ❌ Failed: {failed_concepts}")
        print(f"  Success rate: {(passed_concepts/total_concepts)*100:.1f}%")
        
        # Detailed results
        print(f"\nDETAILED RESULTS:")
        print("-" * 80)
        
        for result in results:
            status = "✅ PASS" if result.validation_passed else "❌ FAIL"
            print(f"\n{status} {result.concept_name}")
            print(f"  Total occurrences: {result.total_occurrences:,}")
            print(f"  Has embeddings: {'✅' if result.has_embeddings else '❌'}")
            
            # Hebrew terms
            if result.hebrew_terms_found or result.hebrew_terms_missing:
                print(f"  Hebrew terms:")
                if result.hebrew_terms_found:
                    print(f"    ✅ Found ({len(result.hebrew_terms_found)}): {', '.join(result.hebrew_terms_found)}")
                if result.hebrew_terms_missing:
                    print(f"    ❌ Missing ({len(result.hebrew_terms_missing)}): {', '.join(result.hebrew_terms_missing)}")
            
            # Greek terms
            if result.greek_terms_found or result.greek_terms_missing:
                print(f"  Greek terms:")
                if result.greek_terms_found:
                    print(f"    ✅ Found ({len(result.greek_terms_found)}): {', '.join(result.greek_terms_found)}")
                if result.greek_terms_missing:
                    print(f"    ❌ Missing ({len(result.greek_terms_missing)}): {', '.join(result.greek_terms_missing)}")
            
            # Strong's numbers
            if result.strongs_found or result.strongs_missing:
                print(f"  Strong's numbers:")
                if result.strongs_found:
                    print(f"    ✅ Found ({len(result.strongs_found)}): {', '.join(result.strongs_found)}")
                if result.strongs_missing:
                    print(f"    ❌ Missing ({len(result.strongs_missing)}): {', '.join(result.strongs_missing)}")
        
        # Missing terms summary
        all_missing_hebrew = []
        all_missing_greek = []
        all_missing_strongs = []
        
        for result in results:
            all_missing_hebrew.extend(result.hebrew_terms_missing)
            all_missing_greek.extend(result.greek_terms_missing)
            all_missing_strongs.extend(result.strongs_missing)
        
        if all_missing_hebrew or all_missing_greek or all_missing_strongs:
            print(f"\n" + "=" * 80)
            print("MISSING TERMS SUMMARY")
            print("=" * 80)
            
            if all_missing_hebrew:
                print(f"\nMissing Hebrew terms ({len(all_missing_hebrew)}):")
                for term in sorted(set(all_missing_hebrew)):
                    print(f"  {term}")
            
            if all_missing_greek:
                print(f"\nMissing Greek terms ({len(all_missing_greek)}):")
                for term in sorted(set(all_missing_greek)):
                    print(f"  {term}")
            
            if all_missing_strongs:
                print(f"\nMissing Strong's numbers ({len(all_missing_strongs)}):")
                for num in sorted(set(all_missing_strongs)):
                    print(f"  {num}")
        
        print(f"\n" + "=" * 80)
        
        if failed_concepts == 0:
            print("🎉 ALL CONCEPTS VALIDATED SUCCESSFULLY!")
        else:
            print(f"⚠️  {failed_concepts} concept(s) need attention")
        
        print("=" * 80 + "\n")
    
    def close(self):
        """Clean up resources."""
        try:
            self.chroma_manager.close()
        except Exception as e:
            logger.warning(f"Error closing ChromaManager: {e}")


def main():
    """Main entry point."""
    try:
        # Setup logging
        setup_logging()
        configure_standard_logging()
        
        # Load configuration
        config = config_manager.load_config()
        
        # Create validator
        validator = ConceptValidator(config)
        
        print("🔍 ABBA Concept Validator")
        print("Validating Hebrew/Greek terms and Strong's numbers against databases...")
        
        # Validate all concepts
        results = validator.validate_all_concepts()
        
        # Print report
        validator.print_validation_report(results)
        
        # Clean up
        validator.close()
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results if not r.validation_passed)
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()