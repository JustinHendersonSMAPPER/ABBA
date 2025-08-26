#!/usr/bin/env python3
"""
Concept Mapper - Batch processing and management for biblical concepts

This module handles:
1. Batch processing all concepts from concepts.yaml
2. Storing results in database for fast retrieval
3. Exporting results to various formats
4. Progress tracking and reporting
"""

import sqlite3
import json
import csv
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from .semantic_concordance import SemanticConcordance, ConceptDefinition, SemanticMatch
from ..database.sqlite_manager import SQLiteManager
from ..logging_setup import logger


@dataclass
class ConceptMappingStats:
    """Statistics for a concept mapping."""
    concept_name: str
    total_matches: int
    lexical_matches: int
    semantic_matches: int
    processing_time: float
    ollama_validations: Dict[str, int]
    average_confidence: float
    coverage_books: int
    last_updated: str


class ConceptMapper:
    """
    Manages batch processing and storage of biblical concept mappings.
    """
    
    def __init__(self, db_path: Path, chroma_path: Path, ollama_config: Dict):
        """Initialize concept mapper."""
        self.db_path = db_path
        self.db_manager = SQLiteManager(db_path)
        self.semantic_concordance = SemanticConcordance(db_path, chroma_path, ollama_config)
        self._ensure_tables()
        
    def _ensure_tables(self):
        """Ensure concept mapping tables exist."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for concept mappings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_mappings (
                    concept_name TEXT NOT NULL,
                    verse_id TEXT NOT NULL,
                    book TEXT NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    match_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    semantic_score REAL,
                    ollama_validation TEXT,
                    ollama_confidence REAL,
                    evidence TEXT,
                    strongs_matched TEXT,
                    original_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (concept_name, verse_id)
                )
            """)
            
            # Table for concept statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_stats (
                    concept_name TEXT PRIMARY KEY,
                    total_matches INTEGER NOT NULL,
                    lexical_matches INTEGER NOT NULL,
                    semantic_matches INTEGER NOT NULL,
                    processing_time REAL NOT NULL,
                    ollama_validations TEXT,
                    average_confidence REAL NOT NULL,
                    coverage_books INTEGER NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_concept_verse 
                ON concept_mappings(verse_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_concept_confidence 
                ON concept_mappings(concept_name, confidence DESC)
            """)
            
            conn.commit()
    
    def process_all_concepts(self, concepts_path: Path, 
                           max_semantic_per_concept: int = 100,
                           validate_semantic: bool = True,
                           force_reprocess: bool = False) -> Dict[str, ConceptMappingStats]:
        """
        Process all concepts from concepts.yaml and store results.
        
        Args:
            concepts_path: Path to concepts.yaml
            max_semantic_per_concept: Maximum semantic matches per concept
            validate_semantic: Whether to validate with Ollama
            force_reprocess: Force reprocessing even if data exists
            
        Returns:
            Dictionary of concept statistics
        """
        logger.info(f"Processing all concepts from {concepts_path}")
        
        # Load concepts
        with open(concepts_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'concepts' not in data:
            raise ValueError("No 'concepts' key found in YAML")
        
        concepts = data['concepts']
        logger.info(f"Found {len(concepts)} concepts to process")
        
        # Check what needs processing
        if not force_reprocess:
            existing = self._get_existing_concepts()
            concepts_to_process = [c for c in concepts if c['name'] not in existing]
            if len(concepts_to_process) < len(concepts):
                logger.info(f"Skipping {len(concepts) - len(concepts_to_process)} already processed concepts")
        else:
            concepts_to_process = concepts
            # Clear existing data
            self._clear_concept_data()
        
        # Process each concept
        stats = {}
        for i, concept_data in enumerate(concepts_to_process, 1):
            name = concept_data['name']
            logger.info(f"\n[{i}/{len(concepts_to_process)}] Processing concept: {name}")
            
            try:
                stat = self.process_concept(
                    concept_data,
                    max_semantic_results=max_semantic_per_concept,
                    validate_semantic=validate_semantic
                )
                stats[name] = stat
                
                # Show progress
                logger.info(f"   ✓ {name}: {stat.total_matches} matches "
                          f"({stat.lexical_matches} lexical, {stat.semantic_matches} semantic)")
                
            except Exception as e:
                logger.error(f"   ✗ Failed to process {name}: {e}")
                continue
        
        logger.info(f"\nCompleted processing {len(stats)} concepts")
        return stats
    
    def process_concept(self, concept_data: Dict,
                       max_semantic_results: int = 100,
                       validate_semantic: bool = True) -> ConceptMappingStats:
        """Process a single concept and store results."""
        start_time = time.time()
        
        # Create concept definition
        concept = ConceptDefinition(
            name=concept_data['name'],
            description=concept_data.get('description', ''),
            primary_strongs=concept_data.get('strongs_numbers', []),
            extended_strongs=[],  # Could be enhanced from YAML
            validation_source="concepts.yaml"
        )
        
        # Build semantic concordance
        matches = self.semantic_concordance.build_semantic_concordance(
            concept,
            max_semantic_results=max_semantic_results,
            validate_semantic=validate_semantic
        )
        
        # Store matches in database
        self._store_matches(concept.name, matches)
        
        # Calculate statistics
        stats = self._calculate_stats(concept.name, matches, time.time() - start_time)
        
        # Store statistics
        self._store_stats(stats)
        
        return stats
    
    def search_concept(self, concept_name: str) -> List[SemanticMatch]:
        """
        Search for a concept by name, using cached results if available.
        
        Args:
            concept_name: Name of the concept to search
            
        Returns:
            List of semantic matches
        """
        # First check if we have cached results
        cached_matches = self._get_cached_matches(concept_name)
        if cached_matches:
            logger.info(f"Found {len(cached_matches)} cached matches for '{concept_name}'")
            return cached_matches
        
        # If not cached, try to find in concepts.yaml and process
        logger.info(f"No cached results for '{concept_name}', searching concepts.yaml")
        
        # This is a simplified search - in production you'd want better matching
        concepts_path = Path("abba/concepts.yaml")
        if concepts_path.exists():
            with open(concepts_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            for concept_data in data.get('concepts', []):
                if concept_data['name'].lower() == concept_name.lower():
                    logger.info(f"Found concept definition, processing...")
                    stats = self.process_concept(concept_data)
                    return self._get_cached_matches(concept_name)
        
        logger.warning(f"Concept '{concept_name}' not found")
        return []
    
    def export_mappings(self, output_path: str, format: str = 'json'):
        """
        Export all concept mappings to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        logger.info(f"Exporting concept mappings to {output_path} as {format}")
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if format.lower() == 'json':
                # Export as JSON
                cursor.execute("""
                    SELECT concept_name, verse_id, book, chapter, verse,
                           match_type, confidence, semantic_score,
                           ollama_validation, evidence, strongs_matched
                    FROM concept_mappings
                    ORDER BY concept_name, confidence DESC
                """)
                
                results = {}
                for row in cursor.fetchall():
                    concept = row[0]
                    if concept not in results:
                        results[concept] = []
                    
                    results[concept].append({
                        'verse_id': row[1],
                        'book': row[2],
                        'chapter': row[3],
                        'verse': row[4],
                        'match_type': row[5],
                        'confidence': row[6],
                        'semantic_score': row[7],
                        'ollama_validation': row[8],
                        'evidence': row[9],
                        'strongs_matched': json.loads(row[10]) if row[10] else []
                    })
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                # Export as CSV
                cursor.execute("""
                    SELECT concept_name, verse_id, book, chapter, verse,
                           match_type, confidence, semantic_score,
                           ollama_validation, evidence
                    FROM concept_mappings
                    ORDER BY concept_name, confidence DESC
                """)
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Concept', 'Verse ID', 'Book', 'Chapter', 'Verse',
                        'Match Type', 'Confidence', 'Semantic Score',
                        'Validation', 'Evidence'
                    ])
                    writer.writerows(cursor.fetchall())
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Export completed: {output_path}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of all concept mappings."""
        report = []
        report.append("# Biblical Concept Mapping Report")
        report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Summary Statistics")
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT concept_name) as concepts,
                    COUNT(*) as total_mappings,
                    AVG(confidence) as avg_confidence,
                    COUNT(DISTINCT verse_id) as unique_verses
                FROM concept_mappings
            """)
            
            stats = cursor.fetchone()
            report.append(f"\n- Total Concepts: {stats[0]}")
            report.append(f"- Total Mappings: {stats[1]:,}")
            report.append(f"- Average Confidence: {stats[2]:.3f}")
            report.append(f"- Unique Verses: {stats[3]:,}")
            
            # Per-concept statistics
            report.append("\n## Concept Details")
            
            cursor.execute("""
                SELECT * FROM concept_stats
                ORDER BY total_matches DESC
            """)
            
            for row in cursor.fetchall():
                name = row[0]
                report.append(f"\n### {name}")
                report.append(f"- Total Matches: {row[1]:,}")
                report.append(f"- Lexical: {row[2]:,} | Semantic: {row[3]:,}")
                report.append(f"- Processing Time: {row[4]:.1f}s")
                report.append(f"- Average Confidence: {row[6]:.3f}")
                report.append(f"- Books Covered: {row[7]}")
                
                # Sample top matches
                cursor.execute("""
                    SELECT verse_id, confidence, match_type
                    FROM concept_mappings
                    WHERE concept_name = ?
                    ORDER BY confidence DESC
                    LIMIT 5
                """, (name,))
                
                report.append("\nTop Matches:")
                for verse_id, conf, mtype in cursor.fetchall():
                    report.append(f"  - {verse_id} ({mtype}, {conf:.3f})")
        
        return '\n'.join(report)
    
    def _store_matches(self, concept_name: str, matches: List[SemanticMatch]):
        """Store matches in database."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing matches for this concept
            cursor.execute("DELETE FROM concept_mappings WHERE concept_name = ?", (concept_name,))
            
            # Insert new matches
            for match in matches:
                cursor.execute("""
                    INSERT INTO concept_mappings (
                        concept_name, verse_id, book, chapter, verse,
                        match_type, confidence, semantic_score,
                        ollama_validation, ollama_confidence,
                        evidence, strongs_matched, original_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept_name,
                    match.verse_id,
                    match.book,
                    match.chapter,
                    match.verse,
                    match.match_type,
                    match.confidence,
                    match.semantic_score,
                    match.ollama_validation,
                    match.ollama_confidence,
                    match.evidence,
                    json.dumps(match.strongs_matched),
                    match.original_text
                ))
            
            conn.commit()
    
    def _calculate_stats(self, concept_name: str, matches: List[SemanticMatch], 
                        processing_time: float) -> ConceptMappingStats:
        """Calculate statistics for a concept mapping."""
        lexical = [m for m in matches if not m.is_semantic_only]
        semantic = [m for m in matches if m.is_semantic_only]
        
        # Ollama validation counts
        ollama_validations = {}
        for match in semantic:
            if match.ollama_validation:
                ollama_validations[match.ollama_validation] = \
                    ollama_validations.get(match.ollama_validation, 0) + 1
        
        # Book coverage
        unique_books = len(set(m.book for m in matches))
        
        # Average confidence
        avg_confidence = sum(m.confidence for m in matches) / len(matches) if matches else 0
        
        return ConceptMappingStats(
            concept_name=concept_name,
            total_matches=len(matches),
            lexical_matches=len(lexical),
            semantic_matches=len(semantic),
            processing_time=processing_time,
            ollama_validations=ollama_validations,
            average_confidence=avg_confidence,
            coverage_books=unique_books,
            last_updated=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _store_stats(self, stats: ConceptMappingStats):
        """Store statistics in database."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO concept_stats (
                    concept_name, total_matches, lexical_matches, semantic_matches,
                    processing_time, ollama_validations, average_confidence,
                    coverage_books, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.concept_name,
                stats.total_matches,
                stats.lexical_matches,
                stats.semantic_matches,
                stats.processing_time,
                json.dumps(stats.ollama_validations),
                stats.average_confidence,
                stats.coverage_books,
                stats.last_updated
            ))
            
            conn.commit()
    
    def _get_existing_concepts(self) -> set:
        """Get set of concepts already processed."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT concept_name FROM concept_mappings")
            return {row[0] for row in cursor.fetchall()}
    
    def _clear_concept_data(self):
        """Clear all concept mapping data."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM concept_mappings")
            cursor.execute("DELETE FROM concept_stats")
            conn.commit()
    
    def _get_cached_matches(self, concept_name: str) -> List[SemanticMatch]:
        """Retrieve cached matches from database."""
        matches = []
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT verse_id, book, chapter, verse, match_type,
                       confidence, semantic_score, ollama_validation,
                       ollama_confidence, evidence, strongs_matched,
                       original_text
                FROM concept_mappings
                WHERE concept_name = ?
                ORDER BY confidence DESC
            """, (concept_name,))
            
            for row in cursor.fetchall():
                match = SemanticMatch(
                    verse_id=row[0],
                    book=row[1],
                    chapter=row[2],
                    verse=row[3],
                    match_type=row[4],
                    confidence=row[5],
                    semantic_score=row[6] or 0.0,
                    ollama_validation=row[7],
                    ollama_confidence=row[8] or 0.0,
                    evidence=row[9],
                    strongs_matched=json.loads(row[10]) if row[10] else [],
                    original_text=row[11],
                    is_semantic_only=(row[4] == 'semantic')
                )
                matches.append(match)
        
        return matches