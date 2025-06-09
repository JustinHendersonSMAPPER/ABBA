"""
Cross-reference data loader for biblical cross-references.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from abba.cross_references.models import (
    CrossReference, ReferenceType, ReferenceRelationship,
    ReferenceConfidence
)
from abba.verse_id import VerseID, parse_verse_id

logger = logging.getLogger(__name__)


class CrossReferenceLoader:
    """Load cross-references from various data sources."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.references_cache: List[CrossReference] = []
    
    def load_from_json(self, file_path: Optional[Path] = None) -> List[CrossReference]:
        """Load cross-references from JSON file."""
        if file_path is None:
            file_path = self.data_dir / "cross_references.json"
        
        if not file_path.exists():
            logger.warning(f"Cross-reference file not found: {file_path}")
            return []
        
        references = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for ref_data in data.get('references', []):
                try:
                    # Parse verse IDs
                    source_verse = parse_verse_id(ref_data['source'])
                    target_verse = parse_verse_id(ref_data['target'])
                    
                    if not source_verse or not target_verse:
                        continue
                    
                    # Map string types to enums
                    ref_type = ReferenceType(ref_data.get('type', 'thematic_parallel'))
                    relationship = ReferenceRelationship(ref_data.get('relationship', 'parallels'))
                    
                    # Create confidence object
                    confidence = ReferenceConfidence(
                        overall_score=ref_data.get('confidence', 0.5),
                        textual_similarity=ref_data.get('textual_similarity', 0.5),
                        thematic_similarity=ref_data.get('thematic_similarity', 0.5),
                        structural_similarity=ref_data.get('structural_similarity', 0.5),
                        scholarly_consensus=ref_data.get('scholarly_consensus', 0.5),
                        uncertainty_factors=ref_data.get('uncertainty_factors', []),
                        lexical_links=ref_data.get('lexical_links', 0),
                        semantic_links=ref_data.get('semantic_links', 0)
                    )
                    
                    # Create cross-reference
                    cross_ref = CrossReference(
                        source_verse=source_verse,
                        target_verse=target_verse,
                        reference_type=ref_type,
                        relationship=relationship,
                        confidence=confidence,
                        topic_tags=ref_data.get('topic_tags', []),
                        theological_theme=ref_data.get('theological_theme', '')
                    )
                    
                    references.append(cross_ref)
                    
                except Exception as e:
                    logger.debug(f"Error parsing reference: {e}")
                    continue
            
            logger.info(f"Loaded {len(references)} cross-references from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading cross-references from {file_path}: {e}")
        
        self.references_cache.extend(references)
        return references
    
    def load_from_tsv(self, file_path: Path) -> List[CrossReference]:
        """Load cross-references from TSV file (Treasury of Scripture Knowledge format)."""
        if not file_path.exists():
            logger.warning(f"TSV file not found: {file_path}")
            return []
        
        references = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip header if present
                first_line = f.readline().strip()
                if not first_line.startswith('#') and '\t' in first_line:
                    # Process first line if it's not a header
                    parts = first_line.split('\t')
                    if len(parts) >= 2:
                        self._process_tsv_line(parts, references)
                
                # Process remaining lines
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            self._process_tsv_line(parts, references)
            
            logger.info(f"Loaded {len(references)} cross-references from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading TSV file {file_path}: {e}")
        
        self.references_cache.extend(references)
        return references
    
    def _process_tsv_line(self, parts: List[str], references: List[CrossReference]):
        """Process a single TSV line."""
        try:
            source_ref = parts[0].strip()
            target_refs = parts[1].strip()
            
            # Parse source verse
            source_verse = parse_verse_id(source_ref)
            if not source_verse:
                return
            
            # Parse target verses (may be semicolon-separated)
            for target_ref in target_refs.split(';'):
                target_ref = target_ref.strip()
                if not target_ref:
                    continue
                
                target_verse = parse_verse_id(target_ref)
                if not target_verse:
                    continue
                
                # Create basic cross-reference
                confidence = ReferenceConfidence(
                    overall_score=0.7,  # Default confidence for TSV imports
                    textual_similarity=0.5,
                    thematic_similarity=0.5,
                    structural_similarity=0.5,
                    scholarly_consensus=0.8,  # TSK is well-established
                    uncertainty_factors=[],
                    lexical_links=0,
                    semantic_links=0
                )
                
                cross_ref = CrossReference(
                    source_verse=source_verse,
                    target_verse=target_verse,
                    reference_type=ReferenceType.THEMATIC_PARALLEL,
                    relationship=ReferenceRelationship.PARALLELS,
                    confidence=confidence,
                    topic_tags=[],
                    theological_theme=""
                )
                
                references.append(cross_ref)
                
        except Exception as e:
            logger.debug(f"Error processing TSV line: {e}")
    
    def find_references_for_verse(self, verse_id: VerseID) -> List[CrossReference]:
        """Find all cross-references for a specific verse."""
        return [
            ref for ref in self.references_cache
            if ref.source_verse == verse_id
        ]
    
    def find_references_to_verse(self, verse_id: VerseID) -> List[CrossReference]:
        """Find all cross-references pointing to a specific verse."""
        return [
            ref for ref in self.references_cache
            if ref.target_verse == verse_id
        ]
    
    def get_bidirectional_references(self, verse_id: VerseID) -> Dict[str, List[CrossReference]]:
        """Get both outgoing and incoming references for a verse."""
        return {
            'from': self.find_references_for_verse(verse_id),
            'to': self.find_references_to_verse(verse_id)
        }
    
    def get_reference_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded cross-references."""
        if not self.references_cache:
            return {
                'total_references': 0,
                'unique_source_verses': 0,
                'unique_target_verses': 0,
                'reference_types': {},
                'relationships': {},
                'average_confidence': 0
            }
        
        # Collect statistics
        source_verses = set()
        target_verses = set()
        type_counts = {}
        relationship_counts = {}
        total_confidence = 0
        
        for ref in self.references_cache:
            source_verses.add(ref.source_verse)
            target_verses.add(ref.target_verse)
            
            type_name = ref.reference_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            rel_name = ref.relationship.value
            relationship_counts[rel_name] = relationship_counts.get(rel_name, 0) + 1
            
            total_confidence += ref.confidence.overall_score
        
        return {
            'total_references': len(self.references_cache),
            'unique_source_verses': len(source_verses),
            'unique_target_verses': len(target_verses),
            'reference_types': type_counts,
            'relationships': relationship_counts,
            'average_confidence': total_confidence / len(self.references_cache)
        }