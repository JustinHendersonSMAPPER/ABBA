"""
Enhanced SQLite schema for semantic concept storage.
Stores all semantic data in the database with optional JSON exports for portability.
"""

import sqlite3
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime


class SemanticSchemaManager:
    """Manages the semantic concept database schema."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
    def initialize_schema(self):
        """Create all semantic concept tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Core concept definitions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    primary_strongs TEXT,      -- JSON array of primary Strong's numbers
                    extended_strongs TEXT,     -- JSON array of extended Strong's numbers
                    semantic_keywords TEXT,    -- JSON array of semantic keywords
                    hebrew_terms TEXT,         -- JSON array of Hebrew terms
                    greek_terms TEXT,          -- JSON array of Greek terms
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_version TEXT
                )
            """)
            
            # Enhanced verse-concept relationships with full context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verse_concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    verse_id TEXT NOT NULL,
                    concept_id INTEGER NOT NULL,
                    book TEXT NOT NULL,
                    chapter INTEGER NOT NULL,
                    verse INTEGER NOT NULL,
                    
                    -- Match information
                    match_type TEXT CHECK(match_type IN ('lexical', 'semantic', 'thematic')),
                    confidence REAL DEFAULT 0.0,
                    semantic_score REAL,
                    
                    -- Original language data
                    original_text TEXT,
                    strongs_matched TEXT,      -- JSON array of matched Strong's numbers
                    word_positions TEXT,       -- JSON array of word position data
                    
                    -- Ollama validation
                    ollama_validated BOOLEAN DEFAULT 0,
                    ollama_confidence REAL,
                    ollama_reasoning TEXT,
                    
                    -- Context
                    verse_text TEXT,           -- Full verse text for quick access
                    previous_verse_id TEXT,
                    next_verse_id TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(verse_id, concept_id, match_type),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """)
            
            # Concept themes for organizing study materials
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_themes (
                    theme_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    theme_name TEXT NOT NULL,
                    description TEXT,
                    verse_ids TEXT,            -- JSON array of verse IDs
                    key_insights TEXT,
                    theological_significance TEXT,
                    practical_application TEXT,
                    display_order INTEGER DEFAULT 0,
                    
                    UNIQUE(concept_id, theme_name),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """)
            
            # Cross-concept relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    related_concept_id INTEGER NOT NULL,
                    relationship_type TEXT CHECK(relationship_type IN 
                        ('related', 'contrasting', 'foundation', 'expression', 
                         'result', 'synonym', 'antonym', 'progression')),
                    strength REAL DEFAULT 0.5,
                    description TEXT,
                    
                    UNIQUE(concept_id, related_concept_id, relationship_type),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id),
                    FOREIGN KEY (related_concept_id) REFERENCES concepts(concept_id)
                )
            """)
            
            # Study notes and insights
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_study_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    note_type TEXT CHECK(note_type IN 
                        ('hebrew_perspective', 'greek_perspective', 'historical_context',
                         'theological_significance', 'practical_application', 'word_study')),
                    content TEXT NOT NULL,
                    references TEXT,           -- JSON array of supporting references
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """)
            
            # Processing statistics for tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_processing_stats (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id INTEGER NOT NULL,
                    total_matches INTEGER DEFAULT 0,
                    lexical_matches INTEGER DEFAULT 0,
                    semantic_matches INTEGER DEFAULT 0,
                    thematic_matches INTEGER DEFAULT 0,
                    books_covered INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    ollama_calls_made INTEGER DEFAULT 0,
                    embedding_searches_performed INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(concept_id),
                    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
                )
            """)
            
            # Semantic clusters for grouping related concepts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_clusters (
                    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    concept_ids TEXT,          -- JSON array of concept IDs
                    cluster_type TEXT CHECK(cluster_type IN 
                        ('virtue', 'emotion', 'action', 'theological', 'relational', 'moral'))
                )
            """)
            
            # Cache for expensive computations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_value TEXT,
                    cache_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            # Create comprehensive indexes for fast lookups
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_verse ON verse_concepts(verse_id)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_concept ON verse_concepts(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_type ON verse_concepts(match_type)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_confidence ON verse_concepts(confidence DESC)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_book ON verse_concepts(book)",
                "CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)",
                "CREATE INDEX IF NOT EXISTS idx_themes_concept ON concept_themes(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_concept ON concept_relationships(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_related ON concept_relationships(related_concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_study_notes_concept ON concept_study_notes(concept_id)",
                "CREATE INDEX IF NOT EXISTS idx_study_notes_type ON concept_study_notes(note_type)",
                "CREATE INDEX IF NOT EXISTS idx_cache_expires ON concept_cache(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_verse_concepts_semantic_score ON verse_concepts(semantic_score DESC) WHERE semantic_score IS NOT NULL"
            ]
            
            for index in indexes:
                cursor.execute(index)
            
            # Create useful views for common queries
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS concept_summary AS
                SELECT 
                    c.name,
                    c.description,
                    COUNT(DISTINCT vc.verse_id) as total_verses,
                    COUNT(DISTINCT vc.book) as books_covered,
                    AVG(vc.confidence) as avg_confidence,
                    MAX(ps.last_updated) as last_updated
                FROM concepts c
                LEFT JOIN verse_concepts vc ON c.concept_id = vc.concept_id
                LEFT JOIN concept_processing_stats ps ON c.concept_id = ps.concept_id
                GROUP BY c.concept_id
            """)
            
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS high_confidence_semantic AS
                SELECT 
                    c.name as concept_name,
                    vc.verse_id,
                    vc.verse_text,
                    vc.semantic_score,
                    vc.ollama_confidence,
                    vc.ollama_reasoning
                FROM verse_concepts vc
                JOIN concepts c ON vc.concept_id = c.concept_id
                WHERE vc.match_type = 'semantic' 
                AND vc.ollama_validated = 1
                AND vc.ollama_confidence >= 0.7
                ORDER BY vc.semantic_score DESC
            """)
            
            conn.commit()
    
    def export_concept_to_json(self, concept_name: str, output_path: Path) -> Dict:
        """Export a concept's complete data to JSON for portability."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get concept details
            cursor.execute("SELECT * FROM concepts WHERE name = ?", (concept_name,))
            concept_row = cursor.fetchone()
            if not concept_row:
                return None
            
            concept_id = concept_row['concept_id']
            
            # Build comprehensive JSON structure
            export_data = {
                "concept": {
                    "name": concept_row['name'],
                    "description": concept_row['description'],
                    "primary_strongs": json.loads(concept_row['primary_strongs'] or '[]'),
                    "extended_strongs": json.loads(concept_row['extended_strongs'] or '[]'),
                    "semantic_keywords": json.loads(concept_row['semantic_keywords'] or '[]'),
                    "hebrew_terms": json.loads(concept_row['hebrew_terms'] or '[]'),
                    "greek_terms": json.loads(concept_row['greek_terms'] or '[]')
                },
                "statistics": {},
                "verses": {
                    "lexical": [],
                    "semantic": [],
                    "thematic": []
                },
                "themes": {},
                "relationships": {},
                "study_notes": {},
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "processing_version": concept_row['processing_version']
                }
            }
            
            # Get statistics
            cursor.execute("""
                SELECT * FROM concept_processing_stats WHERE concept_id = ?
            """, (concept_id,))
            stats_row = cursor.fetchone()
            if stats_row:
                export_data["statistics"] = {
                    "total_matches": stats_row['total_matches'],
                    "lexical_matches": stats_row['lexical_matches'],
                    "semantic_matches": stats_row['semantic_matches'],
                    "books_covered": stats_row['books_covered'],
                    "last_updated": stats_row['last_updated']
                }
            
            # Get verses by type
            for match_type in ['lexical', 'semantic', 'thematic']:
                cursor.execute("""
                    SELECT * FROM verse_concepts 
                    WHERE concept_id = ? AND match_type = ?
                    ORDER BY confidence DESC
                """, (concept_id, match_type))
                
                verses = []
                for row in cursor.fetchall():
                    verse_data = {
                        "verse_id": row['verse_id'],
                        "book": row['book'],
                        "chapter": row['chapter'],
                        "verse": row['verse'],
                        "text": row['verse_text'],
                        "confidence": row['confidence']
                    }
                    
                    if row['original_text']:
                        verse_data['original_text'] = row['original_text']
                    if row['strongs_matched']:
                        verse_data['strongs_matched'] = json.loads(row['strongs_matched'])
                    if match_type == 'semantic':
                        verse_data['semantic_score'] = row['semantic_score']
                        if row['ollama_validated']:
                            verse_data['ollama_validation'] = {
                                'confidence': row['ollama_confidence'],
                                'reasoning': row['ollama_reasoning']
                            }
                    
                    verses.append(verse_data)
                
                export_data["verses"][match_type] = verses
            
            # Get themes
            cursor.execute("""
                SELECT * FROM concept_themes 
                WHERE concept_id = ? 
                ORDER BY display_order
            """, (concept_id,))
            
            for row in cursor.fetchall():
                export_data["themes"][row['theme_name']] = {
                    "description": row['description'],
                    "verse_ids": json.loads(row['verse_ids'] or '[]'),
                    "key_insights": row['key_insights'],
                    "theological_significance": row['theological_significance'],
                    "practical_application": row['practical_application']
                }
            
            # Get relationships
            cursor.execute("""
                SELECT cr.*, c.name as related_name
                FROM concept_relationships cr
                JOIN concepts c ON cr.related_concept_id = c.concept_id
                WHERE cr.concept_id = ?
            """, (concept_id,))
            
            for row in cursor.fetchall():
                rel_type = row['relationship_type']
                if rel_type not in export_data["relationships"]:
                    export_data["relationships"][rel_type] = []
                export_data["relationships"][rel_type].append({
                    "concept": row['related_name'],
                    "strength": row['strength'],
                    "description": row['description']
                })
            
            # Get study notes
            cursor.execute("""
                SELECT * FROM concept_study_notes WHERE concept_id = ?
            """, (concept_id,))
            
            for row in cursor.fetchall():
                export_data["study_notes"][row['note_type']] = {
                    "content": row['content'],
                    "references": json.loads(row['references'] or '[]')
                }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return export_data
    
    def get_verses_for_concept(self, concept_name: str, 
                              min_confidence: float = 0.5,
                              match_types: List[str] = None) -> List[Dict]:
        """Quick lookup for verses related to a concept."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if match_types is None:
                match_types = ['lexical', 'semantic', 'thematic']
            
            placeholders = ','.join(['?' for _ in match_types])
            query = f"""
                SELECT vc.*, c.name as concept_name
                FROM verse_concepts vc
                JOIN concepts c ON vc.concept_id = c.concept_id
                WHERE c.name = ?
                AND vc.confidence >= ?
                AND vc.match_type IN ({placeholders})
                ORDER BY vc.confidence DESC, vc.semantic_score DESC
            """
            
            cursor.execute(query, [concept_name, min_confidence] + match_types)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_related_concepts(self, concept_name: str) -> Dict[str, List[str]]:
        """Get all concepts related to the given concept."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cr.relationship_type, c2.name
                FROM concept_relationships cr
                JOIN concepts c1 ON cr.concept_id = c1.concept_id
                JOIN concepts c2 ON cr.related_concept_id = c2.concept_id
                WHERE c1.name = ?
                ORDER BY cr.strength DESC
            """, (concept_name,))
            
            relationships = {}
            for row in cursor.fetchall():
                rel_type = row[0]
                if rel_type not in relationships:
                    relationships[rel_type] = []
                relationships[rel_type].append(row[1])
            
            return relationships
    
    def search_concepts_by_verse(self, verse_id: str) -> List[Dict]:
        """Find all concepts associated with a specific verse."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.name, vc.match_type, vc.confidence, 
                       vc.semantic_score, vc.ollama_confidence
                FROM verse_concepts vc
                JOIN concepts c ON vc.concept_id = c.concept_id
                WHERE vc.verse_id = ?
                ORDER BY vc.confidence DESC
            """, (verse_id,))
            
            return [dict(row) for row in cursor.fetchall()]