#!/usr/bin/env python3
"""
Hybrid alignment training system that combines:
1. Parallel corpus alignment (IBM Model 1)
2. Monolingual embeddings with cross-lingual mapping
3. Ensemble scoring for higher accuracy

This approach leverages both direct translation evidence and 
contextual similarity for more robust word alignment.
"""

import argparse
import json
import logging
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abba.alignment.word_alignment import IBMModel1

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MonolingualEmbeddings:
    """Simple word embeddings based on co-occurrence patterns."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.vocab = set()
        self.word_to_id = {}
        self.id_to_word = {}
        self.cooccurrence = defaultdict(Counter)
        self.embeddings = {}
        
    def build_vocab(self, sentences: List[List[str]]):
        """Build vocabulary from tokenized sentences."""
        for sent in sentences:
            self.vocab.update(sent)
        
        # Create mappings
        for i, word in enumerate(sorted(self.vocab)):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            
        logger.info(f"Built vocabulary of {len(self.vocab)} words")
        
    def compute_cooccurrence(self, sentences: List[List[str]]):
        """Compute word co-occurrence statistics."""
        for sent in sentences:
            for i, word in enumerate(sent):
                # Look at surrounding context
                start = max(0, i - self.window_size)
                end = min(len(sent), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = sent[j]
                        self.cooccurrence[word][context_word] += 1
                        
    def compute_embeddings(self, dim: int = 100):
        """Compute simple embeddings from co-occurrence."""
        n_words = len(self.vocab)
        
        # Initialize with random values
        embedding_matrix = np.random.randn(n_words, dim) * 0.1
        
        # Use PMI (Pointwise Mutual Information) weighted by co-occurrence
        total_count = sum(sum(counter.values()) for counter in self.cooccurrence.values())
        
        for word, context_counts in self.cooccurrence.items():
            if word not in self.word_to_id:
                continue
                
            word_id = self.word_to_id[word]
            word_count = sum(context_counts.values())
            
            # Create embedding based on PMI with top contexts
            embedding = np.zeros(dim)
            top_contexts = context_counts.most_common(dim)
            
            for i, (context, count) in enumerate(top_contexts):
                if context in self.word_to_id:
                    # Simplified PMI
                    context_count = sum(self.cooccurrence[context].values())
                    pmi = np.log((count * total_count) / (word_count * context_count + 1e-10))
                    embedding[i] = max(0, pmi)  # Positive PMI
                    
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embedding_matrix[word_id] = embedding
            
        # Store as dictionary
        for word, word_id in self.word_to_id.items():
            self.embeddings[word] = embedding_matrix[word_id]
            
        logger.info(f"Computed {dim}-dimensional embeddings for {len(self.embeddings)} words")
        
    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words."""
        if word1 not in self.embeddings or word2 not in self.embeddings:
            return 0.0
            
        vec1 = self.embeddings[word1]
        vec2 = self.embeddings[word2]
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


class CrossLingualMapper:
    """Maps embeddings across languages using anchor words."""
    
    def __init__(self):
        self.mapping_matrix = None
        self.anchor_pairs = []
        
    def add_anchor_pairs(self, pairs: List[Tuple[str, str]]):
        """Add known translation pairs as anchors."""
        self.anchor_pairs.extend(pairs)
        
    def learn_mapping(self, source_embeddings: Dict[str, np.ndarray],
                     target_embeddings: Dict[str, np.ndarray]):
        """Learn linear mapping between embedding spaces."""
        # Collect anchor embeddings
        X, Y = [], []
        
        for src_word, tgt_word in self.anchor_pairs:
            if src_word in source_embeddings and tgt_word in target_embeddings:
                X.append(source_embeddings[src_word])
                Y.append(target_embeddings[tgt_word])
                
        if len(X) < 10:
            logger.warning(f"Only {len(X)} anchor pairs found, mapping may be unreliable")
            return
            
        X = np.array(X)
        Y = np.array(Y)
        
        # Learn mapping using least squares
        # W = Y @ X.T @ np.linalg.inv(X @ X.T)
        self.mapping_matrix = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        logger.info(f"Learned mapping from {len(X)} anchor pairs")
        
    def map_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Map embedding to target space."""
        if self.mapping_matrix is None:
            return embedding
        return self.mapping_matrix @ embedding
        
    def find_translation_candidates(self, word: str, source_embeddings: Dict[str, np.ndarray],
                                  target_embeddings: Dict[str, np.ndarray], 
                                  top_k: int = 10) -> List[Tuple[str, float]]:
        """Find translation candidates based on embedding similarity."""
        if word not in source_embeddings:
            return []
            
        # Map source embedding to target space
        src_embedding = source_embeddings[word]
        mapped_embedding = self.map_embedding(src_embedding)
        
        # Find nearest neighbors in target space
        candidates = []
        for tgt_word, tgt_embedding in target_embeddings.items():
            similarity = np.dot(mapped_embedding, tgt_embedding) / (
                np.linalg.norm(mapped_embedding) * np.linalg.norm(tgt_embedding) + 1e-10
            )
            candidates.append((tgt_word, similarity))
            
        # Return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


class HybridAlignmentModel:
    """Combines IBM Model 1 with cross-lingual embeddings."""
    
    def __init__(self):
        self.ibm_model = None
        self.source_embeddings = None
        self.target_embeddings = None
        self.cross_lingual_mapper = None
        
        # Weights for combining scores
        self.ibm_weight = 0.7
        self.embedding_weight = 0.3
        
    def train_parallel(self, parallel_pairs: List[Tuple[List[Dict], str]], 
                      iterations: int = 10):
        """Train IBM Model 1 on parallel data."""
        logger.info("Training IBM Model 1 on parallel corpus...")
        
        self.ibm_model = IBMModel1(
            use_morphology=True,
            use_strongs=True,
            use_lexicon=True
        )
        self.ibm_model.num_iterations = iterations
        self.ibm_model.train(parallel_pairs, verbose=True)
        
    def train_monolingual(self, source_sentences: List[List[str]], 
                         target_sentences: List[List[str]],
                         anchor_pairs: Optional[List[Tuple[str, str]]] = None):
        """Train monolingual embeddings and cross-lingual mapping."""
        logger.info("Training monolingual embeddings...")
        
        # Train source embeddings
        logger.info("Training source language embeddings...")
        source_model = MonolingualEmbeddings()
        source_model.build_vocab(source_sentences)
        source_model.compute_cooccurrence(source_sentences)
        source_model.compute_embeddings()
        self.source_embeddings = source_model.embeddings
        
        # Train target embeddings
        logger.info("Training target language embeddings...")
        target_model = MonolingualEmbeddings()
        target_model.build_vocab(target_sentences)
        target_model.compute_cooccurrence(target_sentences)
        target_model.compute_embeddings()
        self.target_embeddings = target_model.embeddings
        
        # Learn cross-lingual mapping
        if anchor_pairs:
            logger.info("Learning cross-lingual mapping...")
            self.cross_lingual_mapper = CrossLingualMapper()
            self.cross_lingual_mapper.add_anchor_pairs(anchor_pairs)
            self.cross_lingual_mapper.learn_mapping(
                self.source_embeddings, 
                self.target_embeddings
            )
        
    def align_with_ensemble(self, source_words: List[str], target_text: str,
                           threshold: float = 0.1) -> List[Tuple[int, int, float, str]]:
        """Align using ensemble of both models."""
        import re
        
        # Tokenize target
        target_words = re.findall(r'\w+|[^\w\s]', target_text.lower())
        
        alignments = []
        
        for src_idx, src_word in enumerate(source_words):
            best_score = 0.0
            best_tgt_idx = -1
            best_method = ""
            
            for tgt_idx, tgt_word in enumerate(target_words):
                # IBM Model 1 score
                ibm_score = 0.0
                if self.ibm_model and src_word in self.ibm_model.source_vocab:
                    ibm_score = self.ibm_model.trans_probs[src_word][tgt_word]
                
                # Embedding similarity score
                embed_score = 0.0
                if self.cross_lingual_mapper and self.source_embeddings and self.target_embeddings:
                    candidates = self.cross_lingual_mapper.find_translation_candidates(
                        src_word, self.source_embeddings, self.target_embeddings, top_k=20
                    )
                    for cand_word, cand_score in candidates:
                        if cand_word == tgt_word:
                            embed_score = cand_score
                            break
                
                # Combine scores
                combined_score = (self.ibm_weight * ibm_score + 
                                self.embedding_weight * embed_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_tgt_idx = tgt_idx
                    best_method = f"IBM:{ibm_score:.3f},Embed:{embed_score:.3f}"
            
            if best_score > threshold and best_tgt_idx >= 0:
                alignments.append((src_idx, best_tgt_idx, best_score, best_method))
                
        return alignments
    
    def save_model(self, output_dir: Path, model_name: str):
        """Save the hybrid model."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save IBM model
        if self.ibm_model:
            self.ibm_model.save_model(output_dir / f"{model_name}_ibm.json")
            
        # Save embeddings and mappings
        model_data = {
            'source_embeddings': {w: v.tolist() for w, v in (self.source_embeddings or {}).items()},
            'target_embeddings': {w: v.tolist() for w, v in (self.target_embeddings or {}).items()},
            'weights': {
                'ibm': self.ibm_weight,
                'embedding': self.embedding_weight
            }
        }
        
        if self.cross_lingual_mapper and self.cross_lingual_mapper.mapping_matrix is not None:
            model_data['mapping_matrix'] = self.cross_lingual_mapper.mapping_matrix.tolist()
            model_data['anchor_pairs'] = self.cross_lingual_mapper.anchor_pairs
            
        with open(output_dir / f"{model_name}_embeddings.json", 'w', encoding='utf-8') as f:
            json.dump(model_data, f)
            
        logger.info(f"Saved hybrid model to {output_dir}")


def extract_anchor_pairs_from_enriched(enriched_dir: Path, 
                                      min_confidence: float = 0.5) -> List[Tuple[str, str]]:
    """Extract high-confidence word pairs from enriched Bible data."""
    anchor_pairs = []
    pair_counts = Counter()
    
    # Look for aligned verses
    for book_file in enriched_dir.glob("*.json"):
        if book_file.name.startswith("_"):
            continue
            
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
            
        for chapter in book_data.get('chapters', []):
            for verse in chapter.get('verses', []):
                # Check alignments
                alignments = verse.get('alignments', {})
                for trans_code, align_list in alignments.items():
                    for alignment in align_list:
                        if alignment.get('confidence', 0) >= min_confidence:
                            src_word = alignment.get('source_word', '')
                            tgt_phrase = alignment.get('target_phrase', '')
                            
                            # Extract single target word if possible
                            tgt_words = tgt_phrase.strip().split()
                            if len(tgt_words) == 1:
                                pair = (src_word, tgt_words[0])
                                pair_counts[pair] += 1
                                
    # Use pairs that appear multiple times
    for pair, count in pair_counts.items():
        if count >= 3:  # Appears at least 3 times
            anchor_pairs.append(pair)
            
    logger.info(f"Extracted {len(anchor_pairs)} anchor pairs from enriched data")
    return anchor_pairs


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for any language."""
    import re
    # Split on whitespace and punctuation, keep everything
    tokens = re.findall(r'\S+', text)
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid alignment models combining parallel and monolingual approaches"
    )
    
    parser.add_argument(
        '--enriched-dir',
        type=Path,
        required=True,
        help='Directory containing enriched Bible export with alignments'
    )
    
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        required=True,
        help='Directory containing language corpus files (heb_bhs.json, grc_na28.json, etc.)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/alignment/hybrid'),
        help='Directory to save hybrid models'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='IBM Model 1 iterations'
    )
    
    parser.add_argument(
        '--languages',
        nargs=2,
        default=['hebrew', 'english'],
        help='Language pair to train (e.g., hebrew english)'
    )
    
    args = parser.parse_args()
    
    # Determine which files to use
    if args.languages[0] == 'hebrew':
        source_corpus = args.corpus_dir / 'heb_bhs.json'
        source_field = 'hebrew'
    elif args.languages[0] == 'greek':
        source_corpus = args.corpus_dir / 'grc_na28.json' 
        source_field = 'greek'
    else:
        logger.error(f"Unknown source language: {args.languages[0]}")
        return 1
        
    # Load enriched data for parallel training
    logger.info("Loading enriched data for parallel training...")
    parallel_pairs = []
    source_sentences = []
    target_sentences = []
    
    # Process enriched export
    for book_file in sorted(args.enriched_dir.glob("*.json")):
        if book_file.name.startswith("_"):
            continue
            
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
            
        for chapter in book_data.get('chapters', []):
            for verse in chapter.get('verses', []):
                # Get source words
                if source_field == 'hebrew':
                    source_words = verse.get('hebrew_words', [])
                    source_text = verse.get('hebrew_text', '')
                else:
                    source_words = verse.get('greek_words', [])
                    source_text = verse.get('greek_text', '')
                    
                if not source_words:
                    continue
                    
                # Get English translation (try both cases)
                translations = verse.get('translations', {})
                eng_text = translations.get('ENG_KJV', '') or translations.get('eng_kjv', '')
                if eng_text:
                    # For parallel training
                    parallel_pairs.append((source_words, eng_text))
                    
                    # For monolingual training
                    if source_text:
                        source_sentences.append(tokenize_text(source_text))
                    target_sentences.append(tokenize_text(eng_text))
    
    logger.info(f"Loaded {len(parallel_pairs)} parallel pairs")
    logger.info(f"Loaded {len(source_sentences)} source sentences")
    logger.info(f"Loaded {len(target_sentences)} target sentences")
    
    # Extract anchor pairs
    anchor_pairs = extract_anchor_pairs_from_enriched(args.enriched_dir)
    
    # Train hybrid model
    model = HybridAlignmentModel()
    
    # Train parallel component
    if parallel_pairs:
        model.train_parallel(parallel_pairs, iterations=args.iterations)
    
    # Train monolingual components
    if source_sentences and target_sentences:
        model.train_monolingual(source_sentences, target_sentences, anchor_pairs)
    
    # Save model
    model_name = f"{args.languages[0]}_{args.languages[1]}_hybrid"
    model.save_model(args.output_dir, model_name)
    
    # Test on a few examples
    logger.info("\nTesting hybrid alignment on sample verses...")
    for i in range(min(3, len(parallel_pairs))):
        source_words_data, target_text = parallel_pairs[i]
        source_words = [w.get('text', '') for w in source_words_data]
        
        alignments = model.align_with_ensemble(source_words, target_text)
        
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Target: {target_text[:80]}...")
        logger.info(f"Alignments: {len(alignments)}")
        for src_idx, tgt_idx, score, method in alignments[:5]:
            if src_idx < len(source_words):
                logger.info(f"  {source_words[src_idx]} -> [word {tgt_idx}] "
                           f"(score: {score:.3f}, {method})")
    
    logger.info("\nHybrid training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())