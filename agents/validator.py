"""
ValidatorAgent - Validates and filters research papers.
Part of the ScholarMind multi-agent system.

IMPROVED: Better multi-word keyword matching, lower relevance threshold.
"""

import os
import re
import logging
import pickle
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Validates papers for:
    1. Minimum length
    2. Deduplication (using embeddings)
    3. Relevance to keywords
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_content_length = 200  # Lowered from 300 - more permissive
        self.similarity_threshold = 0.95  # High - only exact duplicates
        self.min_relevance_density = 0.0003  # Very permissive (0.03%)
        
        # Load embedding model
        embedding_model = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Historical embeddings for deduplication
        self.historical_embeddings = []
        self.embedding_cache_path = os.path.join(
            config.get('logs_dir', './logs'),
            'paper_embeddings.pkl'
        )
        self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load historical embeddings from cache."""
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    self.historical_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.historical_embeddings)} historical embeddings.")
            except Exception as e:
                logger.warning(f"Could not load embeddings cache: {e}")
                self.historical_embeddings = []
    
    def _save_embeddings(self) -> None:
        """Save embeddings to cache."""
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(self.historical_embeddings, f)
    
    def _check_length(self, paper: Dict) -> bool:
        """Check if paper has sufficient content."""
        chunks = paper.get('chunks', [])
        total_length = sum(len(chunk) for chunk in chunks)
        
        if total_length < self.min_content_length:
            logger.debug(f"Paper too short: {total_length} chars")
            return False
        return True
    
    def _check_duplicate(self, paper: Dict) -> bool:
        """Check if paper is a duplicate of existing papers."""
        chunks = paper.get('chunks', [])
        if not chunks:
            return False
        
        # Create paper embedding from first few chunks
        paper_text = ' '.join(chunks[:5])[:2000]  # Use first 2000 chars
        paper_embedding = self.embedding_model.encode(paper_text, convert_to_tensor=True)
        
        # FIXED: Convert to float32 to avoid dtype mismatch (Half vs float)
        paper_embedding = paper_embedding.float()
        
        # Check similarity with historical embeddings
        for hist_emb in self.historical_embeddings:
            # Ensure tensors are on the same device and dtype
            hist_emb = hist_emb.to(paper_embedding.device).float()
            
            similarity = util.cos_sim(paper_embedding, hist_emb)[0][0].item()
            if similarity > self.similarity_threshold:
                logger.debug(f"Duplicate detected: {similarity:.2f} similarity")
                return False
        
        # Add to historical embeddings (store as float32)
        self.historical_embeddings.append(paper_embedding.cpu())
        return True
    
    def _check_relevance(self, paper: Dict) -> bool:
        """
        Check if paper is relevant to our keywords.
        
        IMPROVED: Better multi-word phrase matching using regex.
        """
        keywords = self.config.get('arxiv_keywords', [])
        if not keywords:
            return True  # No keywords to filter by
        
        chunks = paper.get('chunks', [])
        paper_text = ' '.join(chunks).lower()
        total_words = len(paper_text.split())
        
        if total_words == 0:
            return False
        
        # IMPROVED: Count keyword occurrences with proper phrase matching
        keyword_count = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if ' ' in kw_lower:
                # Multi-word phrases: exact phrase match
                keyword_count += paper_text.count(kw_lower)
            else:
                # Single words: word boundary matching to avoid partial matches
                matches = re.findall(r'\b' + re.escape(kw_lower) + r'\b', paper_text)
                keyword_count += len(matches)
        
        density = keyword_count / total_words
        
        if density < self.min_relevance_density:
            logger.debug(f"Low relevance: {density:.5f} (threshold: {self.min_relevance_density})")
            return False
        
        return True
    
    def validate(self, processed: List[Dict]) -> List[Dict]:
        """
        Validate and filter preprocessed papers.
        
        Args:
            processed: List of preprocessed papers
            
        Returns:
            List of validated papers
        """
        validated = []
        
        for paper in processed:
            title = paper.get('metadata', {}).get('title', 'Unknown')
            
            # Check length
            if not self._check_length(paper):
                logger.info(f"Discarded (too short): {title[:50]}...")
                continue
            
            # Check relevance
            if not self._check_relevance(paper):
                logger.info(f"Discarded (low relevance): {title[:50]}...")
                continue
            
            # Check for duplicates
            if not self._check_duplicate(paper):
                logger.info(f"Discarded (duplicate): {title[:50]}...")
                continue
            
            validated.append(paper)
            logger.info(f"Validated: {title[:50]}...")
        
        # Save updated embeddings
        self._save_embeddings()
        
        # Log validation statistics
        pass_rate = len(validated) / len(processed) * 100 if processed else 0
        logger.info(f"Validated {len(validated)} out of {len(processed)} papers ({pass_rate:.1f}% pass rate)")
        return validated
    
    def reset_history(self) -> None:
        """Clear historical embeddings (for fresh start)."""
        self.historical_embeddings = []
        if os.path.exists(self.embedding_cache_path):
            os.remove(self.embedding_cache_path)
        logger.info("Embedding history cleared.")