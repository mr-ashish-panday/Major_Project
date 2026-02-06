"""
VectorStoreAgent - Manages paper embeddings with FAISS for semantic retrieval.
Part of the ScholarMind multi-agent system.
"""

import os
import json
import logging
import hashlib
import pickle
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStoreAgent:
    """
    Manages a persistent vector database for research papers using FAISS.
    Enables semantic search to retrieve relevant context for RAG.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.vector_db_path = config.get('vector_db_path', './vector_store')
        
        # Create directory
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(self.vector_db_path, 'faiss.index')
        self.metadata_path = os.path.join(self.vector_db_path, 'metadata.pkl')
        
        # Initialize embedding model
        embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.index = None
        self.documents = []  # List of (content, metadata) tuples
        self._load_or_create_index()
        
        logger.info(f"VectorStoreAgent initialized. Index has {self.index.ntotal} documents.")
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded existing index with {self.index.ntotal} documents.")
            except Exception as e:
                logger.warning(f"Could not load index: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine sim after normalization)
        self.documents = []
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved index with {self.index.ntotal} documents.")
    
    def _generate_chunk_id(self, paper_title: str, chunk_index: int) -> str:
        """Generate a unique ID for each chunk."""
        content = f"{paper_title}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_papers(self, processed_papers: List[Dict]) -> int:
        """
        Add preprocessed papers to the vector database.
        
        Args:
            processed_papers: List of papers with 'metadata' and 'chunks' fields
            
        Returns:
            Number of chunks added
        """
        if not processed_papers:
            logger.warning("No papers to add to vector store.")
            return 0
        
        all_texts = []
        all_metadatas = []
        
        for paper in processed_papers:
            try:
                metadata = paper.get('metadata', {})
                chunks = paper.get('chunks', [])
                
                if not chunks:
                    continue
                
                paper_title = metadata.get('title', 'Unknown')
                
                for i, chunk in enumerate(chunks):
                    if not chunk or len(chunk.strip()) < 50:
                        continue
                    
                    all_texts.append(chunk)
                    all_metadatas.append({
                        'title': paper_title,
                        'authors': ', '.join(metadata.get('authors', [])) if isinstance(metadata.get('authors'), list) else str(metadata.get('authors', '')),
                        'date': str(metadata.get('date', '')),
                        'url': metadata.get('url', ''),
                        'chunk_index': i,
                        'abstract': metadata.get('abstract', '')[:500]
                    })
                    
            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                continue
        
        if not all_texts:
            logger.warning("No valid chunks to add.")
            return 0
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents
        for i, (text, meta) in enumerate(zip(all_texts, all_metadatas)):
            self.documents.append({'content': text, 'metadata': meta})
        
        # Save
        self._save_index()
        
        logger.info(f"Added {len(all_texts)} chunks. Total: {self.index.ntotal}")
        return len(all_texts)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Semantic search for relevant paper chunks.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with 'content', 'metadata', and 'score'
        """
        if top_k is None:
            top_k = self.config.get('retrieval_top_k', 5)
        
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(score)
                    })
            
            logger.info(f"Search returned {len(results)} results.")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_documents': self.index.ntotal if self.index else 0,
            'vector_db_path': self.vector_db_path,
            'embedding_model': self.config.get('embedding_model', 'unknown'),
            'embedding_dim': self.embedding_dim
        }
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        logger.warning("Clearing vector store!")
        self._create_new_index()
        self._save_index()
        logger.info("Vector store cleared.")
    
    # Alias for compatibility
    @property
    def collection(self):
        """Compatibility property for code expecting ChromaDB-style interface."""
        class _Collection:
            def __init__(self, parent):
                self._parent = parent
            def count(self):
                return self._parent.index.ntotal if self._parent.index else 0
        return _Collection(self)
