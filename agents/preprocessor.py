"""
PreprocessorAgent - Extracts and processes text from research papers.
Part of the ScholarMind multi-agent system.
"""

import os
import re
import logging
from typing import List, Dict

import fitz  # PyMuPDF
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class PreprocessorAgent:
    """
    Preprocesses research papers:
    1. Extract text from PDFs
    2. Clean text (remove citations, figures, etc.)
    3. Chunk text for training and embedding
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = 512  # tokens per chunk
        self.chunk_overlap = 50  # overlap between chunks
        
        # Load tokenizer for accurate chunking
        base_model = config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                token=os.environ.get('HF_TOKEN')
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {base_model}: {e}")
            # Fallback to GPT-2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing noise."""
        # Remove citation brackets [1], [2-4], etc.
        text = re.sub(r'\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]', '', text)
        
        # Remove figure/table references
        text = re.sub(r'(?:Figure|Fig\.|Table)\s*\d+\.?.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        
        # Remove equation references
        text = re.sub(r'(?:Equation|Eq\.)\s*\d+\.?', '', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove references/bibliography section
        text = re.sub(r'(?:References|Bibliography|REFERENCES).*$', '', text, flags=re.DOTALL)
        
        # Remove acknowledgments section
        text = re.sub(r'(?:Acknowledgments?|ACKNOWLEDGMENTS?).*?(?=\n[A-Z]|\Z)', '', text, flags=re.DOTALL)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for training."""
        if not text:
            return []
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            if len(chunk_text.strip()) > 50:  # Only keep meaningful chunks
                chunks.append(chunk_text)
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < len(tokens) else end
        
        return chunks
    
    def preprocess(self, papers: List[Dict]) -> List[Dict]:
        """
        Preprocess a list of papers.
        
        Args:
            papers: List of paper dictionaries with 'pdf_path', 'title', etc.
            
        Returns:
            List of processed papers with 'metadata' and 'chunks'
        """
        processed = []
        
        for paper in papers:
            try:
                pdf_path = paper.get('pdf_path', '')
                
                if not pdf_path or not os.path.exists(pdf_path):
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                
                # Extract text
                text = self._extract_text_from_pdf(pdf_path)
                
                if not text:
                    logger.warning(f"No text extracted from: {paper.get('title', 'Unknown')}")
                    continue
                
                # Clean text
                cleaned = self._clean_text(text)
                
                if len(cleaned) < 500:  # Skip very short papers
                    logger.warning(f"Paper too short after cleaning: {paper.get('title', 'Unknown')}")
                    continue
                
                # Chunk text
                chunks = self._chunk_text(cleaned)
                
                if not chunks:
                    logger.warning(f"No chunks created for: {paper.get('title', 'Unknown')}")
                    continue
                
                processed.append({
                    'metadata': {
                        'title': paper.get('title', 'Unknown'),
                        'authors': paper.get('authors', []),
                        'date': str(paper.get('date', '')),
                        'url': paper.get('url', ''),
                        'abstract': paper.get('abstract', ''),
                        'pdf_path': pdf_path
                    },
                    'chunks': chunks
                })
                
                logger.info(f"Preprocessed: {paper.get('title', 'Unknown')[:50]}... ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"Failed to preprocess {paper.get('title', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed)} out of {len(papers)} papers")
        return processed