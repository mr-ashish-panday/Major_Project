"""
ScholarFormer Tokenizer — BPE tokenizer wrapper with section markers.

Wraps HuggingFace's AutoTokenizer (using Phi-3's vocabulary) and adds
custom section marker tokens for Section-Aware Positional Encoding.

Section markers:
    <|abstract|>, <|introduction|>, <|methods|>, <|results|>,
    <|discussion|>, <|conclusion|>, <|other|>
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Section detection patterns (regex)
SECTION_PATTERNS = {
    'abstract': re.compile(r'\b(abstract)\b', re.IGNORECASE),
    'introduction': re.compile(r'\b(introduction|1\.\s*introduction)\b', re.IGNORECASE),
    'methods': re.compile(r'\b(method|methodology|approach|experimental\s+setup|materials?\s+and\s+methods?)\b', re.IGNORECASE),
    'results': re.compile(r'\b(results?|experiments?|experimental\s+results?|findings)\b', re.IGNORECASE),
    'discussion': re.compile(r'\b(discussion|analysis|limitations?)\b', re.IGNORECASE),
    'conclusion': re.compile(r'\b(conclusion|summary|future\s+work|concluding\s+remarks?)\b', re.IGNORECASE),
}

# Section name to ID mapping
SECTION_TO_ID = {
    'abstract': 0,
    'introduction': 1,
    'methods': 2,
    'results': 3,
    'discussion': 4,
    'conclusion': 5,
    'other': 6,
}

# Custom section marker tokens
SECTION_MARKERS = [
    '<|abstract|>',
    '<|introduction|>',
    '<|methods|>',
    '<|results|>',
    '<|discussion|>',
    '<|conclusion|>',
    '<|other|>',
]


class ScholarFormerTokenizer:
    """
    Tokenizer for ScholarFormer.
    
    Wraps Phi-3's BPE tokenizer (32K vocab) and adds:
    1. Section marker tokens for section-aware encoding
    2. Section detection from raw paper text
    3. Utility methods for preparing training data
    
    Usage:
        tokenizer = ScholarFormerTokenizer()
        
        # Basic tokenization
        result = tokenizer.encode("This is a research abstract.", section="abstract")
        
        # Detect sections from raw text
        chunks_with_sections = tokenizer.detect_sections(paper_text)
        
        # Prepare for model input
        tokens, section_ids = tokenizer.prepare_input(text, section="methods")
    """
    
    def __init__(self, base_tokenizer_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize with Phi-3's tokenizer vocabulary.
        
        Args:
            base_tokenizer_name: HuggingFace tokenizer to use as base
        """
        logger.info(f"Loading base tokenizer: {base_tokenizer_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load {base_tokenizer_name}: {e}")
            logger.info("Falling back to GPT-2 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add section marker tokens
        new_tokens = [t for t in SECTION_MARKERS if t not in self.tokenizer.get_vocab()]
        if new_tokens:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
            logger.info(f"Added {num_added} section marker tokens to vocabulary")
        
        # Store section marker token IDs
        self.section_marker_ids = {}
        for marker in SECTION_MARKERS:
            self.section_marker_ids[marker] = self.tokenizer.convert_tokens_to_ids(marker)
        
        logger.info(f"Tokenizer ready: vocab_size={self.vocab_size}, "
                     f"section_markers={list(self.section_marker_ids.keys())}")
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id
    
    def encode(self, text: str, max_length: int = 1024,
               truncation: bool = True, 
               return_tensors: Optional[str] = None) -> dict:
        """
        Tokenize text using the base tokenizer.
        
        Args:
            text: Input text to tokenize
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length
            return_tensors: 'pt' for PyTorch tensors, None for lists
        
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            padding=False
        )
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def detect_section(self, text: str) -> str:
        """
        Detect the most likely section type from text content.
        
        Uses keyword matching to identify paper sections.
        Returns section name (string).
        """
        text_lower = text[:500].lower()  # Only check beginning
        
        # Check each section pattern
        for section_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(text_lower):
                return section_name
        
        return 'other'
    
    def prepare_input(self, text: str, section: str = 'other',
                      max_length: int = 1024,
                      return_tensors: str = 'pt') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare model input with section IDs.
        
        Args:
            text: Input text
            section: Section type ('abstract', 'methods', etc.)
            max_length: Maximum sequence length
            return_tensors: tensor type ('pt' for PyTorch)
        
        Returns:
            Tuple of (input_ids, section_ids) as tensors
        """
        # Tokenize
        encoded = self.encode(text, max_length=max_length, return_tensors=return_tensors)
        input_ids = encoded['input_ids']
        
        # Create matching section IDs
        section_id = SECTION_TO_ID.get(section, SECTION_TO_ID['other'])
        section_ids = torch.full_like(input_ids, fill_value=section_id)
        
        return input_ids, section_ids
    
    def prepare_chunked_input(self, chunks: List[Dict], 
                               max_length: int = 1024) -> List[Dict]:
        """
        Prepare a list of text chunks with detected sections.
        
        Args:
            chunks: List of dicts with at least a 'text' key
                    Optionally a 'section' key (will be auto-detected if missing)
        
        Returns:
            List of dicts with 'input_ids', 'section_ids', 'section_name'
        """
        prepared = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            section = chunk.get('section', self.detect_section(text))
            
            input_ids, section_ids = self.prepare_input(
                text, section=section, max_length=max_length
            )
            
            prepared.append({
                'input_ids': input_ids.squeeze(0),
                'section_ids': section_ids.squeeze(0),
                'section_name': section,
                'text': text
            })
        
        return prepared
    
    def batch_encode(self, texts: List[str], sections: Optional[List[str]] = None,
                     max_length: int = 1024, padding: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts with padding and section IDs.
        
        Args:
            texts: List of input texts
            sections: List of section types (one per text). Auto-detected if None.
            max_length: Maximum sequence length
            padding: Whether to pad to the same length
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'section_ids'
        """
        if sections is None:
            sections = [self.detect_section(t) for t in texts]
        
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=padding,
            return_tensors='pt'
        )
        
        # Create section IDs matching input_ids shape
        batch_size, seq_len = encoded['input_ids'].shape
        section_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        for i, section in enumerate(sections):
            section_id = SECTION_TO_ID.get(section, SECTION_TO_ID['other'])
            # Only fill non-padding positions
            mask = encoded['attention_mask'][i].bool()
            section_ids[i][mask] = section_id
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'section_ids': section_ids,
        }
    
    def save(self, path: str):
        """Save the tokenizer to disk."""
        self.tokenizer.save_pretrained(path)
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ScholarFormerTokenizer':
        """Load a saved tokenizer."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
            instance.tokenizer.pad_token_id = instance.tokenizer.eos_token_id
        
        instance.section_marker_ids = {}
        for marker in SECTION_MARKERS:
            instance.section_marker_ids[marker] = instance.tokenizer.convert_tokens_to_ids(marker)
        
        logger.info(f"Tokenizer loaded from {path}")
        return instance
