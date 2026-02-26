"""
RAGPipeline - Retrieval-Augmented Generation for research question answering.
Part of the ScholarMind multi-agent system.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from agents.vector_store import VectorStoreAgent

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline for research question answering with citations."""
    
    # Phi-3 special tokens
    SYS_START = chr(60) + "|system|" + chr(62)
    SYS_END = chr(60) + "|end|" + chr(62)
    USER_START = chr(60) + "|user|" + chr(62)
    ASST_START = chr(60) + "|assistant|" + chr(62)
    
    def __init__(self, config: Dict, model_path: Optional[str] = None):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.vector_store = VectorStoreAgent(config)
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load the fine-tuned model for generation."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            base_model_name = self.config.get("base_model", "microsoft/Phi-3-mini-4k-instruct")
            logger.info(f"Loading base model: {base_model_name}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN")
            )
            
            if os.path.exists(model_path):
                logger.info(f"Loading LoRA weights from {model_path}")
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                logger.warning("LoRA weights not found, using base model only")
                self.model = base_model
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN")
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _format_context(self, retrieved_docs: List[Dict]) -> Tuple[str, List[Dict]]:
        """Format retrieved documents into context with citations."""
        if not retrieved_docs:
            return "", []
        
        context_parts = []
        citations = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            if len(content) > 800:
                content = content[:800] + "..."
            
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Unknown")
            context_parts.append(f"[{i}] (Source: {title})\n{content}")
            citations.append({
                "id": i,
                "title": title,
                "authors": metadata.get("authors", "Unknown"),
                "url": metadata.get("url", ""),
                "score": doc.get("score", 0)
            })
        
        return "\n\n".join(context_parts), citations
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for the LLM with retrieved context using Phi-3 format."""
        system_msg = (
            "You are ScholarMind, an expert AI research assistant specializing in "
            "Large Language Models (LLMs), Natural Language Processing (NLP), and "
            "machine learning. Your role is to provide clear, accurate, and well-structured "
            "answers based on the provided research context. "
            "Always write your answer as a direct, conversational response. "
            "Do NOT output in Question/Answer format. "
            "Do NOT output metadata, numbers, or instruction labels. "
            "Cite sources using [1], [2], etc. when referencing specific papers."
        )
        
        if context:
            user_msg = (
                f"Based on the following research papers, answer my question.\n\n"
                f"RESEARCH CONTEXT:\n{context}\n\n"
                f"MY QUESTION: {query}\n\n"
                f"Provide a clear, direct answer:"
            )
        else:
            user_msg = f"{query}\n\nProvide a clear, direct answer:"
        
        prompt = f"{self.SYS_START}\n{system_msg}{self.SYS_END}\n"
        prompt += f"{self.USER_START}\n{user_msg}{self.SYS_END}\n"
        prompt += f"{self.ASST_START}\n"
        
        return prompt
    
    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove training data artifacts and noise from model output."""
        if not text:
            return ""
        
        # Remove leading garbage tokens (numbers, scores, metadata)
        # e.g., "6 9.36\nQuestion:..." or "🤖 6 9.36"
        text = re.sub(r'^[\s\d\.\,\;\:\!\?\#\*\-\_\=\+\~\`\|\\\/\@\$\%\^\&\(\)]*(?:\d+\.?\d*\s*)+', '', text)
        
        # Remove "Question: ... Answer: ..." wrapper if the model dumps Q&A format
        qa_match = re.search(r'(?:Answer|A)\s*[:\.]\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if qa_match and 'Question:' in text[:200]:
            text = qa_match.group(1).strip()
        
        # Remove instruction leakage patterns
        patterns_to_remove = [
            r'Instruction\s*\d+\s*\(.*?\)',       # "Instruction 2 (More Diff0"
            r'<\|.*?\|>',                           # Any remaining special tokens
            r'\[INST\].*?\[/INST\]',                # Instruction wrappers
            r'###\s*(?:Human|System|User)\s*:.*?$', # Role labels at end
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove trailing incomplete sentences (cut off at last period/question mark)
        text = text.strip()
        if text and text[-1] not in '.!?\n"\'':
            last_sentence_end = max(
                text.rfind('.'),
                text.rfind('!'),
                text.rfind('?'),
            )
            if last_sentence_end > len(text) * 0.3:  # Only trim if we keep >30%
                text = text[:last_sentence_end + 1]
        
        return text.strip()
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Answer a research question using RAG.
        
        Args:
            question: The research question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with 'answer', 'citations', and 'retrieved_docs'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, top_k)
        
        # Format context and get citations
        context, citations = self._format_context(retrieved_docs)
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate response
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072  # Use more of the 4k context window
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 512),
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract only the generated tokens (not the prompt)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up training data artifacts
            answer = self._clean_response(answer)
            
            # Fallback if cleaned answer is too short
            if not answer or len(answer) < 15:
                # Try the full decode approach as backup
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                if self.ASST_START in full_response:
                    parts = full_response.split(self.ASST_START)
                    if len(parts) > 1:
                        answer = parts[-1]
                        if self.SYS_END in answer:
                            answer = answer.split(self.SYS_END)[0]
                        answer = self._clean_response(answer)
            
            if not answer or len(answer) < 15:
                answer = "I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_docs": retrieved_docs,
            "query": question
        }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "model_loaded": self.is_loaded,
            "base_model": self.config.get("base_model", "unknown"),
            "vector_store": self.vector_store.get_stats()
        }
