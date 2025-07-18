import fitz  # PyMuPDF
import re
import logging
from transformers import AutoTokenizer
from typing import List, Dict

logger = logging.getLogger(__name__)

class PreprocessorAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], token=os.environ.get('HF_TOKEN'))

    def preprocess(self, papers: List[Dict]) -> List[Dict]:
        processed = []
        for paper in papers:
            try:
                doc = fitz.open(paper['pdf_path'])
                text = ''
                for page in doc:
                    text += page.get_text()
                doc.close()

                # Clean text
                text = re.sub(r'\[.*?\]', '', text)  # Citations
                text = re.sub(r'Figure \d+.*?\n', '', text)  # Figures
                text = re.sub(r'Equation \d+.*?\n', '', text)  # Equations
                text = re.sub(r'References.*$', '', text, flags=re.DOTALL)  # Bibliography

                # Chunking (token-based for accuracy)
                tokens = self.tokenizer.encode(text)
                chunks = [self.tokenizer.decode(tokens[i:i+512]) for i in range(0, len(tokens), 512)]

                processed.append({
                    'metadata': {
                        'title': paper['title'],
                        'authors': paper['authors'],
                        'date': paper['date'],
                        'url': paper['url'],
                        'abstract': paper['abstract']
                    },
                    'chunks': chunks
                })
            except Exception as e:
                logger.error(f"Failed to preprocess {paper['title']}: {e}")
        logger.info(f"Preprocessed {len(processed)} papers.")
        return processed