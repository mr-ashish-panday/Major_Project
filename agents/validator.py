from sentence_transformers import SentenceTransformer, util
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ValidatorAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.historical_embeddings = []  # Persist to file for production

    def validate(self, processed: List[Dict]) -> List[Dict]:
        validated = []
        for paper in processed:
            total_length = sum(len(chunk) for chunk in paper['chunks'])
            if total_length < 300:
                logger.warning(f"Discarded {paper['metadata']['title']} due to insufficient length.")
                continue

            paper_text = ' '.join(paper['chunks'])
            paper_embedding = self.embedding_model.encode(paper_text)
            similarities = [util.cos_sim(paper_embedding, hist_emb)[0][0].item() for hist_emb in self.historical_embeddings]
            if any(sim > 0.95 for sim in similarities):
                logger.warning(f"Discarded {paper['metadata']['title']} due to duplication.")
                continue

            keywords = set(kw.lower() for kw in self.config['arxiv_keywords'])
            density = sum(paper_text.lower().count(kw) for kw in keywords) / len(paper_text.split())
            if density < 0.05:
                logger.warning(f"Discarded {paper['metadata']['title']} due to low relevance.")
                continue

            # Additional ethical/bias checks can be added here

            validated.append(paper)
            self.historical_embeddings.append(paper_embedding)
        logger.info(f"Validated {len(validated)} out of {len(processed)} papers.")
        return validated