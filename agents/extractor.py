import arxiv
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ExtractorAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.arxiv_client = arxiv.Client()

    def extract(self) -> List[Dict]:
        query = ' OR '.join(self.config['arxiv_keywords'])
        search = arxiv.Search(query=query, max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for result in self.arxiv_client.results(search):
            pdf_path = os.path.join(self.config['data_dir'], f"{result.entry_id.split('/')[-1]}.pdf")
            try:
                result.download_pdf(dirpath=self.config['data_dir'], filename=os.path.basename(pdf_path))
                papers.append({
                    'title': result.title,
                    'authors': [a.name for a in result.authors],
                    'date': result.published,
                    'url': result.pdf_url,
                    'pdf_path': pdf_path,
                    'abstract': result.summary
                })
            except Exception as e:
                logger.error(f"Failed to download {result.title}: {e}")
        logger.info(f"Extracted {len(papers)} papers.")
        return papers