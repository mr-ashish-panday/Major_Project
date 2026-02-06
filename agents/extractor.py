"""
ExtractorAgent - Extracts papers from arXiv.
Part of the ScholarMind multi-agent system.

IMPROVED: Added retry logic with exponential backoff for arXiv HTTP 500 errors.
"""

import arxiv
import os
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# arXiv rate limit: 3 seconds between requests recommended
ARXIV_DELAY = 3.0
MAX_RETRIES = 3
INITIAL_BACKOFF = 10  # seconds


class ExtractorAgent:
    def __init__(self, config: Dict):
        self.config = config
        # Configure client with longer timeout and delay
        self.arxiv_client = arxiv.Client(
            page_size=50,  # Smaller pages = less likely to timeout
            delay_seconds=ARXIV_DELAY,
            num_retries=3
        )

    def extract(self) -> List[Dict]:
        """
        Extract papers from arXiv with robust error handling.
        
        Returns empty list on persistent failures instead of crashing.
        """
        query = ' OR '.join(self.config['arxiv_keywords'])
        max_results = self.config.get('arxiv_max_results', 200)
        start_offset = self.config.get('arxiv_offset', self.config.get('arxiv_start_offset', 0))
        
        logger.info(f"Fetching {max_results} papers starting from offset {start_offset}")
        
        search = arxiv.Search(
            query=query, 
            max_results=max_results + start_offset,  # Fetch enough to skip offset
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        
        # Retry loop for HTTP 500 errors
        for attempt in range(MAX_RETRIES):
            try:
                count = 0
                for result in self.arxiv_client.results(search):
                    count += 1
                    # Skip papers before the offset
                    if count <= start_offset:
                        continue
                    
                    pdf_path = os.path.join(
                        self.config['data_dir'], 
                        f"{result.entry_id.split('/')[-1]}.pdf"
                    )
                    
                    try:
                        result.download_pdf(
                            dirpath=self.config['data_dir'], 
                            filename=os.path.basename(pdf_path)
                        )
                        papers.append({
                            'title': result.title,
                            'authors': [a.name for a in result.authors],
                            'date': result.published,
                            'url': result.pdf_url,
                            'pdf_path': pdf_path,
                            'abstract': result.summary
                        })
                    except Exception as e:
                        logger.warning(f"Failed to download {result.title}: {e}")
                        continue
                
                # Success! Break out of retry loop
                logger.info(f"Extracted {len(papers)} papers.")
                return papers
                
            except arxiv.HTTPError as e:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"arXiv HTTP error (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Waiting {backoff} seconds before retry...")
                    time.sleep(backoff)
                else:
                    logger.error("arXiv API failed after all retries. Returning empty list.")
                    return []
                    
            except Exception as e:
                logger.error(f"Unexpected error during extraction: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_BACKOFF)
                else:
                    return []
        
        return papers