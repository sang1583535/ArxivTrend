from backend.services.arxiv_service import fetch_arxiv_papers


def search_papers(query: str, max_results: int = 10):
    return fetch_arxiv_papers(query=query, max_results=max_results)
