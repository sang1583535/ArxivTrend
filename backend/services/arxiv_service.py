import xml.etree.ElementTree as ET
from datetime import datetime

import requests

ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _extract_paper_id(arxiv_url: str) -> str:
    suffix = arxiv_url.rstrip("/").split("/")[-1]
    return suffix or arxiv_url


def _safe_year(date_text: str) -> int:
    if not date_text:
        return 0
    try:
        return datetime.fromisoformat(date_text.replace("Z", "+00:00")).year
    except ValueError:
        return 0


def _parse_arxiv_response(response_text: str) -> list[dict]:
    root = ET.fromstring(response_text)
    normalized_papers: list[dict] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split())
        abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split())
        published_date = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
        updated_date = entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip()

        authors = [
            name.text.strip()
            for name in entry.findall("atom:author/atom:name", ATOM_NS)
            if name.text and name.text.strip()
        ]

        primary_category_node = entry.find("arxiv:primary_category", ATOM_NS)
        primary_category = ""
        if primary_category_node is not None:
            primary_category = primary_category_node.attrib.get("term", "").strip()

        categories: list[str] = []
        for cat in entry.findall("atom:category", ATOM_NS):
            term = cat.attrib.get("term", "").strip()
            if term and term not in categories:
                categories.append(term)

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            title_attr = link.attrib.get("title", "").lower()
            type_attr = link.attrib.get("type", "").lower()
            href = link.attrib.get("href", "").strip()
            if title_attr == "pdf" or type_attr == "application/pdf":
                pdf_url = href
                break

        if not pdf_url and arxiv_url:
            pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"

        normalized_papers.append(
            {
                "paper_id": _extract_paper_id(arxiv_url),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published_date": published_date,
                "published_year": _safe_year(published_date),
                "updated_date": updated_date,
                "primary_category": primary_category,
                "categories": categories,
                "arxiv_url": arxiv_url,
                "pdf_url": pdf_url,
            }
        )

    return normalized_papers


def fetch_arxiv_papers(query: str, max_results: int = 100) -> list[dict]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    params = {
        "search_query": f"all:{cleaned_query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_SEARCH_URL, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"arXiv request failed: {exc}") from exc

    try:
        return _parse_arxiv_response(response.text)
    except Exception as exc:
        raise RuntimeError(f"arXiv response parsing failed: {exc}") from exc
