import pytest
from datetime import datetime, timezone

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_topic_rejects_empty_query() -> None:
    response = client.post("/analyze-topic", json={"query": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_topic_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.api import routes

    def _mock_fetch_arxiv_papers(query: str, max_results: int) -> list[dict]:
        return [
            {
                "paper_id": "p1",
                "title": "MMMU benchmark",
                "abstract": "We evaluate on MMMU dataset.",
                "authors": ["A"],
                "published_date": "2024-01-01T00:00:00Z",
                "published_year": 2024,
                "updated_date": "2024-01-02T00:00:00Z",
                "primary_category": "cs.CV",
                "categories": ["cs.CV"],
                "arxiv_url": "https://arxiv.org/abs/0001.00001",
                "pdf_url": "https://arxiv.org/pdf/0001.00001.pdf",
            }
        ]

    monkeypatch.setattr(routes, "fetch_arxiv_papers", _mock_fetch_arxiv_papers)

    response = client.post(
        "/analyze-topic",
        json={"query": "multimodal cultural benchmark", "max_results": 5, "granularity": "year"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "multimodal cultural benchmark"
    assert "summary" in payload
    assert set(payload["summary"].keys()) == {"total_papers", "year_range"}
    assert "papers" in payload
    assert "trend" in payload
    assert "category_distribution" in payload
    assert "top_keywords" in payload
    assert "top_entities" not in payload
    assert "evidence_table" not in payload


def test_analyze_topic_rejects_invalid_year_range() -> None:
    response = client.post(
        "/analyze-topic",
        json={"query": "multimodal", "start_year": 2026, "end_year": 2020},
    )
    assert response.status_code == 400
    assert "start_year" in response.json()["detail"]


def test_analyze_topic_filters_by_year_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.api import routes

    def _mock_fetch_arxiv_papers(query: str, max_results: int) -> list[dict]:
        return [
            {
                "paper_id": "p1",
                "title": "Older paper",
                "abstract": "Older abstract",
                "authors": ["A"],
                "published_date": "2023-01-01T00:00:00Z",
                "published_year": 2023,
                "updated_date": "2023-01-02T00:00:00Z",
                "primary_category": "cs.CV",
                "categories": ["cs.CV"],
                "arxiv_url": "https://arxiv.org/abs/0001.00001",
                "pdf_url": "https://arxiv.org/pdf/0001.00001.pdf",
            },
            {
                "paper_id": "p2",
                "title": "Newer paper",
                "abstract": "Newer abstract",
                "authors": ["B"],
                "published_date": "2025-01-01T00:00:00Z",
                "published_year": 2025,
                "updated_date": "2025-01-02T00:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "arxiv_url": "https://arxiv.org/abs/0002.00002",
                "pdf_url": "https://arxiv.org/pdf/0002.00002.pdf",
            },
        ]

    monkeypatch.setattr(routes, "fetch_arxiv_papers", _mock_fetch_arxiv_papers)

    response = client.post(
        "/analyze-topic",
        json={"query": "multimodal", "start_year": 2025, "end_year": 2025},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_papers"] == 1
    assert len(payload["papers"]) == 1
    assert payload["papers"][0]["published_year"] == 2025


def test_analyze_topic_default_year_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.api import routes

    current_year = datetime.now(timezone.utc).year
    inside_year = current_year - 3
    outside_year = current_year - 10

    def _mock_fetch_arxiv_papers(query: str, max_results: int) -> list[dict]:
        return [
            {
                "paper_id": "p1",
                "title": "Inside window",
                "abstract": "Inside abstract",
                "authors": ["A"],
                "published_date": f"{inside_year}-01-01T00:00:00Z",
                "published_year": inside_year,
                "updated_date": f"{inside_year}-01-02T00:00:00Z",
                "primary_category": "cs.CV",
                "categories": ["cs.CV"],
                "arxiv_url": "https://arxiv.org/abs/0001.00001",
                "pdf_url": "https://arxiv.org/pdf/0001.00001.pdf",
            },
            {
                "paper_id": "p2",
                "title": "Outside window",
                "abstract": "Outside abstract",
                "authors": ["B"],
                "published_date": f"{outside_year}-01-01T00:00:00Z",
                "published_year": outside_year,
                "updated_date": f"{outside_year}-01-02T00:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "arxiv_url": "https://arxiv.org/abs/0002.00002",
                "pdf_url": "https://arxiv.org/pdf/0002.00002.pdf",
            },
        ]

    monkeypatch.setattr(routes, "fetch_arxiv_papers", _mock_fetch_arxiv_papers)

    response = client.post("/analyze-topic", json={"query": "multimodal"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_papers"] == 1
    assert len(payload["papers"]) == 1
    assert payload["papers"][0]["published_year"] == inside_year