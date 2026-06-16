import pytest

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