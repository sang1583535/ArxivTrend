from backend.services.stats_service import (
    compute_category_distribution,
    compute_summary,
    compute_topic_trend,
    limit_papers_by_year_range,
)


def _sample_papers() -> list[dict]:
    return [
        {
            "paper_id": "p1",
            "title": "MMMU: A multimodal benchmark for model evaluation",
            "abstract": "We evaluate LMMs on MMMU benchmark and MMBench dataset.",
            "authors": ["A"],
            "published_date": "2024-01-01T00:00:00Z",
            "published_year": 2024,
            "updated_date": "2024-01-02T00:00:00Z",
            "primary_category": "cs.CV",
            "categories": ["cs.CV", "cs.AI"],
            "arxiv_url": "https://arxiv.org/abs/0001.00001",
            "pdf_url": "https://arxiv.org/pdf/0001.00001.pdf",
        },
        {
            "paper_id": "p2",
            "title": "Cultural VQA with CMMMU",
            "abstract": "This benchmark uses CMMMU test set for multilingual evaluation.",
            "authors": ["B"],
            "published_date": "2025-02-01T00:00:00Z",
            "published_year": 2025,
            "updated_date": "2025-02-02T00:00:00Z",
            "primary_category": "cs.CL",
            "categories": ["cs.CL"],
            "arxiv_url": "https://arxiv.org/abs/0002.00002",
            "pdf_url": "https://arxiv.org/pdf/0002.00002.pdf",
        },
        {
            "paper_id": "p3",
            "title": "Vision language pretraining",
            "abstract": "We propose pretraining objectives for multimodal learning.",
            "authors": ["C"],
            "published_date": "2025-04-01T00:00:00Z",
            "published_year": 2025,
            "updated_date": "2025-04-02T00:00:00Z",
            "primary_category": "cs.CV",
            "categories": ["cs.CV"],
            "arxiv_url": "https://arxiv.org/abs/0003.00003",
            "pdf_url": "https://arxiv.org/pdf/0003.00003.pdf",
        },
    ]


def test_trend_counting() -> None:
    trend = compute_topic_trend(_sample_papers(), granularity="year")
    assert trend == [
        {"period": "2024", "paper_count": 1},
        {"period": "2025", "paper_count": 2},
    ]


def test_category_counting() -> None:
    distribution = compute_category_distribution(_sample_papers())
    assert distribution[0] == {"category": "cs.CV", "paper_count": 2}
    assert distribution[1] == {"category": "cs.CL", "paper_count": 1}


def test_summary_calculation() -> None:
    summary = compute_summary(_sample_papers())

    assert summary["total_papers"] == 3
    assert summary["year_range"] == "2024-2025"


def test_limit_papers_by_year_range() -> None:
    filtered = limit_papers_by_year_range(_sample_papers(), start_year=2025, end_year=2025)
    assert len(filtered) == 2
    assert all(paper["published_year"] == 2025 for paper in filtered)
