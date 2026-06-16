from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    AnalyzeTopicRequest,
    AnalyzeTopicResponse,
)

from backend.services.arxiv_service import fetch_arxiv_papers
from backend.services.keyword_service import extract_top_keywords
from backend.services.stats_service import (
    compute_category_distribution,
    compute_summary,
    compute_topic_trend,
    limit_papers_by_year_range,
)

router = APIRouter()

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
    
@router.post("/analyze-topic", response_model=AnalyzeTopicResponse)
def analyze_topic(request: AnalyzeTopicRequest) -> AnalyzeTopicResponse:
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    current_year = datetime.now(timezone.utc).year
    effective_end_year = request.end_year if request.end_year is not None else current_year
    effective_start_year = request.start_year if request.start_year is not None else effective_end_year - 5

    if (
        effective_start_year is not None
        and effective_end_year is not None
        and effective_start_year > effective_end_year
    ):
        raise HTTPException(status_code=400, detail="start_year cannot be greater than end_year.")

    try:
        papers = fetch_arxiv_papers(cleaned_query, request.max_results)
        papers = limit_papers_by_year_range(
            papers,
            start_year=effective_start_year,
            end_year=effective_end_year,
        )
        trend = compute_topic_trend(papers, request.granularity)
        category_distribution = compute_category_distribution(papers)
        top_keywords = extract_top_keywords(papers, top_k=20)
        summary = compute_summary(papers)

        return AnalyzeTopicResponse(
            query=cleaned_query,
            summary=summary,
            papers=papers,
            trend=trend,
            category_distribution=category_distribution,
            top_keywords=top_keywords,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"arXiv request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze topic failed: {exc}") from exc