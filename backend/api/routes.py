from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    AnalyzeTopicRequest,
    AnalyzeTopicResponse,
)

from backend.services.arxiv_service import fetch_arxiv_papers
from backend.services.keyword_service import extract_top_keywords
from backend.services.stats_service import compute_category_distribution, compute_summary, compute_topic_trend

router = APIRouter()

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
    
@router.post("/analyze-topic", response_model=AnalyzeTopicResponse)
def analyze_topic(request: AnalyzeTopicRequest) -> AnalyzeTopicResponse:
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        papers = fetch_arxiv_papers(cleaned_query, request.max_results)
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