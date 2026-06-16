from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeTopicRequest(BaseModel):
    query: str = Field(..., description="The search query string.")
    max_results: int = Field(100, ge=1, le=500, description="Maximum number of results to return.")
    granularity: Literal["year"] = Field("year", description="Trend granularity. MVP supports only 'year'.")


class AnalyzeSummary(BaseModel):
    total_papers: int
    year_range: str


class NormalizedPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    published_year: int
    updated_date: str
    primary_category: str
    categories: list[str]
    arxiv_url: str
    pdf_url: str


class TrendPoint(BaseModel):
    period: str
    paper_count: int


class CategoryPoint(BaseModel):
    category: str
    paper_count: int


class KeywordPoint(BaseModel):
    keyword: str
    score: float
    paper_count: int


class AnalyzeTopicResponse(BaseModel):
    query: str = Field(..., description="The original search query string.")
    summary: AnalyzeSummary
    papers: list[NormalizedPaper]
    trend: list[TrendPoint]
    category_distribution: list[CategoryPoint]
    top_keywords: list[KeywordPoint]