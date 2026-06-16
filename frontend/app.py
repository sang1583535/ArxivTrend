from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None


API_URL = "http://localhost:8000/analyze-topic"
DEFAULT_QUERY = "multimodal cultural benchmark"
DEFAULT_MAX_RESULTS = 100
CURRENT_YEAR = datetime.now(timezone.utc).year
MIN_YEAR = 1991


st.set_page_config(
    page_title="ArxivTrendIR",
    page_icon="📚",
    layout="wide",
)


st.title("ArxivTrend")
st.caption("Explore arXiv research topics through publication trends, categories, keywords, and paper metadata.")


def analyze_topic(api_url: str, query: str, max_results: int, start_year: int, end_year: int) -> dict[str, Any]:
    payload = {
        "query": query,
        "max_results": max_results,
        "granularity": "year",
        "start_year": start_year,
        "end_year": end_year,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=60)
    except requests.Timeout as exc:
        raise RuntimeError("Request to backend timed out.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to connect to backend API: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip()
        try:
            error_json = response.json()
            if isinstance(error_json, dict):
                detail = error_json.get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Backend returned an invalid JSON response.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Backend returned an unexpected response format.")

    return result


def filter_papers_by_year(papers: list[dict[str, Any]], start_year: int, end_year: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for paper in papers:
        year = paper.get("published_year")
        if isinstance(year, int) and start_year <= year <= end_year:
            filtered.append(paper)
    return filtered


def build_trend_from_papers(papers: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter()
    for paper in papers:
        year = paper.get("published_year")
        if isinstance(year, int):
            counts[str(year)] += 1
    data = [{"period": period, "paper_count": count} for period, count in sorted(counts.items(), key=lambda item: item[0])]
    return pd.DataFrame(data)


def build_category_distribution_from_papers(papers: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter()
    for paper in papers:
        category = str(paper.get("primary_category") or "unknown").strip() or "unknown"
        counts[category] += 1
    data = [{"category": category, "paper_count": count} for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return pd.DataFrame(data)


def build_summary_from_papers(papers: list[dict[str, Any]], start_year: int, end_year: int) -> dict[str, Any]:
    categories = build_category_distribution_from_papers(papers)
    return {
        "total_papers": len(papers),
        "year_range": f"{start_year}-{end_year}",
        "num_categories": int(len(categories)),
    }


def _safe_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _as_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(items) if items else pd.DataFrame()


def _empty_message(message: str) -> None:
    st.info(message)


with st.sidebar:
    st.header("🔎 Search")
    query = st.text_input("Research topic", value=DEFAULT_QUERY)
    max_results = st.slider("Max results", 10, 300, DEFAULT_MAX_RESULTS, step=10)
    start_year = st.number_input("Start year", min_value=MIN_YEAR, max_value=CURRENT_YEAR, value=2020, step=1)
    end_year = st.number_input("End year", min_value=MIN_YEAR, max_value=CURRENT_YEAR, value=CURRENT_YEAR, step=1)
    analyze = st.button("Analyze Topic", type="primary", width="stretch")
    st.caption("Retrieved arXiv papers are treated as a topic-specific corpus for analysis.")


if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "analysis_error" not in st.session_state:
    st.session_state["analysis_error"] = None


if analyze:
    if int(start_year) > int(end_year):
        st.sidebar.error("Start year must be less than or equal to end year.")
        st.session_state["analysis_error"] = "Start year must be less than or equal to end year."
        st.session_state["analysis_result"] = None
    elif not query.strip():
        st.sidebar.error("Query cannot be empty.")
        st.session_state["analysis_error"] = "Query cannot be empty."
        st.session_state["analysis_result"] = None
    else:
        with st.spinner("Analyzing topic with arXiv corpus..."):
            try:
                st.session_state["analysis_result"] = analyze_topic(API_URL, query.strip(), int(max_results), int(start_year), int(end_year))
                st.session_state["analysis_error"] = None
            except Exception as exc:
                st.session_state["analysis_error"] = str(exc)
                st.session_state["analysis_result"] = None


if st.session_state.get("analysis_error"):
    st.error(st.session_state["analysis_error"])


result = st.session_state.get("analysis_result")
if not result:
    st.info(
        "Use the sidebar to search a research topic. The dashboard will show topic activity over time, arXiv category distribution, top keywords, and retrieved papers."
    )
    st.write("Example queries:")
    st.markdown(
        "- multimodal cultural benchmark\n- vision language model evaluation\n- retrieval augmented generation\n- large language model reasoning"
    )
    st.stop()


backend_papers = _safe_list(result.get("papers"))
filtered_papers = filter_papers_by_year(backend_papers, int(start_year), int(end_year))
filtered_trend = build_trend_from_papers(filtered_papers)
filtered_categories = build_category_distribution_from_papers(filtered_papers)
summary = build_summary_from_papers(filtered_papers, int(start_year), int(end_year))

top_keywords = _safe_list(result.get("top_keywords"))

if not filtered_papers:
    st.warning("No papers remain after applying the selected year filter.")

st.header("📌 Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total papers", summary["total_papers"])
col2.metric("Year range", summary["year_range"])
col3.metric("Categories", summary["num_categories"])
col4.metric("Keywords", len(top_keywords))

st.header("📈 Topic Activity Over Time")
if filtered_trend.empty:
    st.info("No trend data available for the selected filters.")
else:
    trend_fig = px.line(filtered_trend, x="period", y="paper_count", markers=True) if px is not None else None
    if trend_fig is not None:
        trend_fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Year",
            yaxis_title="Paper count",
        )
        st.plotly_chart(trend_fig, width="stretch")
    else:
        st.line_chart(filtered_trend.set_index("period")["paper_count"])

st.header("🧭 arXiv Category Distribution")
if filtered_categories.empty:
    st.info("No category distribution available for the selected filters.")
else:
    category_df = filtered_categories.sort_values("paper_count", ascending=False).head(10)
    category_fig = px.bar(category_df, x="category", y="paper_count") if px is not None else None
    if category_fig is not None:
        category_fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Category",
            yaxis_title="Paper count",
        )
        st.plotly_chart(category_fig, width="stretch")
    else:
        st.bar_chart(category_df.set_index("category")["paper_count"])

st.header("🔑 Top Keywords")
st.caption("Keywords are generated from the retrieved corpus.")
if not top_keywords:
    st.info("No keywords available.")
else:
    keyword_df = _as_dataframe(top_keywords)
    if not keyword_df.empty:
        if {"paper_count", "score"}.issubset(keyword_df.columns):
            keyword_df = keyword_df.sort_values(["paper_count", "score"], ascending=[False, False]).head(20)
        elif "paper_count" in keyword_df.columns:
            keyword_df = keyword_df.sort_values("paper_count", ascending=False).head(20)
        else:
            keyword_df = keyword_df.head(20)

        chart_metric = "paper_count" if "paper_count" in keyword_df.columns and keyword_df["paper_count"].fillna(0).sum() > 0 else "score"
        if chart_metric in keyword_df.columns:
            chart_df = keyword_df[[col for col in ["keyword", chart_metric] if col in keyword_df.columns]].copy()
            if not chart_df.empty:
                keyword_fig = px.bar(
                    chart_df.sort_values(chart_metric, ascending=True),
                    x=chart_metric,
                    y="keyword",
                    orientation="h",
                ) if px is not None else None
                if keyword_fig is not None:
                    keyword_fig.update_layout(
                        template="plotly_white",
                        height=max(360, 24 * len(chart_df) + 120),
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Paper count" if chart_metric == "paper_count" else "Score",
                        yaxis_title="Keyword",
                    )
                    st.plotly_chart(keyword_fig, width="stretch")
                else:
                    st.bar_chart(chart_df.set_index("keyword")[chart_metric])

        st.dataframe(keyword_df[[col for col in ["keyword", "score", "paper_count"] if col in keyword_df.columns]], width="stretch", hide_index=True)

st.header("📄 Retrieved Papers")
if not filtered_papers:
    st.info("No papers returned from the backend.")
else:
    st.caption(f"Showing first {min(30, len(filtered_papers))} of {len(filtered_papers)} papers.")
    for index, paper in enumerate(filtered_papers[:30], start=1):
        title = paper.get("title", "Untitled paper")
        with st.expander(f"{index}. {title}"):
            authors = paper.get("authors", [])
            published_date = paper.get("published_date", "-")
            primary_category = paper.get("primary_category", "-")
            abstract = paper.get("abstract", "") or "No abstract available."
            arxiv_url = paper.get("arxiv_url", "")
            pdf_url = paper.get("pdf_url", "")

            st.markdown(f"**Authors:** {', '.join(authors) if authors else 'Unknown'}")
            st.markdown(f"**Published:** {published_date}")
            st.markdown(f"**Primary category:** {primary_category}")
            st.write(abstract[:1000])

            left, right = st.columns(2)
            if arxiv_url:
                left.link_button("Open arXiv", arxiv_url)
            if pdf_url:
                right.link_button("Open PDF", pdf_url)

    if len(filtered_papers) > 30:
        st.caption(f"Showing first 30 of {len(filtered_papers)} papers.")
