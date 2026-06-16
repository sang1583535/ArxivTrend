from collections import Counter, defaultdict


def limit_papers_by_year_range(
    papers: list[dict],
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[dict]:
    if start_year is None and end_year is None:
        return papers

    filtered: list[dict] = []
    for paper in papers:
        year = paper.get("published_year")
        if not isinstance(year, int) or year <= 0:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        filtered.append(paper)

    return filtered


def compute_topic_trend(papers: list[dict], granularity: str = "year") -> list[dict]:
    if granularity != "year":
        raise ValueError("Unsupported granularity. Only 'year' is supported.")

    year_counter: defaultdict[int, int] = defaultdict(int)
    for paper in papers:
        year = paper.get("published_year")
        if isinstance(year, int) and year > 0:
            year_counter[year] += 1

    return [
        {"period": str(year), "paper_count": count}
        for year, count in sorted(year_counter.items(), key=lambda item: item[0])
    ]


def compute_category_distribution(papers: list[dict]) -> list[dict]:
    counter: Counter[str] = Counter()

    for paper in papers:
        category = (paper.get("primary_category") or "unknown").strip() or "unknown"
        counter[category] += 1

    return [
        {"category": category, "paper_count": count}
        for category, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def compute_summary(papers: list[dict]) -> dict:
    total_papers = len(papers)

    years = [paper.get("published_year", 0) for paper in papers if isinstance(paper.get("published_year"), int)]
    years = [year for year in years if year > 0]
    if years:
        year_range = f"{min(years)}-{max(years)}"
    else:
        year_range = ""

    return {
        "total_papers": total_papers,
        "year_range": year_range,
    }
