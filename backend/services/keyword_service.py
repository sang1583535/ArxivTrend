import logging
from pathlib import Path
import re
from typing import Any
from collections import defaultdict

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path(".model/all-MiniLM-L6-v2")

_model: Any = None
_kw_model: Any = None

LOGGER = logging.getLogger(__name__)
DOMAIN_STOPWORDS = {
    "paper", "method", "methods", "approach", "approaches",
    "model", "models", "task", "tasks", "data", "result", "results",
    "performance", "proposed", "existing", "using", "based",
    "training", "learning", "input", "output", "feature", "features",
    "module", "framework",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_normalized_keyword(normalized_keyword: str, normalized_text: str) -> bool:
    if not normalized_keyword or not normalized_text:
        return False

    return f" {normalized_keyword} " in f" {normalized_text} "


def _has_adjacent_duplicate_tokens(tokens: list[str]) -> bool:
    return any(tokens[i] == tokens[i + 1] for i in range(len(tokens) - 1))


def _is_valid_keyword(normalized_keyword: str) -> bool:
    if len(normalized_keyword) < 3:
        return False

    tokens = normalized_keyword.split()
    if not tokens or len(tokens) > 4:
        return False

    if _has_adjacent_duplicate_tokens(tokens):
        return False

    if len(tokens) == 1:
        return tokens[0] not in DOMAIN_STOPWORDS

    stopword_count = sum(1 for token in tokens if token in DOMAIN_STOPWORDS)
    return stopword_count <= (len(tokens) / 2)


def get_keyword_model() -> Any:
    global _model, _kw_model

    if _kw_model is not None:
        return _kw_model

    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        if MODEL_DIR.exists():
            _model = SentenceTransformer(str(MODEL_DIR))
        else:
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            _model = SentenceTransformer(MODEL_NAME)
            _model.save(str(MODEL_DIR))

        _kw_model = KeyBERT(model=_model)
        return _kw_model
    except Exception as exc:
        LOGGER.exception("Failed to initialize KeyBERT model: %s", exc)
        raise


def extract_keywords(papers: list[dict], top_k: int = 20) -> list[dict]:
    if not papers:
        return []

    try:
        kw_model = get_keyword_model()
    except Exception as exc:
        LOGGER.exception("KeyBERT model initialization failed: %s", exc)
        return []

    keyword_scores: dict[str, list[float]] = defaultdict(list)
    keyword_papers: dict[str, set[int]] = defaultdict(set)
    normalized_paper_texts: dict[int, str] = {}

    for index, paper in enumerate(papers):
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
        normalized_paper_text = normalize_text(paper_text)
        if not normalized_paper_text:
            continue

        normalized_paper_texts[index] = normalized_paper_text

        try:
            raw_keywords = kw_model.extract_keywords(
                paper_text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=8,
                use_mmr=True,
                diversity=0.4,
            )
        except Exception as exc:
            LOGGER.warning("KeyBERT extraction failed for one paper: %s", exc)
            continue

        for row in raw_keywords:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue

            raw_keyword = str(row[0]).strip()
            normalized_keyword = normalize_text(raw_keyword)
            if not _is_valid_keyword(normalized_keyword):
                continue

            try:
                score = float(row[1])
            except (TypeError, ValueError):
                continue

            keyword_scores[normalized_keyword].append(score)
            keyword_papers[normalized_keyword].add(index)

    aggregated = []
    for keyword, scores in keyword_scores.items():
        if not scores:
            continue

        for paper_index, normalized_paper_text in normalized_paper_texts.items():
            if _contains_normalized_keyword(keyword, normalized_paper_text):
                keyword_papers[keyword].add(paper_index)

        paper_count = len(keyword_papers[keyword])
        if paper_count == 0:
            continue

        average_score = round(sum(scores) / len(scores), 4)
        aggregated.append(
            {
                "keyword": keyword,
                "score": average_score,
                "paper_count": paper_count,
            }
        )

    aggregated.sort(key=lambda item: (-item["paper_count"], -item["score"], item["keyword"]))
    return aggregated[:top_k]


def extract_top_keywords(papers: list[dict], top_k: int = 20) -> list[dict]:
    return extract_keywords(papers, top_k=top_k)
