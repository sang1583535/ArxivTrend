import logging
from pathlib import Path
from typing import Any

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path(".model/all-MiniLM-L6-v2")

_model: Any = None
_kw_model: Any = None

LOGGER = logging.getLogger(__name__)


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

    paper_texts = []
    lowered_paper_texts = []
    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
        if text:
            paper_texts.append(text)
            lowered_paper_texts.append(text.lower())

    if not paper_texts:
        return []

    corpus_text = "\n".join(paper_texts)

    try:
        kw_model = get_keyword_model()
        raw_keywords = kw_model.extract_keywords(
            corpus_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_k,
            use_mmr=True,
            diversity=0.5,
        )
    except Exception as exc:
        LOGGER.exception("KeyBERT keyword extraction failed: %s", exc)
        return []

    keywords = []
    seen = set()
    for row in raw_keywords:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue

        keyword = str(row[0]).strip()
        if not keyword:
            continue

        normalized_keyword = keyword.lower()
        if normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)

        paper_count = sum(1 for paper_text in lowered_paper_texts if normalized_keyword in paper_text)

        try:
            score = round(float(row[1]), 4)
        except (TypeError, ValueError):
            score = 0.0

        keywords.append(
            {
                "keyword": keyword,
                "score": score,
                "paper_count": paper_count,
            }
        )

    return keywords


def extract_top_keywords(papers: list[dict], top_k: int = 20) -> list[dict]:
    return extract_keywords(papers, top_k=top_k)
