import types

from backend.services import keyword_service


def _reset_keyword_globals() -> None:
    keyword_service._model = None
    keyword_service._kw_model = None


def test_extract_keywords_empty_returns_empty_list() -> None:
    _reset_keyword_globals()
    assert keyword_service.extract_keywords([]) == []


def test_extract_keywords_output_shape_and_non_zero_paper_count(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, text, *args, **kwargs):
            if "Reasoning" in text:
                return [("Vision Language", 0.82)]
            if "Cultural" in text:
                return [("Multimodal", 0.61)]
            return []

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [
        {"title": "Vision Language Models for Reasoning", "abstract": "A strong multimodal baseline."},
        {"title": "Cultural evaluation", "abstract": "We study MULTIMODAL systems."},
        {"title": "Audio only", "abstract": "No matching phrase here."},
    ]

    result = keyword_service.extract_keywords(papers, top_k=5)

    assert len(result) == 2
    assert set(result[0].keys()) == {"keyword", "score", "paper_count"}
    assert all(row["paper_count"] > 0 for row in result)

    by_keyword = {row["keyword"]: row for row in result}
    assert by_keyword["vision language"]["paper_count"] == 1
    assert by_keyword["multimodal"]["paper_count"] == 2


def test_extract_keywords_returns_empty_on_model_error(monkeypatch) -> None:
    _reset_keyword_globals()

    def _raise_error():
        raise RuntimeError("model init failed")

    monkeypatch.setattr(keyword_service, "get_keyword_model", _raise_error)

    papers = [{"title": "Vision", "abstract": "Language"}]
    assert keyword_service.extract_keywords(papers, top_k=5) == []


def test_extract_keywords_hyphen_normalization(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, text, *args, **kwargs):
            if "practice" in text:
                return [("vision-language", 0.77)]
            return []

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [
        {
            "title": "Vision-language models in practice",
            "abstract": "Hyphenated phrase source.",
        },
        {"title": "Vision language systems", "abstract": "Space-separated variant."},
    ]

    result = keyword_service.extract_keywords(papers, top_k=5)
    by_keyword = {row["keyword"]: row for row in result}

    assert by_keyword["vision language"]["paper_count"] == 2


def test_extract_keywords_filters_repeated_and_generic_phrases(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, text, *args, **kwargs):
            return [
                ("text text", 0.9),
                ("paper", 0.8),
                ("model framework", 0.7),
                ("vision language", 0.6),
            ]

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [{"title": "Vision language paper", "abstract": "model framework test"}]
    result = keyword_service.extract_keywords(papers, top_k=10)
    keywords = {row["keyword"] for row in result}

    assert "text text" not in keywords
    assert "paper" not in keywords
    assert "model framework" not in keywords
    assert "vision language" in keywords


def test_extract_keywords_ranks_by_paper_count_then_score(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, text, *args, **kwargs):
            if "Paper One" in text:
                return [("alpha topic", 0.2), ("rare signal", 0.99)]
            if "Paper Two" in text:
                return [("alpha topic", 0.25)]
            return []

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [
        {"title": "Paper One", "abstract": "Discuss alpha topic and rare signal."},
        {"title": "Paper Two", "abstract": "More alpha topic details."},
    ]

    result = keyword_service.extract_keywords(papers, top_k=10)

    assert len(result) >= 2
    assert result[0]["keyword"] == "alpha topic"
    assert result[0]["paper_count"] == 2
    assert result[1]["keyword"] == "rare signal"
    assert result[1]["paper_count"] == 1


def test_extract_keywords_skips_single_paper_failure(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, text, *args, **kwargs):
            if "Bad paper" in text:
                raise RuntimeError("bad extraction")
            return [("vision language", 0.7)]

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [
        {"title": "Bad paper", "abstract": "Will fail."},
        {"title": "Good paper", "abstract": "Vision language details."},
    ]
    result = keyword_service.extract_keywords(papers, top_k=10)

    assert len(result) == 1
    assert result[0]["keyword"] == "vision language"


def test_get_keyword_model_uses_local_model_dir_if_exists(monkeypatch, tmp_path) -> None:
    _reset_keyword_globals()

    model_dir = tmp_path / ".model" / "all-MiniLM-L6-v2"
    model_dir.mkdir(parents=True)

    calls = {"loaded_from": None, "saved_to": None, "keybert_model": None}

    class _FakeSentenceTransformer:
        def __init__(self, model_ref: str):
            calls["loaded_from"] = model_ref

        def save(self, out_path: str) -> None:
            calls["saved_to"] = out_path

    class _FakeKeyBERT:
        def __init__(self, model):
            calls["keybert_model"] = model

        def extract_keywords(self, *args, **kwargs):
            return []

    monkeypatch.setattr(keyword_service, "MODEL_DIR", model_dir)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "keybert",
        types.SimpleNamespace(KeyBERT=_FakeKeyBERT),
    )

    kw_model = keyword_service.get_keyword_model()

    assert kw_model is not None
    assert calls["loaded_from"] == str(model_dir)
    assert calls["saved_to"] is None
