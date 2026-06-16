import types

from backend.services import keyword_service


def _reset_keyword_globals() -> None:
    keyword_service._model = None
    keyword_service._kw_model = None


def test_extract_keywords_empty_returns_empty_list() -> None:
    _reset_keyword_globals()
    assert keyword_service.extract_keywords([]) == []


def test_extract_keywords_output_shape_and_paper_count(monkeypatch) -> None:
    _reset_keyword_globals()

    class _FakeKwModel:
        def extract_keywords(self, *args, **kwargs):
            return [
                ("Vision Language Models", 0.82),
                ("Multimodal", 0.61),
            ]

    monkeypatch.setattr(keyword_service, "get_keyword_model", lambda: _FakeKwModel())

    papers = [
        {"title": "Vision Language Models for Reasoning", "abstract": "A strong multimodal baseline."},
        {"title": "Cultural evaluation", "abstract": "We study MULTIMODAL systems."},
        {"title": "Audio only", "abstract": "No matching phrase here."},
    ]

    result = keyword_service.extract_keywords(papers, top_k=5)

    assert len(result) == 2
    assert set(result[0].keys()) == {"keyword", "score", "paper_count"}

    by_keyword = {row["keyword"].lower(): row for row in result}
    assert by_keyword["vision language models"]["paper_count"] == 1
    assert by_keyword["multimodal"]["paper_count"] == 2


def test_extract_keywords_returns_empty_on_model_error(monkeypatch) -> None:
    _reset_keyword_globals()

    def _raise_error():
        raise RuntimeError("model init failed")

    monkeypatch.setattr(keyword_service, "get_keyword_model", _raise_error)

    papers = [{"title": "Vision", "abstract": "Language"}]
    assert keyword_service.extract_keywords(papers, top_k=5) == []


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
