import guard, structured

def fake_dense(score):
    return lambda q, k=4: [{"source": "x p.1", "score": score, "text": "t"}]

def test_none_below_threshold(monkeypatch):
    monkeypatch.setattr(guard, "search_dense", fake_dense(0.60))
    hits, conf = guard.retrieve("q")
    assert conf == "none" and hits == []

def test_high_above_threshold(monkeypatch):
    monkeypatch.setattr(guard, "search_dense", fake_dense(0.80))
    assert guard.retrieve("q")[1] == "high"

def test_low_band_adds_bm25(monkeypatch):
    monkeypatch.setattr(guard, "search_dense", fake_dense(0.72))
    monkeypatch.setattr(guard, "search_bm25",
                        lambda q, k=2: [{"source": "y p.2", "score": 1.0, "text": "u"}])
    hits, conf = guard.retrieve("q")
    assert conf == "low" and len(hits) == 2

class R:
    def __init__(self, text): self.text = text

def test_retry_recovers_from_invalid_json(monkeypatch):
    monkeypatch.setattr(structured, "run", lambda q: "1595")
    replies = iter([R('{"answer":"x","refused":false,"confidence":5}'),
                    R('{"answer":"1595","value":1595,"refused":false,"confidence":0.9}')])
    class FakeModels:
        def generate_content(self, **kw): return next(replies)
    class FakeClient: models = FakeModels()
    monkeypatch.setattr(structured, "client", FakeClient())
    assert structured.structured_run("q").value == 1595
