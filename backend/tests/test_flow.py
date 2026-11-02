import json

import pytest
from fastapi.testclient import TestClient

from app.chunking import split_text
from app.config import Settings
from app.errors import AppError
from app.extraction import extract_sections
from app.generation import validate_answer
from app.main import create_app
from app.models import Passage
from app.retrieval import retrieve


class FakeEmbeddings:
    def embed(self, texts):
        return [[1.0, 0.0] if "helios" in text.lower() else [0.0, 1.0] for text in texts]


class FakeGenerator:
    def answer(self, question, passages):
        return validate_answer(
            json.dumps({"claims": [{"source": 1, "quote": "ORBIT-42"}]})
            if passages
            else '{"claims": []}',
            passages,
        )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_mb=1))) as client:
        client.app.state.embeddings = FakeEmbeddings()
        client.app.state.generator = FakeGenerator()
        yield client


def test_upload_ask_delete_and_persist(client):
    response = client.post(
        "/api/documents",
        files={"file": ("helios.txt", b"Synthetic Helios launch code: ORBIT-42.", "text/plain")},
    )
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "ready"
    assert len(client.get("/api/documents").json()) == 1
    answer = client.post("/api/chat", json={"question": "Helios launch code?"}).json()
    assert "ORBIT-42" in answer["answer"]
    assert answer["citations"][0]["document_id"] == document["id"]
    assert answer["citations"][0]["filename"] == "helios.txt"
    assert answer["citations"][0]["section"] == 1
    assert answer["citations"][0]["page"] is None
    store = client.app.state.store
    assert len(type(store)(store.path, "nomic-embed-text").passages()) == 1
    assert client.delete(f"/api/documents/{document['id']}").status_code == 204
    assert store.passages() == []
    assert client.post("/api/chat", json={"question": "Helios?"}).json()["insufficient_evidence"]


def test_retrieval_threshold_and_ranking(client):
    for filename, text in [("one.txt", b"Helios uses ORBIT-42."), ("two.txt", b"Other project.")]:
        assert client.post("/api/documents", files={"file": (filename, text)}).status_code == 201
    passages = retrieve(client.app.state.store, [1.0, 0.0], 5, 0.5)
    assert [passage.filename for passage in passages] == ["one.txt"]
    assert retrieve(client.app.state.store, [-1.0, -1.0], 5, 0.5) == []
    with pytest.raises(AppError):
        retrieve(client.app.state.store, [1.0], 5, 0.5)


def test_chunking_overlap_and_boundaries():
    text = " ".join(f"word{index}" for index in range(100))
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert chunks[0][-19:] in chunks[1]
    assert chunks[-1].endswith("word99")
    assert split_text(" \n ") == []
    assert split_text("short") == ["short"]
    with pytest.raises(ValueError):
        split_text("test", 10, 10)


@pytest.mark.parametrize(
    "filename,content,code",
    [
        ("empty.txt", b"   ", 400),
        ("bad.pdf", b"not a PDF", 400),
        ("bad.exe", b"binary", 415),
        ("bad.txt", b"\xff", 400),
        ("binary.txt", b"a\x00b", 400),
        ("huge.txt", b"a" * (1024 * 1024 + 1), 413),
    ],
)
def test_upload_errors_leave_no_index(client, filename, content, code):
    assert client.post("/api/documents", files={"file": (filename, content)}).status_code == code
    assert client.get("/api/documents").json() == []


def test_model_failure_leaves_no_index(client):
    class Unavailable:
        def embed(self, texts):
            raise AppError("Embeddings unavailable.", 503)

    client.app.state.embeddings = Unavailable()
    response = client.post("/api/documents", files={"file": ("ok.txt", b"Some text.")})
    assert response.status_code == 503
    assert response.json()["detail"] == "Embeddings unavailable."
    assert client.get("/api/documents").json() == []


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"claims": []}',
        '{"claims": [{"source": 2, "quote": "fact"}]}',
        '{"claims": [{"source": 1, "quote": "invented"}]}',
        '{"claims": [{"text": "False [9]", "source": 1, "quote": "fact"}]}',
    ],
)
def test_invalid_citations_abstain(raw):
    passage = Passage(id="a", document_id="b", filename="x.pdf", page=2, section=2, text="A fact.")
    answer = validate_answer(raw, [passage])
    assert answer.insufficient_evidence
    assert not answer.citations


def test_pdf_page_citation_and_empty_question(client):
    passage = Passage(id="a", document_id="b", filename="x.pdf", page=7, section=7, text="A fact.")
    answer = validate_answer('{"claims": [{"source": 1, "quote": "A fact."}]}', [passage])
    assert answer.citations[0].page == 7
    assert answer.answer == '"A fact." [1]'
    assert client.post("/api/chat", json={"question": "  "}).status_code == 422
    assert client.delete("/api/documents/missing").status_code == 404


def test_extracted_text_limit():
    with pytest.raises(AppError, match="too large"):
        extract_sections("large.txt", b"a" * 1001, 1000)


def test_request_body_limit(client):
    response = client.post("/api/documents", files={"file": ("big.txt", b"x" * 1_200_000)})
    assert response.status_code == 413
    assert client.app.state.store.passages() == []


def test_unrelated_question_abstains(client):
    client.post("/api/documents", files={"file": ("helios.txt", b"Helios uses ORBIT-42.")})
    answer = client.post("/api/chat", json={"question": "How many oceans are there?"}).json()
    assert answer["insufficient_evidence"]
    assert answer["citations"] == []
