import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Opt-in: requires running Ollama and pulled models"
)


def test_real_ollama_upload_answer_and_abstention(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        upload = client.post(
            "/api/documents",
            files={"file": ("synthetic.txt", b"The Helios observatory launch code is ORBIT-42.")},
        )
        assert upload.status_code == 201, upload.text
        response = client.post("/api/chat", json={"question": "What is the Helios launch code?"})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert not answer["insufficient_evidence"], answer
        assert "ORBIT-42" in answer["answer"]
        assert answer["citations"][0]["filename"] == "synthetic.txt"
        assert answer["citations"][0]["section"] == 1
        unknown = client.post(
            "/api/chat",
            json={"question": "What is the exact population of Tokyo in the year 1920?"},
        )
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["insufficient_evidence"], unknown.text
        assert unknown.json()["citations"] == []
