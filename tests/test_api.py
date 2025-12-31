from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_schema():
    r = client.get("/schema")
    assert r.status_code == 200
    assert "required_features" in r.json()
