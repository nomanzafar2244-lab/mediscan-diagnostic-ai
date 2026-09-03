from fastapi.testclient import TestClient
from app.main import app
from sklearn.datasets import load_breast_cancer
client=TestClient(app)
def test_health(): assert client.get("/health").status_code==200
def test_info(): assert client.get("/model-info").json()["features"]==30
def test_validation(): assert client.post("/predict",json={"features":[1,2]}).status_code==422
