import pytest
from fastapi.testclient import TestClient

from src.config import CLASS_NAMES, MAX_BATCH_SIZE
from src.main import app

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}


@pytest.fixture
def client():
    # Using "with" runs the lifespan function, which loads the model
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_setosa(client):
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] == 0
    assert body["species"] == "setosa"
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["confidence"] == max(body["probabilities"].values())


def test_predict_virginica(client):
    response = client.post("/predict", json=VIRGINICA)
    assert response.status_code == 200
    assert response.json()["species"] == "virginica"


@pytest.mark.parametrize("bad_input", [
    {**SETOSA, "sepal_length": -1.0},
    {**SETOSA, "petal_width": 0},
    {**SETOSA, "petal_length": 25},
    {**SETOSA, "sepal_width": "abc"},
    {key: value for key, value in SETOSA.items() if key != "petal_width"},
    {**SETOSA, "colour": "purple"},
])
def test_predict_rejects_bad_input(client, bad_input):
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_batch_predict(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA, VIRGINICA]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["species"] for p in body["predictions"]] == ["setosa", "virginica"]


def test_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422


def test_batch_rejects_too_many_samples(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA] * (MAX_BATCH_SIZE + 1)})
    assert response.status_code == 422


def test_batch_rejects_one_bad_sample(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA, {**SETOSA, "sepal_length": -2}]})
    assert response.status_code == 422


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "DecisionTreeClassifier"
    assert body["classes"] == CLASS_NAMES
    assert 0 < body["test_accuracy"] <= 1


def test_returns_503_when_model_is_missing(client):
    client.app.state.model = None
    assert client.get("/").status_code == 503
    assert client.post("/predict", json=SETOSA).status_code == 503
    assert client.post("/predict/batch", json={"samples": [SETOSA]}).status_code == 503
    assert client.get("/model/info").status_code == 503
