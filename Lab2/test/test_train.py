import json

from sklearn.metrics import accuracy_score

from src.config import CLASS_NAMES
from src.data import load_data, split_data
from src.predict import load_model
from src.train import fit_model, save_model


def test_model_accuracy():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.9


def test_save_model_writes_model_and_info(tmp_path):
    X, y = load_data()
    model = fit_model(X, y)
    model_path = tmp_path / "iris_model.pkl"
    info_path = tmp_path / "model_info.json"

    save_model(model, 0.95, model_path=model_path, info_path=info_path)

    assert load_model(model_path).predict([[5.1, 3.5, 1.4, 0.2]])[0] == 0
    info = json.loads(info_path.read_text())
    assert info["test_accuracy"] == 0.95
    assert info["classes"] == CLASS_NAMES
