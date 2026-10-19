import json

import joblib

from src.config import CLASS_NAMES, MODEL_INFO_PATH, MODEL_PATH


def load_model(path=MODEL_PATH):
    """
    Load the trained model from disk.
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    return joblib.load(path)


def load_model_info(path=MODEL_INFO_PATH):
    """
    Load the model details saved by train.py.
    Returns:
        dict or None: The details, or None if the file does not exist.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text())


def predict_data(model, X):
    """
    Predict the class and class probabilities for the input data.
    Args:
        model: A trained classifier.
        X (list): One [sepal_length, sepal_width, petal_length, petal_width] row per sample.
    Returns:
        list: One dict per sample with class_id, species, confidence and probabilities.
    """
    class_ids = model.predict(X)
    probabilities = model.predict_proba(X)

    results = []
    for class_id, probs in zip(class_ids, probabilities):
        results.append({
            "class_id": int(class_id),
            "species": CLASS_NAMES[int(class_id)],
            "confidence": round(float(max(probs)), 4),
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        })
    return results
