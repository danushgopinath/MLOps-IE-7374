import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.config import CLASS_NAMES, FEATURE_NAMES, MODEL_INFO_PATH, MODEL_PATH
from src.data import load_data, split_data


def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        DecisionTreeClassifier: The trained model.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier


def save_model(model, test_accuracy, model_path=MODEL_PATH, info_path=MODEL_INFO_PATH):
    """
    Save the model, plus a JSON file with details about it that the API returns at /model/info.
    Args:
        model: The trained model.
        test_accuracy (float): Accuracy on the test split.
        model_path (Path): Where to save the model.
        info_path (Path): Where to save the model details.
    Returns:
        dict: The saved model details.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    info = {
        "model_type": type(model).__name__,
        "params": {"max_depth": model.max_depth, "random_state": model.random_state},
        "test_accuracy": round(float(test_accuracy), 4),
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    return info


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    save_model(model, test_accuracy)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved model details to {MODEL_INFO_PATH}")
