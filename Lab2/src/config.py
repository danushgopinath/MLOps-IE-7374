from pathlib import Path

# Paths are based on this file's location, so the code works no matter
# which folder it is started from.
LAB_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = LAB_DIR / "model" / "iris_model.pkl"
MODEL_INFO_PATH = LAB_DIR / "model" / "model_info.json"

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Largest number of samples accepted by /predict/batch
MAX_BATCH_SIZE = 100
