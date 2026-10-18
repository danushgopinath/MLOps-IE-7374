import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]

# Local runs write here (ignored by git). The workflows pass --out-dir . so that
# models and metrics land in Lab2/models and Lab2/metrics and get committed.
DEFAULT_OUT_DIR = LAB_DIR / "outputs"

N_SAMPLES = 3000
DATA_SEED = 42


def load_splits(n_samples=N_SAMPLES, seed=DATA_SEED):
    """
    Generates the synthetic dataset and splits it 60/20/20 into train,
    calibration and test sets.
    The same seed always gives the same split, so every run of the workflows
    scores models on exactly the same test set.
    Args:
        n_samples (int): Total number of samples to generate.
        seed (int): Random seed for the data and the split.
    Returns:
        dict: {"train": (X, y), "calib": (X, y), "test": (X, y)}
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.05,
        class_sep=0.8,
        random_state=seed,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "calib": (X_calib, y_calib),
        "test": (X_test, y_test),
    }


def model_path(out_dir, version, suffix="rf"):
    return Path(out_dir) / "models" / f"model_{version}_{suffix}.joblib"


def metrics_path(out_dir, name):
    return Path(out_dir) / "metrics" / f"{name}.json"


def latest_path(out_dir):
    return Path(out_dir) / "models" / "latest.json"


def read_latest(out_dir):
    """
    Reads models/latest.json, which points to the model version currently in use.
    Returns:
        dict or None: The file contents, or None if no model has been promoted yet.
    """
    path = latest_path(out_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Average gap between predicted probability and observed positive rate,
    weighted by how many samples fall in each probability bin.
    Args:
        y_true (array): True 0/1 labels.
        y_prob (array): Predicted probability of class 1.
        n_bins (int): Number of equal-width bins between 0 and 1.
    Returns:
        float: 0 means perfectly calibrated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            gap = abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
            ece += gap * in_bin.mean()
    return float(ece)


def run_info():
    """Which commit and workflow run produced a model (local runs say "local")."""
    return {
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }


def add_to_job_summary(markdown):
    """Appends markdown to the GitHub Actions job summary. Does nothing locally."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(markdown + "\n")


def annotate(level, message):
    """
    Prints a message. Inside GitHub Actions it is printed as a workflow command
    (::warning:: or ::error::) so it also shows up as an annotation on the run page.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")
    else:
        print(f"{level.upper()}: {message}")
