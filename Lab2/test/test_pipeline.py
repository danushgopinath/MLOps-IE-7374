import json

import numpy as np
import pytest

from src.calibrate_model import calibrate
from src.common import expected_calibration_error, load_splits, read_latest
from src.evaluate_model import evaluate, quality_gate
from src.train_model import train


def test_splits_are_reproducible():
    first = load_splits()
    second = load_splits()
    for name in ("train", "calib", "test"):
        assert np.array_equal(first[name][0], second[name][0])
        assert np.array_equal(first[name][1], second[name][1])


def test_split_sizes():
    splits = load_splits(n_samples=1000)
    assert len(splits["train"][1]) == 600
    assert len(splits["calib"][1]) == 200
    assert len(splits["test"][1]) == 200


def test_quality_gate_first_model():
    passed, _ = quality_gate(0.90, None, min_f1=0.85, max_drop=0.01)
    assert passed
    passed, reason = quality_gate(0.80, None, min_f1=0.85, max_drop=0.01)
    assert not passed
    assert "below the minimum" in reason


@pytest.mark.parametrize("candidate_f1, champion_f1, expected", [
    (0.95, 0.90, True),   # better than current model
    (0.90, 0.90, True),   # same as current model
    (0.895, 0.90, True),  # small drop, within max_drop
    (0.88, 0.90, False),  # drop larger than max_drop
    (0.84, 0.80, False),  # better than current model but below min_f1
])
def test_quality_gate_against_current_model(candidate_f1, champion_f1, expected):
    passed, _ = quality_gate(candidate_f1, champion_f1, min_f1=0.85, max_drop=0.01)
    assert passed == expected


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    # Always predicting 0.99 when only half are positive is badly calibrated
    assert expected_calibration_error(y, np.full(100, 0.99)) == pytest.approx(0.49)


def test_good_model_is_promoted(tmp_path):
    train("v1", out_dir=tmp_path, n_estimators=20)
    report = evaluate("v1", out_dir=tmp_path)

    assert report["gate"]["passed"]
    assert (tmp_path / "metrics" / "v1_metrics.json").exists()
    assert read_latest(tmp_path)["version"] == "v1"


def test_worse_model_is_rejected(tmp_path):
    train("v1", out_dir=tmp_path, n_estimators=20)
    evaluate("v1", out_dir=tmp_path)

    train("v2", out_dir=tmp_path, n_estimators=5, max_depth=1)
    report = evaluate("v2", out_dir=tmp_path)

    assert not report["gate"]["passed"]
    assert report["champion"]["version"] == "v1"
    # The rejected model's metrics are still saved, but latest.json does not change
    saved = json.loads((tmp_path / "metrics" / "v2_metrics.json").read_text())
    assert saved["gate"]["passed"] is False
    assert read_latest(tmp_path)["version"] == "v1"


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_report(tmp_path, method):
    train("v1", out_dir=tmp_path, n_estimators=20)
    evaluate("v1", out_dir=tmp_path)
    report = calibrate(tmp_path, method=method)

    assert report["version"] == "v1"
    assert (tmp_path / "metrics" / f"v1_calibration_{method}.json").exists()
    calibrated_file = tmp_path / "models" / f"model_v1_rf_calibrated_{method}.joblib"
    assert calibrated_file.exists() == report["improved"]


def test_calibration_without_model_fails(tmp_path):
    with pytest.raises(ValueError):
        calibrate(tmp_path)
