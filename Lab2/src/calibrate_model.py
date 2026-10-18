import argparse
import sys
from datetime import datetime, timezone

from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, f1_score, log_loss

from src.common import (
    DEFAULT_OUT_DIR,
    add_to_job_summary,
    annotate,
    expected_calibration_error,
    latest_path,
    load_splits,
    metrics_path,
    model_path,
    read_latest,
    run_info,
    write_json,
)

METHODS = ("sigmoid", "isotonic")


def probability_metrics(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "brier": round(float(brier_score_loss(y, y_prob)), 4),
        "log_loss": round(float(log_loss(y, y_prob)), 4),
        "ece": round(expected_calibration_error(y, y_prob), 4),
        "f1": round(float(f1_score(y, model.predict(X))), 4),
    }


def calibrate(out_dir=DEFAULT_OUT_DIR, method="sigmoid", version=None):
    """
    Calibrates the predicted probabilities of a saved model.
    The saved forest is frozen (not retrained). Only the calibrator (Platt scaling
    for "sigmoid", isotonic regression for "isotonic") is fit, on the calibration
    split. Before/after metrics are measured on the test split.
    The calibrated model is saved only if it lowers the Brier score.
    Args:
        out_dir (str/Path): Folder that contains models/ and metrics/.
        method (str): "sigmoid" or "isotonic".
        version (str/None): Model version to calibrate. Defaults to the one in models/latest.json.
    Returns:
        dict: The calibration report that was written to the metrics file.
    Raises:
        ValueError: If the method is unknown or there is no model to calibrate.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    latest = read_latest(out_dir)
    if version is None:
        if latest is None:
            raise ValueError("No models/latest.json found. Train and evaluate a model first.")
        version = latest["version"]

    model = load(model_path(out_dir, version))
    splits = load_splits()
    X_calib, y_calib = splits["calib"]
    X_test, y_test = splits["test"]

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_calib, y_calib)

    before = probability_metrics(model, X_test, y_test)
    after = probability_metrics(calibrated, X_test, y_test)
    improved = after["brier"] < before["brier"]

    calibrated_file = model_path(out_dir, version, suffix=f"rf_calibrated_{method}")
    report = {
        "version": version,
        "method": method,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **run_info(),
        "before": before,
        "after": after,
        "improved": improved,
        "calibrated_model_file": f"models/{calibrated_file.name}" if improved else None,
    }
    write_json(metrics_path(out_dir, f"{version}_calibration_{method}"), report)

    if improved:
        dump(calibrated, calibrated_file, compress=3)
        # Only point latest.json at the calibrated model if we calibrated the latest version.
        if latest is not None and latest["version"] == version:
            latest["calibrated_model_file"] = report["calibrated_model_file"]
            latest["calibration_method"] = method
            write_json(latest_path(out_dir), latest)

    add_to_job_summary(summary_markdown(report))
    return report


def summary_markdown(report):
    rows = [
        f"| {name} | {report['before'][name]} | {report['after'][name]} |"
        for name in report["before"]
    ]
    if report["improved"]:
        result = f"Saved as `{report['calibrated_model_file']}`"
    else:
        result = "Brier score did not improve, calibrated model was not saved"
    return "\n".join([
        f"### Calibration of model {report['version']} ({report['method']})",
        "",
        "| Metric (test set) | Before | After |",
        "|---|---|---|",
        *rows,
        "",
        f"**{result}**",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate a trained model's probabilities.")
    parser.add_argument("--method", choices=METHODS, default="sigmoid")
    parser.add_argument("--version", default=None, help="Defaults to the version in models/latest.json")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    report = calibrate(args.out_dir, args.method, args.version)

    print(f"Calibrating model {report['version']} with {report['method']}")
    print(f"Before: {report['before']}")
    print(f"After:  {report['after']}")
    if report["improved"]:
        print(f"Brier score improved. Saved {report['calibrated_model_file']}")
    else:
        annotate("warning", f"{report['method']} calibration did not improve the Brier score, model not saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
