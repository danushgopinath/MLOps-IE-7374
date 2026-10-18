import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from joblib import load
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from src.common import (
    DEFAULT_OUT_DIR,
    add_to_job_summary,
    annotate,
    latest_path,
    load_splits,
    metrics_path,
    model_path,
    read_latest,
    run_info,
    write_json,
)


def score(model, X, y):
    """
    Scores a classifier on the given data.
    Returns:
        dict: F1, accuracy, ROC AUC and Brier score, rounded to 4 decimals.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "f1": round(float(f1_score(y, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
        "brier": round(float(brier_score_loss(y, y_prob)), 4),
    }


def quality_gate(candidate_f1, champion_f1, min_f1, max_drop):
    """
    Decides whether a new model is good enough to become the new version.
    Args:
        candidate_f1 (float): Test F1 of the new model.
        champion_f1 (float/None): Test F1 of the model in models/latest.json,
            or None if there is no model yet.
        min_f1 (float): The new model needs at least this F1.
        max_drop (float): The new model may be at most this much worse than the current one.
    Returns:
        tuple: (passed (bool), reason (str))
    """
    if candidate_f1 < min_f1:
        return False, f"F1 {candidate_f1:.4f} is below the minimum of {min_f1:.2f}"
    if champion_f1 is None:
        return True, f"F1 {candidate_f1:.4f} meets the minimum (first model)"
    if candidate_f1 < champion_f1 - max_drop:
        return False, (
            f"F1 {candidate_f1:.4f} is more than {max_drop} below "
            f"the current model's F1 of {champion_f1:.4f}"
        )
    return True, (
        f"F1 {candidate_f1:.4f} meets the minimum and is not more than {max_drop} "
        f"below the current model's F1 of {champion_f1:.4f}"
    )


def evaluate(version, out_dir=DEFAULT_OUT_DIR, min_f1=0.85, max_drop=0.01):
    """
    Scores the new model and the current model (from models/latest.json) on the
    same test set, applies the quality gate, and writes metrics/<version>_metrics.json.
    If the gate passes, models/latest.json is updated to point to the new model.
    Returns:
        dict: The evaluation report that was written to the metrics file.
    """
    X_test, y_test = load_splits()["test"]

    candidate = load(model_path(out_dir, version))
    candidate_metrics = score(candidate, X_test, y_test)

    champion = None
    latest = read_latest(out_dir)
    if latest is not None:
        champion_file = Path(out_dir) / latest["model_file"]
        if champion_file.exists():
            champion = {
                "version": latest["version"],
                "test_metrics": score(load(champion_file), X_test, y_test),
            }
        else:
            annotate("warning", f"{champion_file} is missing, comparing against the minimum F1 only")

    champion_f1 = champion["test_metrics"]["f1"] if champion else None
    passed, reason = quality_gate(candidate_metrics["f1"], champion_f1, min_f1, max_drop)

    report = {
        "version": version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **run_info(),
        "params": {
            "n_estimators": candidate.n_estimators,
            "max_depth": candidate.max_depth,
        },
        "test_metrics": candidate_metrics,
        "champion": champion,
        "gate": {"min_f1": min_f1, "max_drop": max_drop, "passed": passed, "reason": reason},
    }
    write_json(metrics_path(out_dir, f"{version}_metrics"), report)

    if passed:
        write_json(latest_path(out_dir), {
            "version": version,
            "model_file": f"models/{model_path(out_dir, version).name}",
            "test_f1": candidate_metrics["f1"],
            "previous_version": latest["version"] if latest else None,
            "promoted_at": report["evaluated_at"],
            **run_info(),
        })

    add_to_job_summary(summary_markdown(report))
    return report


def summary_markdown(report):
    champion = report["champion"]
    header = "| Metric | New model |"
    divider = "|---|---|"
    if champion:
        header += f" Current model ({champion['version']}) |"
        divider += "---|"

    rows = []
    for name, value in report["test_metrics"].items():
        row = f"| {name} | {value} |"
        if champion:
            row += f" {champion['test_metrics'][name]} |"
        rows.append(row)

    gate = report["gate"]
    result = "PASSED" if gate["passed"] else "FAILED"
    return "\n".join([
        f"### Model version {report['version']}",
        f"Parameters: `{report['params']}`",
        "",
        header,
        divider,
        *rows,
        "",
        f"**Quality gate {result}** (min F1 {gate['min_f1']}, max drop {gate['max_drop']}): {gate['reason']}",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a model version and apply the quality gate.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--max-drop", type=float, default=0.01)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    report = evaluate(args.version, args.out_dir, args.min_f1, args.max_drop)

    print(f"New model {args.version}: {report['test_metrics']}")
    if report["champion"]:
        print(f"Current model {report['champion']['version']}: {report['champion']['test_metrics']}")
    else:
        print("No current model yet, only the minimum F1 is checked.")

    if report["gate"]["passed"]:
        print(f"Quality gate passed. models/latest.json now points to {args.version}")
        return 0
    annotate("error", f"Quality gate failed: {report['gate']['reason']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
