# Lab 2 - GitHub Actions for Model Training, Versioning and Calibration

In this lab, GitHub Actions retrains a random forest classifier when the Lab 2 code is pushed, checks if the new model is good enough, saves it in the repo with a timestamp version, and then calibrates it in a second workflow.

Original lab: https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2

## My changes

Workflows (moved from `Lab2/workflows/` to `.github/workflows/`, since GitHub only runs workflows from there):

- `lab2_model_retraining.yml` now has three jobs: `test` runs pytest, `train` trains and evaluates the model, and `publish` commits the model and metrics to the repo and tags the commit as `lab2-model-<timestamp>`.
- Added a quality gate to the `train` job. The new model is only published if its test F1 is at least 0.85 and not more than 0.01 below the current model's F1. Otherwise the job fails and nothing is committed.
- The workflow can also be run manually from the Actions tab with `n_estimators`, `max_depth` and `min_f1` as inputs.
- It only runs when `Lab2/src`, `Lab2/test`, `Lab2/requirements.txt` or the workflow file change.
- Added `permissions: contents: write` so it can push. The model and metrics are uploaded as an artifact even if the gate fails.
- New `lab2_model_calibration.yml` runs after a successful retraining run (or manually). It calibrates the latest model (sigmoid or isotonic) and commits the calibrated model only if the Brier score improves. The original lab didn't have any calibration code.
- Removed the scheduled `model_calibration.yml`, which used wrong paths and a variable that was never set.

Python code:

- `src/common.py`: fixed dataset (seed 42) split into train, calibration and test sets, plus shared helper functions.
- `src/train_model.py`: trains on the training split only.
- `src/evaluate_model.py`: compares the new model with the current one on the test set, applies the gate, writes `metrics/<version>_metrics.json` and updates `models/latest.json`.
- `src/calibrate_model.py`: new calibration script.
- `test/test_pipeline.py`: 14 pytest tests.
- Fixed two bugs in the original: the dataset size was random and could be 0 (crash), and the evaluation data was generated differently from the training data, so the F1 score didn't mean anything.
- Pinned package versions and removed mlflow (it only logged to a folder that was deleted after each run), pandas, ipykernel and the unused notebook.

## Requirements

- Git
- Python 3.12 or newer (numpy 2.5.4 doesn't support 3.11)
- A GitHub account, to run the workflows

Packages (installed from `requirements.txt`): scikit-learn 1.9.1, numpy 2.5.4, joblib 1.6.0, pytest 9.1.1

## Running it locally

```bash
git clone https://github.com/danushgopinath/MLOps-IE-7374.git
cd MLOps-IE-7374/Lab2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `python` instead of `python3`, and `.venv\Scripts\activate` to activate.

Run the tests:

```bash
pytest -v
```

All 14 tests should pass.

Train, evaluate and calibrate a model:

```bash
python -m src.train_model --version local1
python -m src.evaluate_model --version local1
python -m src.calibrate_model
```

Local runs save to `Lab2/outputs/`, which git ignores. You should see an F1 of about 0.94, "Quality gate passed", and the Brier score dropping after calibration (about 0.050 to 0.047).

## Running it on GitHub Actions

1. Fork the repo and enable workflows in the Actions tab.
2. In Settings > Actions > General > Workflow permissions, select "Read and write permissions" and save.
3. Go to Actions > Lab2 Model Retraining > Run workflow, or push a change to `Lab2/src`.

A successful run looks like this:

- Lab2 Model Retraining shows `test`, `train` and `publish` all green, with a metrics table on the run's summary page.
- A commit from github-actions[bot] adds `Lab2/models/model_<timestamp>_rf.joblib` and `Lab2/metrics/<timestamp>_metrics.json`, updates `Lab2/models/latest.json`, and a tag `lab2-model-<timestamp>` is created.
- Lab2 Model Calibration then runs by itself and commits the calibrated model.

To see the gate reject a model, run the retraining workflow with `max_depth` set to 1. The `train` job fails at the quality gate step and nothing is committed.
