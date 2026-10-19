# Lab 2 - Serving an ML Model with FastAPI

This lab trains a decision tree classifier on the Iris dataset and serves it as a REST API using FastAPI and uvicorn.

Original lab: https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs

## My changes

API (`src/main.py`):

- The model is loaded once when the server starts (lifespan event) instead of being read from disk on every request.
- Added input validation. Every measurement must be greater than 0 and at most 10 cm, and unknown fields are rejected, so bad input gets a 422 error with the reason. Before, negative values were accepted and still got a prediction.
- `/predict` now returns the species name, the confidence and the probability of each species, instead of just a class number.
- New `POST /predict/batch` endpoint to predict up to 100 flowers in one request.
- New `GET /model/info` endpoint that returns the model type, parameters, test accuracy and when it was trained.
- Endpoints get the model through a FastAPI dependency (`Depends(get_model)`), which returns 503 if the model isn't loaded. The health check (`GET /`) also returns 503 in that case, instead of always saying healthy.
- If a prediction fails, the client gets a plain 500 message and the real error is logged on the server, instead of sending the Python error text back.
- The endpoints use `def` instead of `async def`. Prediction is CPU work, so FastAPI runs them in its thread pool instead of blocking the event loop.
- Added an example request to the schema, so the `/docs` page is pre-filled.

Other changes:

- `src/config.py` (new): file paths are based on the file's location. The original used `../model/...` and `from predict import ...`, which only worked when the server was started from inside `src/`.
- `src/train.py`: prints the test accuracy and saves `model/model_info.json` next to the model.
- `test/test_api.py` and `test/test_train.py` (new): 17 pytest tests using FastAPI's TestClient.
- Updated `requirements.txt` and retrained the saved model. The old `scikit-learn==1.5.1` doesn't install on Python 3.13 or newer.
- Fixed the run command (the original README said `uvicorn app:main`). It now uses `python -m uvicorn`, so it always runs the uvicorn from the virtual environment. I also removed the screenshots of the old API.

## Requirements

- Git
- Python 3.11 or newer (tested on 3.11, 3.12 and 3.14)

Packages (installed from `requirements.txt`): fastapi 0.146.0, uvicorn 0.54.0, scikit-learn 1.9.1, joblib 1.6.0, pytest 9.1.1, httpx2 2.14.0 (needed by the test client)

## Setup

```bash
git clone https://github.com/danushgopinath/MLOps-IE-7374.git
cd MLOps-IE-7374/Lab2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `python` instead of `python3`, and `.venv\Scripts\activate` to activate.

Run all the commands below from the `Lab2` folder.

## Train the model

A trained model is already in `model/`, so this step is optional.

```bash
python -m src.train
```

It should print `Test accuracy: 0.9556` and where the model was saved.

## Run the tests

```bash
pytest -v
```

All 17 tests should pass.

## Start the API

```bash
python -m uvicorn src.main:app --reload
```

You should see `Application startup complete.` and `Uvicorn running on http://127.0.0.1:8000`. Stop it with Ctrl+C.

Endpoints:

- `GET /` - health check
- `POST /predict` - predict one flower
- `POST /predict/batch` - predict up to 100 flowers, sent as `{"samples": [...]}`
- `GET /model/info` - details about the loaded model

The easiest way to try them is the docs page at http://127.0.0.1:8000/docs. Click an endpoint, then "Try it out" and "Execute".

Or with curl from a second terminal (Mac/Linux):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.6}'
```

Response:

```json
{"class_id":2,"species":"virginica","confidence":0.6667,"probabilities":{"setosa":0.0,"versicolor":0.3333,"virginica":0.6667}}
```

Sending an invalid value, for example `"sepal_length": -1`, returns a 422 error with `"Input should be greater than 0"`.
