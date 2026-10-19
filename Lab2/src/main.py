import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.config import MAX_BATCH_SIZE
from src.predict import load_model, load_model_info, predict_data

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, instead of on every request."""
    try:
        app.state.model = load_model()
    except FileNotFoundError:
        logger.error("Model file not found. Run 'python -m src.train' first.")
        app.state.model = None
    app.state.model_info = load_model_info()
    yield


app = FastAPI(
    title="Iris Classifier API",
    description="Predicts the iris species from sepal and petal measurements.",
    version="2.0.0",
    lifespan=lifespan,
)


class IrisData(BaseModel):
    """
    Iris flower measurements in centimeters.
    Each value must be greater than 0 and at most 10. Unknown fields are rejected.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        },
    )

    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")

    def to_row(self):
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class IrisResponse(BaseModel):
    class_id: int
    species: str
    confidence: float
    probabilities: dict[str, float]


class IrisBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[IrisData] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    count: int
    predictions: list[IrisResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    model_type: str
    params: dict
    test_accuracy: float
    features: list[str]
    classes: list[str]
    sklearn_version: str
    trained_at: str


def get_model(request: Request):
    """Dependency that gives an endpoint the loaded model, or returns 503 if there isn't one."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python -m src.train' and restart the server.",
        )
    return model


def run_prediction(model, rows):
    try:
        return predict_data(model, rows)
    except Exception:
        # Log the real error on the server, but don't send internal details to the client
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed."
        )


# The endpoints below use plain "def" instead of "async def". Prediction is CPU work,
# so FastAPI runs these in its thread pool instead of blocking the event loop.

@app.get("/", response_model=HealthResponse)
def health_ping(model=Depends(get_model)):
    """Health check. Returns 503 if the model could not be loaded."""
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=IrisResponse)
def predict_iris(iris_features: IrisData, model=Depends(get_model)):
    """
    Predict the iris species for one flower.
    Returns the class id, species name, confidence and the probability of each species.
    """
    return run_prediction(model, [iris_features.to_row()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: IrisBatch, model=Depends(get_model)):
    """Predict the species for up to 100 flowers in one request."""
    predictions = run_prediction(model, [sample.to_row() for sample in batch.samples])
    return BatchResponse(count=len(predictions), predictions=predictions)


@app.get("/model/info", response_model=ModelInfo)
def model_info(request: Request, model=Depends(get_model)):
    """Details about the loaded model: type, parameters, test accuracy and when it was trained."""
    info = request.app.state.model_info
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model details not found. Run 'python -m src.train' to create them.",
        )
    return info
