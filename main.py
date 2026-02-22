from typing import List
import os
import json
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import h5py
from tensorflow import keras
import traceback


MODEL_PATH = "diabetes_model.h5"
EXPECTED_FEATURE_COUNT = 8
LEGACY_FEATURE_COUNT = 8
model = None
model_load_error = None
RENDER_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")


def build_example_features(feature_count: int) -> List[float]:
    base_example = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    if feature_count <= len(base_example):
        return [float(v) for v in base_example[:feature_count]]
    return [float(v) for v in (base_example + [0] * (feature_count - len(base_example)))]

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes using a deep learning model",
    version="1.1.0",
    openapi_tags=[
        {"name": "system", "description": "Service and model health endpoints"},
        {"name": "prediction", "description": "Single and batch prediction endpoints"},
    ],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Unhandled error: {str(exc)}",
            "error_type": type(exc).__name__,
            "traceback_tail": traceback.format_exc().splitlines()[-5:],
        },
    )

def infer_expected_feature_count(model_path: str, fallback: int = 8) -> int:
    if not os.path.exists(model_path):
        return fallback

    try:
        with h5py.File(model_path, "r") as h5_file:
            model_config = h5_file.attrs.get("model_config")
            if model_config is None:
                return fallback
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            config = json.loads(model_config)
            for layer in config.get("config", {}).get("layers", []):
                if layer.get("class_name") == "InputLayer":
                    layer_cfg = layer.get("config", {})
                    batch_shape = layer_cfg.get("batch_shape") or layer_cfg.get(
                        "batch_input_shape"
                    )
                    if (
                        isinstance(batch_shape, list)
                        and len(batch_shape) > 1
                        and batch_shape[1] is not None
                    ):
                        return int(batch_shape[1])
    except Exception:
        return fallback

    return fallback


EXPECTED_FEATURE_COUNT = infer_expected_feature_count(MODEL_PATH, fallback=8)
EXAMPLE_FEATURES = build_example_features(EXPECTED_FEATURE_COUNT)
LEGACY_EXAMPLE_FEATURES = build_example_features(LEGACY_FEATURE_COUNT)


def normalize_feature_row(row: List[float]) -> List[float]:
    row = [float(v) for v in row]
    if len(row) == EXPECTED_FEATURE_COUNT:
        return row
    if len(row) < EXPECTED_FEATURE_COUNT:
        return row + [0.0] * (EXPECTED_FEATURE_COUNT - len(row))
    return row[:EXPECTED_FEATURE_COUNT]


def patch_h5_model_for_compatibility(source_path: str, patched_path: str) -> None:
    """
    Patch legacy InputLayer config key from `batch_shape` to `batch_input_shape`
    in a copied .h5 file so current TF/Keras can deserialize it.
    """
    shutil.copy2(source_path, patched_path)
    with h5py.File(patched_path, "r+") as h5_file:
        model_config = h5_file.attrs.get("model_config")
        if model_config is None:
            return

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        config = json.loads(model_config)

        changed = False

        def normalize_dtype_policy(obj):
            nonlocal changed
            if isinstance(obj, dict):
                if (
                    obj.get("class_name") == "DTypePolicy"
                    and isinstance(obj.get("config"), dict)
                    and "name" in obj["config"]
                ):
                    changed = True
                    return obj["config"]["name"]
                return {k: normalize_dtype_policy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [normalize_dtype_policy(v) for v in obj]
            return obj

        for layer in config.get("config", {}).get("layers", []):
            if layer.get("class_name") != "InputLayer":
                continue
            layer_cfg = layer.get("config", {})
            if "batch_shape" in layer_cfg and "batch_input_shape" not in layer_cfg:
                layer_cfg["batch_input_shape"] = layer_cfg.pop("batch_shape")
                changed = True

        config = normalize_dtype_policy(config)

        if changed:
            h5_file.attrs.modify("model_config", json.dumps(config))


def load_model_with_fallback(model_path: str):
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"

    try:
        return keras.models.load_model(model_path, compile=False), None
    except Exception as first_error:
        try:
            patched_path = os.path.splitext(model_path)[0] + "_compat.h5"
            patch_h5_model_for_compatibility(model_path, patched_path)
            return keras.models.load_model(patched_path, compile=False), None
        except Exception as second_error:
            return None, f"Error loading model: {str(first_error)} | fallback failed: {str(second_error)}"


model, model_load_error = load_model_with_fallback(MODEL_PATH)


class PredictionInput(BaseModel):
    """Input features for a single diabetes prediction."""

    features: List[float] = Field(
        ...,
        description=(
            f"Numeric features expected by the model ({EXPECTED_FEATURE_COUNT} values). "
            f"Legacy {LEGACY_FEATURE_COUNT}-value payloads are auto-padded with zeros."
        ),
        examples=[EXAMPLE_FEATURES, LEGACY_EXAMPLE_FEATURES],
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"features": EXAMPLE_FEATURES}}
    )

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, value: List[float]) -> List[float]:
        return normalize_feature_row(value)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str


class BatchPredictionInput(BaseModel):
    """Input for batch predictions."""

    data: List[List[float]] = Field(
        ...,
        min_length=1,
        description=f"List of rows, each row with exactly {EXPECTED_FEATURE_COUNT} features",
        examples=[
            [
                EXAMPLE_FEATURES,
                LEGACY_EXAMPLE_FEATURES,
            ]
        ],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    EXAMPLE_FEATURES,
                    LEGACY_EXAMPLE_FEATURES,
                ]
            }
        }
    )

    @field_validator("data")
    @classmethod
    def validate_row_lengths(cls, value: List[List[float]]) -> List[List[float]]:
        normalized_rows = []
        for idx, row in enumerate(value):
            try:
                normalized_rows.append(normalize_feature_row(row))
            except ValueError:
                raise ValueError(
                    f"Row {idx} must contain exactly {EXPECTED_FEATURE_COUNT} features, got {len(row)}"
                )
        return normalized_rows


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    risk_levels: List[str]
    count: int


class ProbabilityResponse(BaseModel):
    probabilities: List[List[float]]
    predicted_class: int


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def require_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error
            or "Model is not available. Check /health and /model-info.",
        )


@app.get("/", tags=["system"])
async def root():
    return {
        "message": "Welcome to Diabetes Prediction API",
        "version": "1.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "build_commit": RENDER_COMMIT,
    }


@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "build_commit": RENDER_COMMIT,
    }


@app.get("/model-info", tags=["system"])
async def model_info():
    return {
        "model_type": str(type(model)),
        "input_shape": model.input_shape if model is not None else None,
        "output_shape": model.output_shape if model is not None else None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict diabetes risk for one sample",
)
async def predict(input_data: PredictionInput):
    try:
        require_model_loaded()
        features = np.array(input_data.features, dtype=np.float32).reshape(1, -1)
        prediction = model.predict(features, verbose=0)
        prediction_values = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if prediction_values.size == 0:
            raise ValueError("Model returned an empty prediction array")
        probability = float(prediction_values[0])
        prediction_class = int(round(probability))

        return PredictionResponse(
            prediction=prediction_class,
            probability=probability,
            risk_level=get_risk_level(probability),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    summary="Predict diabetes risk for multiple samples",
)
async def predict_batch(batch_input: BatchPredictionInput):
    try:
        require_model_loaded()
        features = np.array(batch_input.data, dtype=np.float32)
        predictions = model.predict(features, verbose=0)
        prediction_values = np.asarray(predictions, dtype=np.float32)
        if prediction_values.ndim == 1:
            probabilities = [float(p) for p in prediction_values]
        else:
            probabilities = [float(p[0]) for p in prediction_values]
        prediction_classes = [int(round(p)) for p in probabilities]
        risk_levels = [get_risk_level(p) for p in probabilities]

        return BatchPredictionResponse(
            predictions=prediction_classes,
            probabilities=probabilities,
            risk_levels=risk_levels,
            count=len(prediction_classes),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post(
    "/predict-proba",
    response_model=ProbabilityResponse,
    tags=["prediction"],
    summary="Return raw model probabilities",
)
async def predict_probability(input_data: PredictionInput):
    try:
        require_model_loaded()
        features = np.array(input_data.features, dtype=np.float32).reshape(1, -1)
        prediction = model.predict(features, verbose=0)
        prediction_values = np.asarray(prediction, dtype=np.float32)
        prediction_class = int(round(float(prediction_values.reshape(-1)[0])))
        return ProbabilityResponse(
            probabilities=prediction_values.tolist(),
            predicted_class=prediction_class,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Diabetes Prediction API",
        version="1.1.0",
        description="Deep Learning API for diabetes risk prediction with comprehensive Swagger documentation",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
