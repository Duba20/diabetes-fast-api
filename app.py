from __future__ import annotations

import os
import pickle
import sys
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

FEATURE_NAMES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
DEFAULT_FEATURES: List[float] = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
MISSING_INDICATOR_FEATURES: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

MODEL_PATH = os.getenv(
    "ML_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "diabetes_prediction_model.pkl"),
)


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="8 diabetes features in this exact order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age",
        examples=[DEFAULT_FEATURES],
    )

    @field_validator("features")
    @classmethod
    def ensure_numeric(cls, value: List[float]) -> List[float]:
        return [float(v) for v in value]


class BatchPredictionInput(BaseModel):
    data: List[List[float]] = Field(
        ...,
        min_length=1,
        description="List of rows, each with exactly 8 features",
        examples=[[DEFAULT_FEATURES]],
    )

    @field_validator("data")
    @classmethod
    def ensure_shape(cls, value: List[List[float]]) -> List[List[float]]:
        normalized: List[List[float]] = []
        for idx, row in enumerate(value):
            if len(row) != 8:
                raise ValueError(f"Row {idx} must contain exactly 8 features")
            normalized.append([float(v) for v in row])
        return normalized


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    risk_levels: List[str]
    count: int


class ArtifactRunner:
    def __init__(self, artifact: Any):
        self.artifact = artifact
        self.model = artifact
        self.scaler = None
        self.imputation_medians = None
        self.expected_features = 8

        if isinstance(artifact, dict):
            self.model = artifact.get("model")
            self.scaler = artifact.get("scaler")
            self.imputation_medians = artifact.get("imputation_medians")

        if self.model is None:
            raise ValueError("Loaded artifact does not contain a usable model")

        for obj in (self.model, self.scaler):
            if obj is not None and hasattr(obj, "n_features_in_"):
                self.expected_features = int(getattr(obj, "n_features_in_"))
                break

    @property
    def base_feature_count(self) -> int:
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            return int(getattr(self.scaler, "n_features_in_"))
        return 8

    def _build_missing_indicators(self, raw_x: np.ndarray) -> np.ndarray:
        indicators: List[np.ndarray] = []
        for feature_name in MISSING_INDICATOR_FEATURES:
            if feature_name in FEATURE_NAMES[: raw_x.shape[1]]:
                idx = FEATURE_NAMES.index(feature_name)
                col = raw_x[:, idx]
                is_missing = np.isnan(col) | (col == 0.0)
                indicators.append(is_missing.astype(np.float32).reshape(-1, 1))
        if not indicators:
            return np.zeros((raw_x.shape[0], 0), dtype=np.float32)
        return np.concatenate(indicators, axis=1)

    def _apply_boxcox_transforms(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(self.artifact, dict):
            return x
        lambdas = self.artifact.get("transformation_lambdas")
        if not isinstance(lambdas, dict):
            return x

        for feature_name, lam in lambdas.items():
            if feature_name not in FEATURE_NAMES[: x.shape[1]]:
                continue
            idx = FEATURE_NAMES.index(feature_name)
            col = x[:, idx].astype(np.float64)
            col = np.where(col <= 0.0, 1e-6, col)
            lam_value = float(lam)
            if abs(lam_value) < 1e-12:
                transformed = np.log(col)
            else:
                transformed = (np.power(col, lam_value) - 1.0) / lam_value
            x[:, idx] = transformed.astype(np.float32)
        return x

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Input must be 2D")

        base_feature_count = self.base_feature_count

        if x.shape[1] != base_feature_count:
            raise ValueError(
                f"Input must contain {base_feature_count} base features but got {x.shape[1]}"
            )

        raw_x = x.copy()
        indicators = self._build_missing_indicators(raw_x)

        if isinstance(self.imputation_medians, dict):
            for idx, feature in enumerate(FEATURE_NAMES[: base_feature_count]):
                median_value = self.imputation_medians.get(feature)
                if median_value is None:
                    continue
                col = x[:, idx]
                if feature in MISSING_INDICATOR_FEATURES:
                    mask = np.isnan(col) | (col == 0.0)
                else:
                    mask = np.isnan(col)
                if np.any(mask):
                    col[mask] = float(median_value)
                    x[:, idx] = col

        x = self._apply_boxcox_transforms(x)

        if self.scaler is not None and hasattr(self.scaler, "transform"):
            x = self.scaler.transform(x)

        x = np.asarray(x, dtype=np.float32)
        if x.shape[1] < self.expected_features:
            extra_needed = self.expected_features - x.shape[1]
            if indicators.shape[1] < extra_needed:
                pad = np.zeros((indicators.shape[0], extra_needed - indicators.shape[1]), dtype=np.float32)
                indicators = np.concatenate([indicators, pad], axis=1)
            x = np.concatenate([x, indicators[:, :extra_needed]], axis=1)
        elif x.shape[1] > self.expected_features:
            x = x[:, : self.expected_features]

        return x

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        x = self._preprocess(x)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)
            probs = np.asarray(probs)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1].astype(float)
            return probs.reshape(-1).astype(float)

        preds = np.asarray(self.model.predict(x)).reshape(-1)
        return preds.astype(float)


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def normalize_feature_row(row: List[float], expected_feature_count: int) -> List[float]:
    values = [float(v) for v in row]
    if len(values) == expected_feature_count:
        return values
    if len(values) < expected_feature_count:
        return values + [0.0] * (expected_feature_count - len(values))
    return values[:expected_feature_count]


def load_runner(model_path: str) -> tuple[Optional[ArtifactRunner], Optional[str]]:
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"

    try:
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np.core
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        runner = ArtifactRunner(artifact)
        return runner, None
    except Exception as exc:
        return None, f"Error loading model artifact: {str(exc)}"


runner, model_load_error = load_runner(MODEL_PATH)

app = FastAPI(
    title="Diabetes ML Prediction API",
    description="FastAPI service for a classical ML diabetes model with Swagger UI docs.",
    version="1.0.0",
    openapi_tags=[
        {"name": "system", "description": "Service status and model metadata"},
        {"name": "prediction", "description": "Single and batch prediction endpoints"},
    ],
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "message": "Welcome to Diabetes ML Prediction API",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {
        "status": "healthy" if runner is not None else "degraded",
        "model_loaded": runner is not None,
        "model_load_error": model_load_error,
        "model_path": MODEL_PATH,
        "expected_feature_count": runner.expected_features if runner else 8,
    }


@app.get("/model-info", tags=["system"])
def model_info() -> dict[str, Any]:
    return {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": runner is not None,
        "model_load_error": model_load_error,
        "model_type": str(type(runner.model)) if runner else None,
        "scaler_type": str(type(runner.scaler)) if runner and runner.scaler is not None else None,
        "expected_feature_count": runner.expected_features if runner else 8,
        "feature_names": FEATURE_NAMES,
    }


@app.get("/feature-schema", tags=["system"])
def feature_schema() -> dict[str, Any]:
    return {
        "feature_names": FEATURE_NAMES,
        "example": DEFAULT_FEATURES,
        "notes": "Provide only these 8 features. No additional indicator columns are required.",
    }


def require_runner() -> ArtifactRunner:
    if runner is None:
        raise HTTPException(status_code=503, detail=model_load_error or "Model not loaded")
    return runner


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionInput) -> PredictionResponse:
    try:
        model_runner = require_runner()
        normalized = normalize_feature_row(payload.features, model_runner.base_feature_count)
        x = np.array(normalized, dtype=np.float32).reshape(1, -1)
        probability = float(model_runner.predict_probability(x)[0])
        prediction = int(probability >= 0.5)
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=get_risk_level(probability),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionInput) -> BatchPredictionResponse:
    try:
        model_runner = require_runner()
        normalized_rows = [
            normalize_feature_row(row, model_runner.base_feature_count)
            for row in payload.data
        ]
        x = np.array(normalized_rows, dtype=np.float32)
        probabilities = model_runner.predict_probability(x)
        probabilities_list = [float(v) for v in probabilities]
        predictions = [int(v >= 0.5) for v in probabilities_list]
        risk_levels = [get_risk_level(v) for v in probabilities_list]
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities_list,
            risk_levels=risk_levels,
            count=len(predictions),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
