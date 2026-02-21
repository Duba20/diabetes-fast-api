<<<<<<< HEAD
# ZTH ML - FastAPI Service

This folder hosts the ML model API (separate from your DL project).

## Run
```powershell
cd "C:\Users\HP\Desktop\ZTH ML"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --default-timeout=300 --retries 10 -r requirements.txt

# Optional: override model path
$env:ML_MODEL_PATH="C:\Users\HP\Desktop\ZTH ML\diabetes_prediction_model.pkl"

.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

## Docs
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
=======
# Diabetes Prediction API

A FastAPI-based REST API for diabetes risk prediction using a deep learning model.

## Features

- **Single Prediction**: Make predictions for individual samples
- **Batch Predictions**: Process multiple samples in a single request
- **Swagger Documentation**: Interactive API documentation at `/docs`
- **ReDoc Documentation**: Alternative documentation at `/redoc`
- **Health Check**: Monitor API status
- **Model Information**: Get details about the loaded model

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone/Download the project** and navigate to the project directory:
```bash
cd "ZTH Deep learning Project"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the API

Start the API server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. **Root Endpoint**
- **URL**: `GET /`
- **Description**: Welcome message and API information
- **Response**: API version and documentation links

### 2. **Health Check**
- **URL**: `GET /health`
- **Description**: Check if API and model are loaded
- **Response**: Server status and model status

### 3. **Model Information**
- **URL**: `GET /model-info`
- **Description**: Get details about the loaded model
- **Response**: Model type, shape, and file path

### 4. **Single Prediction**
- **URL**: `POST /predict`
- **Description**: Make a prediction for a single sample
- **Request Body**:
```json
{
  "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]
}
```
- **Response**:
```json
{
  "prediction": 1,
  "probability": 0.795,
  "risk_level": "High"
}
```

### 5. **Batch Predictions**
- **URL**: `POST /predict-batch`
- **Description**: Make predictions for multiple samples
- **Request Body**:
```json
{
  "data": [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31]
  ]
}
```
- **Response**:
```json
{
  "predictions": [1, 0],
  "probabilities": [0.795, 0.245],
  "risk_levels": ["High", "Low"],
  "count": 2
}
```

### 6. **Detailed Probability**
- **URL**: `POST /predict-proba`
- **Description**: Get detailed probability predictions
- **Request Body**:
```json
{
  "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]
}
```
- **Response**:
```json
{
  "probabilities": [[0.795]],
  "predicted_class": 1
}
```

## Using the Swagger UI

After starting the server, open your browser and navigate to:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

From the Swagger UI, you can:
- View all available endpoints
- Read detailed documentation for each endpoint
- Test endpoints directly from the browser
- See request/response examples
- View parameter descriptions

## Example Usage with cURL

### Single Prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}'
```

### Batch Predictions:
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"data": [[6, 148, 72, 35, 0, 33.6, 0.627, 50], [1, 85, 66, 29, 0, 26.6, 0.351, 31]]}'
```

## Example Usage with Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict-batch",
    json={
        "data": [
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31]
        ]
    }
)
print(response.json())
```

## Risk Levels

The API categorizes predictions into risk levels:
- **Low**: Probability < 0.3
- **Medium**: Probability 0.3 - 0.7
- **High**: Probability > 0.7

## Troubleshooting

### ModuleNotFoundError
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Not Found
Ensure `diabetes_model.h5` is in the same directory as `main.py`

### Port Already in Use
Use a different port:
```bash
uvicorn main:app --port 8001
```

### CORS Issues
To enable CORS, modify `main.py` to add:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Project Structure

```
ZTH Deep learning Project/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── diabetes_model.h5       # Pre-trained model
└── README.md              # This file
```

## License

This project uses a pre-trained diabetes prediction model.

## Support

For issues or questions, check the Swagger documentation at `/docs`
>>>>>>> e6d24ade02a8dade3a2abb10f73aee660e1ac407
