"""
FastAPI webservice for the REST API.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Loan Approval API",
    description="API for predicting loan approval based on applicant information",
    version="1.0.0",
)

FEATURE_NAMES = [
    "income",
    "credit_score",
    "loan_amount",
    "years_employed",
    "points",
]


class LoanFeatures(BaseModel):
    """Input features for loan approval prediction."""

    income: float = Field(..., gt=0, description="Annual income (>0)")
    credit_score: int = Field(..., ge=250, le=900, description="Credit score (300-850)")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount (>0)")
    years_employed: float = Field(..., ge=0, description="Years at current job (>=0)")
    points: float = Field(..., ge=0, description="Applicant points score (>=0)")


class PredictionResponse(BaseModel):
    """Response model for loan approval prediction."""

    approve: bool = Field(..., description="Loan approval prediction")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")


MODEL_PATH = Path(__file__).parent.parent / "models" / "loan_model.joblib"

model = None
model_loaded = False

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Please train the model first."
    )


@app.get("/")
def root():
    """Root endpoint returning API information."""
    return {
        "name": "Loan Approval API",
        "version": "1.0.0",
        "description": "API for predicting loan approval based on applicant information",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint for monitoring systems.

    Returns:
        HealthResponse containing health status and model information
    """
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: LoanFeatures) -> PredictionResponse:
    """Predict loan approval based on input features.

    Args:
        features: Input features for prediction

    Returns:
        Prediction response containing approval decision

    Raises:
        HTTPException: If prediction fails
    """
    try:
        input_data = [features.model_dump()]
        input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
        prediction = model.predict(input_df)[0]
        # Convert numeric prediction (0/1) to boolean
        approve = bool(prediction)

        return PredictionResponse(approve=approve)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed.",
        )
