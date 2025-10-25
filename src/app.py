"""
FastAPI webservice for the REST API.
"""

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Loan Approval API",
    description="API for predicting loan approval based on applicant information",
    version="1.0.0",
)


class LoanFeatures(BaseModel):
    """Input features for loan approval prediction."""

    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score")
    loan_amount: float = Field(..., description="Requested loan amount")
    years_employed: float = Field(..., description="Years at current job")
    points: float = Field(..., description="Applicant points score")


class PredictionResponse(BaseModel):
    """Response model for loan approval prediction."""

    approve: bool = Field(..., description="Loan approval prediction")


MODEL_PATH = Path(__file__).parent.parent / "models" / "loan_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
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
        feature_list = [
            features.income,
            features.credit_score,
            features.loan_amount,
            features.years_employed,
            features.points,
        ]

        # Model expects features as a 2D array
        prediction = model.predict([feature_list])[0]

        # Convert numeric prediction (0/1) to boolean
        approve = bool(prediction)

        return PredictionResponse(approve=approve)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
