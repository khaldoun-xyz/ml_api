"""Tests for the FastAPI application."""

import pytest


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Loan Approval API"
        assert data["version"] == "1.0.0"
        assert "description" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"

    def test_health_response_schema(self, client):
        """Test that health response matches expected schema."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_with_valid_features(self, client):
        """Test prediction with valid input features."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "approve" in data
        assert isinstance(data["approve"], bool)

    def test_predict_with_minimum_valid_values(self, client):
        """Test prediction with minimum valid values."""
        payload = {
            "income": 0.01,  # Just above 0
            "credit_score": 250,  # Minimum allowed
            "loan_amount": 0.01,  # Just above 0
            "years_employed": 0,  # Minimum allowed
            "points": 0,  # Minimum allowed
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["approve"], bool)

    def test_predict_with_maximum_valid_values(self, client):
        """Test prediction with maximum valid values."""
        payload = {
            "income": 1000000,
            "credit_score": 900,  # Maximum allowed
            "loan_amount": 5000000,
            "years_employed": 50,
            "points": 1000,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["approve"], bool)

    def test_predict_with_invalid_income_zero(self, client):
        """Test prediction fails with zero income."""
        payload = {
            "income": 0,  # Invalid: must be > 0
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_income_negative(self, client):
        """Test prediction fails with negative income."""
        payload = {
            "income": -50000,  # Invalid: must be > 0
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_credit_score_too_low(self, client):
        """Test prediction fails with credit score below minimum."""
        payload = {
            "income": 50000,
            "credit_score": 249,  # Invalid: must be >= 250
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_credit_score_too_high(self, client):
        """Test prediction fails with credit score above maximum."""
        payload = {
            "income": 50000,
            "credit_score": 901,  # Invalid: must be <= 900
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_loan_amount_zero(self, client):
        """Test prediction fails with zero loan amount."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 0,  # Invalid: must be > 0
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_years_employed_negative(self, client):
        """Test prediction fails with negative years employed."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": -1,  # Invalid: must be >= 0
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_invalid_points_negative(self, client):
        """Test prediction fails with negative points."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": 5,
            "points": -10,  # Invalid: must be >= 0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_with_missing_field(self, client):
        """Test prediction fails with missing required field."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 200000,
            # Missing years_employed and points
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_response_schema(self, client):
        """Test that prediction response matches expected schema."""
        payload = {
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 200000,
            "years_employed": 5,
            "points": 100,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "approve" in data
        assert len(data) == 1  # Only approve field in response
