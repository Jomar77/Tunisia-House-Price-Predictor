"""
Pydantic schemas for API request/response validation.
Primary adapter (driving) - FastAPI layer.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any


class PredictionRequest(BaseModel):
    """Request schema for price prediction."""
    
    area: float = Field(..., gt=0, description="House area in square meters")
    rooms: float = Field(..., gt=0, description="Number of rooms")
    bathrooms: float = Field(..., gt=0, description="Number of bathrooms")
    age: float = Field(..., ge=0, description="Age of the house in years")
    location: str = Field(..., min_length=1, description="Location name")
    
    @field_validator('location')
    @classmethod
    def strip_location(cls, v: str) -> str:
        """Normalize location by stripping whitespace."""
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area": 150.0,
                    "rooms": 3.0,
                    "bathrooms": 2.0,
                    "age": 5.0,
                    "location": "Hammamet"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for price prediction."""
    
    predicted_price: float = Field(..., description="Predicted price in EUR")
    inputs: Dict[str, Any] = Field(..., description="Echo of input values")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 250000.0,
                    "inputs": {
                        "area": 150.0,
                        "rooms": 3.0,
                        "bathrooms": 2.0,
                        "age": 5.0,
                        "location": "Hammamet"
                    }
                }
            ]
        }
    }


class MetadataResponse(BaseModel):
    """Response schema for model metadata."""
    
    numeric_features: List[str] = Field(..., description="List of numeric feature names")
    supported_locations: List[str] = Field(..., description="Valid location names (UI should hide 'other')")
    model_version: str = Field(..., description="Model artifact version/timestamp")
    n_features: int = Field(..., description="Total number of features")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "numeric_features": ["area", "rooms", "bathrooms", "age"],
                    "supported_locations": ["Hammamet", "La Marsa", "..."],
                    "model_version": "2026-01-19T00:00:00Z",
                    "n_features": 64
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    detail: str = Field(..., description="Error message")
    error_type: str = Field(default="validation_error", description="Error category")
