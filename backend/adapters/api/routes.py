"""
FastAPI route handlers.
Primary adapter (driving) - HTTP interface.
"""
from fastapi import APIRouter, HTTPException, Depends
from .schemas import PredictionRequest, PredictionResponse, MetadataResponse, ErrorResponse
from ...domain.predictor import PricePredictorService
from typing import Annotated
import json


router = APIRouter(prefix="/api/v1", tags=["predictions"])


# Dependency injection for predictor service
def get_predictor() -> PricePredictorService:
    """
    Get the predictor service instance.
    This will be replaced with proper DI container in main.py
    """
    from ...main import predictor_service
    return predictor_service


PredictorDep = Annotated[PricePredictorService, Depends(get_predictor)]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
    summary="Predict house price",
    description="Predict house price based on area, rooms, bathrooms, age, and location."
)
async def predict_price(
    request: PredictionRequest,
    predictor: PredictorDep
) -> PredictionResponse:
    """
    Predict house price endpoint.
    
    Returns predicted price in EUR based on input features.
    """
    try:
        result = predictor.predict_price(
            area=request.area,
            rooms=request.rooms,
            bathrooms=request.bathrooms,
            age=request.age,
            location=request.location
        )
        return PredictionResponse(**result)
    
    except ValueError as e:
        # Domain validation errors (unknown location, invalid values)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Get model metadata",
    description="Returns model configuration, supported locations, and feature list."
)
async def get_metadata(predictor: PredictorDep) -> MetadataResponse:
    """
    Get model metadata endpoint.
    
    Returns information about the loaded model including:
    - Numeric features
    - Supported locations (excluding 'other' for UI)
    - Model version/timestamp
    - Feature count
    """
    try:
        # Load metadata from model_metadata.json
        import json
        from pathlib import Path
        
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        preferred = repo_root / "artifacts" / "model_metadata.json"
        metadata_path = preferred if preferred.exists() else (repo_root / "model_metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Get supported locations from predictor (excludes 'other')
        all_locations = predictor.get_supported_locations()
        
        # Filter out 'other' for UI display
        supported_locations = [loc for loc in all_locations if loc.lower() != 'other']
        
        return MetadataResponse(
            numeric_features=metadata["numeric_features"],
            supported_locations=supported_locations,
            model_version=metadata["trained_at_utc"],
            n_features=metadata["n_features"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metadata: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and model is loaded."
)
async def health_check(predictor: PredictorDep) -> dict:
    """
    Health check endpoint.
    
    Returns 200 OK if service is ready.
    """
    return {
        "status": "healthy",
        "service": "tunisia-house-price-predictor",
        "model_loaded": True
    }
