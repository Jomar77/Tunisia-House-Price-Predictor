"""
Domain service for house price prediction.
Orchestrates vectorization and model inference.
"""
from typing import Protocol
import numpy as np


class ModelPort(Protocol):
    """Port (interface) for model inference."""
    
    def predict(self, X: np.ndarray) -> float:
        """Predict from feature vector."""
        ...


class PricePredictorService:
    """
    Core domain service for predicting house prices.
    Pure business logic - no framework dependencies.
    """
    
    def __init__(self, model: ModelPort, vectorizer):
        """
        Initialize predictor with model and vectorizer.
        
        Args:
            model: Model implementing ModelPort interface
            vectorizer: FeatureVectorizer instance
        """
        self.model = model
        self.vectorizer = vectorizer
    
    def predict_price(
        self,
        area: float,
        rooms: float,
        bathrooms: float,
        age: float,
        location: str
    ) -> dict:
        """
        Predict house price based on features.
        
        Args:
            area: House area in square meters
            rooms: Number of rooms
            bathrooms: Number of bathrooms
            age: Age of the house in years
            location: Location name
            
        Returns:
            Dictionary with prediction and metadata
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate numeric inputs
        if area <= 0:
            raise ValueError("Area must be positive")
        if rooms <= 0:
            raise ValueError("Rooms must be positive")
        if bathrooms <= 0:
            raise ValueError("Bathrooms must be positive")
        if age < 0:
            raise ValueError("Age cannot be negative")
        
        # Vectorize inputs
        feature_vector = self.vectorizer.vectorize(
            area=area,
            rooms=rooms,
            bathrooms=bathrooms,
            age=age,
            location=location
        )
        
        # Get prediction from model
        predicted_price = self.model.predict(feature_vector)
        
        return {
            "predicted_price": predicted_price,
            "inputs": {
                "area": area,
                "rooms": rooms,
                "bathrooms": bathrooms,
                "age": age,
                "location": location
            }
        }
    
    def get_supported_locations(self) -> list:
        """Get list of valid location names."""
        return self.vectorizer.get_supported_locations()
