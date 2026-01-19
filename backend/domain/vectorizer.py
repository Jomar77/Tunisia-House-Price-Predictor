"""
Core domain logic for feature vectorization.
Pure Python - no FastAPI/framework imports.
"""
import numpy as np
from typing import Dict, List


class FeatureVectorizer:
    """Converts user inputs into model-ready feature vectors."""
    
    def __init__(self, feature_columns: List[str], numeric_features: List[str]):
        """
        Initialize vectorizer with feature contract.
        
        Args:
            feature_columns: Ordered list of all feature names (from columns.json)
            numeric_features: List of numeric feature names in order
        """
        self.feature_columns = feature_columns
        self.numeric_features = numeric_features
        self._validate_contract()
    
    def _validate_contract(self) -> None:
        """Ensure numeric features are present in feature columns."""
        for feat in self.numeric_features:
            if feat not in self.feature_columns:
                raise ValueError(
                    f"Numeric feature '{feat}' not found in feature_columns"
                )
    
    def vectorize(
        self,
        area: float,
        rooms: float,
        bathrooms: float,
        age: float,
        location: str
    ) -> np.ndarray:
        """
        Convert inputs to feature vector matching trained model ordering.
        
        Args:
            area: House area in square meters
            rooms: Number of rooms
            bathrooms: Number of bathrooms
            age: Age of the house in years
            location: Location name (must match training data exactly; unknowns may map to 'other' if available)
        
        Returns:
            Numpy array of features in correct order
            
        Raises:
            ValueError: If location is not in the supported list
        """
        # Initialize zero vector
        x_vec = np.zeros(len(self.feature_columns), dtype=np.float32)
        
        # Set numeric features
        feature_values = {
            "area": area,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "age": age
        }
        
        for feature_name in self.numeric_features:
            try:
                idx = self.feature_columns.index(feature_name)
                x_vec[idx] = feature_values[feature_name]
            except ValueError:
                raise ValueError(f"Feature '{feature_name}' not found in columns")
        
        # Set location (one-hot encoding)
        if location in self.feature_columns:
            location_column = location
        elif "other" in self.feature_columns:
            location_column = "other"
        else:
            raise ValueError(
                f"Unknown location: '{location}'. "
                "No 'other' fallback is present in feature_columns. "
                "Must be one of the supported locations."
            )

        loc_idx = self.feature_columns.index(location_column)
        x_vec[loc_idx] = 1.0
        
        return x_vec
    
    def get_supported_locations(self) -> List[str]:
        """Get list of supported location names (excluding numeric features)."""
        return [
            col for col in self.feature_columns 
            if col not in self.numeric_features
        ]
