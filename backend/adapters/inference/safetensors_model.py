"""
Safetensors-based Linear Regression model loader and inference adapter.
Secondary adapter (driven) - implements model loading port.
"""
import numpy as np
from pathlib import Path
from safetensors.numpy import load_file
from typing import Optional


class SafetensorsLinearModel:
    """
    Linear Regression model loaded from Safetensors format.
    Provides zero-copy, secure model loading and inference.
    """
    
    def __init__(self):
        self.coef: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None
        self._loaded = False
    
    def load(self, model_path: Path) -> None:
        """
        Load model weights from Safetensors file.
        
        Args:
            model_path: Path to .safetensors file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load tensors from safetensors (zero-copy memory mapping)
        tensors = load_file(str(model_path))
        
        # Extract coefficients and intercept
        if "coef" not in tensors:
            raise ValueError("Missing 'coef' tensor in model file")
        if "intercept" not in tensors:
            raise ValueError("Missing 'intercept' tensor in model file")
        
        self.coef = tensors["coef"]
        self.intercept = tensors["intercept"]
        
        self._loaded = True
    
    def predict(self, X: np.ndarray) -> float:
        """
        Predict price using linear regression: y = X · w + b
        
        Args:
            X: Feature vector (1D numpy array)
            
        Returns:
            Predicted price
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input shape doesn't match model
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if X.shape[0] != self.coef.shape[0]:
            raise ValueError(
                f"Input feature count ({X.shape[0]}) doesn't match "
                f"model features ({self.coef.shape[0]})"
            )
        
        # Linear regression: y = w^T · x + b
        prediction = np.dot(X, self.coef) + self.intercept[0]
        
        return float(prediction)
    
    def get_feature_count(self) -> int:
        """Get expected number of input features."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return self.coef.shape[0]
