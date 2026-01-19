"""
FastAPI application entry point.
Handles lifespan events (model loading), middleware, and route registration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging

from .adapters.inference.safetensors_model import SafetensorsLinearModel
from .domain.vectorizer import FeatureVectorizer
from .domain.predictor import PricePredictorService
from .adapters.api import routes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global service instance (loaded at startup)
predictor_service: PricePredictorService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads model artifacts once at startup, cleans up on shutdown.
    """
    global predictor_service
    
    logger.info("🚀 Starting Tunisia House Price Predictor API...")
    
    # Define paths to artifacts
    base_path = Path(__file__).parent.parent
    model_path = base_path / "tunisia_home_prices_model.safetensors"
    columns_path = base_path / "columns.json"
    
    try:
        # Load columns.json (feature contract)
        logger.info(f"📋 Loading feature contract from {columns_path}")
        with open(columns_path, "r", encoding="utf-8") as f:
            columns_data = json.load(f)
        
        feature_columns = columns_data["data_columns"]
        numeric_features = columns_data["numeric_features"]
        
        logger.info(f"✅ Loaded {len(feature_columns)} features ({len(numeric_features)} numeric)")
        
        # Initialize vectorizer
        vectorizer = FeatureVectorizer(
            feature_columns=feature_columns,
            numeric_features=numeric_features
        )
        
        # Load model from Safetensors
        logger.info(f"🧠 Loading model from {model_path}")
        model = SafetensorsLinearModel()
        model.load(model_path)
        
        logger.info(f"✅ Model loaded: {model.get_feature_count()} features")
        
        # Initialize predictor service
        predictor_service = PricePredictorService(
            model=model,
            vectorizer=vectorizer
        )
        
        logger.info("✨ API ready to serve predictions!")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    finally:
        # Cleanup (if needed)
        logger.info("🛑 Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Tunisia House Price Predictor API",
    description="Predict house prices in Tunisia using Linear Regression with Safetensors",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware (for development - tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite/React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(routes.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API information."""
    return {
        "service": "Tunisia House Price Predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
