# Tunisia House Price Predictor

A full-stack web application for predicting house prices in Tunisia using machine learning. The system uses a Linear Regression model trained on Tunisian real estate data and exposes predictions through a modern React frontend connected to a FastAPI backend.

## 🏗️ Architecture

This project follows a **Hexagonal Architecture** (Ports & Adapters) pattern:

- **Frontend**: React + TypeScript + Vite + TanStack Query
- **Backend**: FastAPI (async) with hexagonal architecture
- **Model**: Linear Regression with Safetensors (no pickle)
- **Deployment Ready**: Containerizable, serverless-compatible

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server (from project root)
python -m uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## 📁 Project Structure

```
├── backend/                   # FastAPI backend
│   ├── domain/               # Core business logic
│   │   ├── predictor.py      # Prediction service
│   │   └── vectorizer.py     # Feature vectorization
│   ├── adapters/             # External interfaces
│   │   ├── api/              # FastAPI routes & schemas
│   │   └── inference/        # Model loading (Safetensors)
│   └── main.py               # Application entry point
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── api/              # API client (native fetch)
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom hooks (TanStack Query)
│   │   └── App.tsx           # Main app component
│   └── package.json
├── columns.json              # Feature schema (source of truth)
├── model_metadata.json       # Model training metadata
├── tunisia_home_prices_model.safetensors  # Model weights
├── data.csv                  # Training dataset
├── dataScrape.py            # Data scraper
└── main.ipynb               # Training notebook
```

## 🔧 Features

### Frontend
- ✨ Clean, responsive UI with real-time validation
- 📍 Dynamic location dropdown loaded from API
- 🔄 Loading states and error handling
- 💰 Formatted price predictions in EUR

### Backend
- ⚡ Fast async API with automatic documentation
- 🛡️ Input validation with Pydantic
- 🎯 Hexagonal architecture for maintainability
- 🔒 Secure model loading with Safetensors (no pickle)
- 📊 Model metadata endpoint for UI configuration

### API Endpoints

- `POST /api/v1/predict` - Predict house price
- `GET /api/v1/metadata` - Get model info and supported locations
- `GET /api/v1/health` - Health check

## 📊 Dataset

The model is trained on Tunisian house price data including:
- **Numeric Features**: Area (m²), Rooms, Bathrooms, Age (years)
- **Categorical Features**: Location (60+ Tunisian locations)
- **Target**: House price in EUR

## 🧠 Model Training

The model is trained using the Jupyter notebook `main.ipynb`:

1. Data cleaning and outlier removal
2. Feature engineering (one-hot encoding for locations)
3. Linear Regression training
4. Export to Safetensors format
5. Generate `columns.json` and `model_metadata.json`

## 🔐 Security & Best Practices

- ✅ No pickle files (uses Safetensors)
- ✅ Input validation on all endpoints
- ✅ CORS configured for development
- ✅ Type-safe with TypeScript and Pydantic
- ✅ Native fetch API (no external HTTP libraries in frontend)

## 🛠️ Development

### Running Tests

```bash
# Backend (if tests exist)
pytest

# Frontend (if tests exist)
cd frontend && npm test
```

### Building for Production

```bash
# Frontend
cd frontend
npm run build

# Backend - use production ASGI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 📝 Environment Variables

### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000/api/v1
```

## 🎯 Future Enhancements

- [ ] PostgreSQL integration for prediction logging
- [ ] User feedback collection endpoint
- [ ] Docker containerization
- [ ] Cloud deployment (Google Cloud Run)
- [ ] Model retraining pipeline
- [ ] Additional ML models (Random Forest, XGBoost)

## 📄 License

This repository is licensed under the MIT License. See the LICENSE file for more information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or create an issue.
