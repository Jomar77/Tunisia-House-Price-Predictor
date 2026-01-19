# Tunisia House Price Predictor - Setup Guide

This guide will help you get the Tunisia House Price Predictor running locally.

## 🎯 System Requirements

- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher
- **npm**: 9.x or higher
- **Operating System**: Windows, macOS, or Linux

## 📥 Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd tunisia-house-price-predictor
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Verify Model Artifacts

Ensure these files exist in the root directory:
- `tunisia_home_prices_model.safetensors` - Model weights
- `columns.json` - Feature schema
- `model_metadata.json` - Model metadata

If missing, run the training notebook `main.ipynb` to generate them.

#### Start the Backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Frontend Setup

#### Navigate to Frontend Directory

```bash
cd frontend
```

#### Install Node Dependencies

```bash
npm install
```

#### Configure Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

#### Start the Frontend

```bash
npm run dev
```

The frontend will be available at: **http://localhost:5173**

## 🧪 Testing the Application

### 1. Test Backend API

Open your browser and go to: http://localhost:8000/docs

Try the `/api/v1/predict` endpoint with this sample data:

```json
{
  "area": 150,
  "rooms": 3,
  "bathrooms": 2,
  "age": 5,
  "location": "Hammamet"
}
```

### 2. Test Frontend

1. Open http://localhost:5173 in your browser
2. Fill in the prediction form:
   - Area: 150 m²
   - Rooms: 3
   - Bathrooms: 2
   - Age: 5 years
   - Location: Select from dropdown (e.g., "Hammamet")
3. Click "Predict Price"
4. You should see a predicted price in EUR

## 🔧 Troubleshooting

### Backend Issues

**Error: `No module named uvicorn`**
```bash
pip install uvicorn[standard]
```

**Error: `ImportError: attempted relative import with no known parent package`**
- Make sure you run the backend from the **project root**, not from the `backend/` directory
- Correct: `python -m uvicorn backend.main:app --reload`
- Wrong: `cd backend && python -m uvicorn main:app --reload`

**Error: `FileNotFoundError: columns.json`**
- Run the training notebook `main.ipynb` to generate model artifacts

### Frontend Issues

**Error: `Cannot find module '@tanstack/react-query'`**
```bash
cd frontend
npm install
```

**Error: `Failed to fetch metadata`**
- Ensure the backend is running on http://localhost:8000
- Check the `.env` file has the correct `VITE_API_URL`
- Check CORS settings in `backend/main.py`

**Error: Network request failed**
- Verify backend is running: http://localhost:8000/api/v1/health
- Check console for CORS errors
- Ensure `.env` file exists with correct API URL

## 🚀 Production Deployment

### Build Frontend

```bash
cd frontend
npm run build
```

The production build will be in `frontend/dist/`

### Run Backend in Production

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

For production, consider:
- Using a process manager like `systemd` or `supervisor`
- Setting up NGINX as a reverse proxy
- Using environment variables for configuration
- Enabling HTTPS with SSL certificates

## 📚 Next Steps

- Explore the API documentation at http://localhost:8000/docs
- Review the code architecture in the main README.md
- Check the model training process in `main.ipynb`
- Customize the frontend styling in `frontend/src/App.css`

## 🆘 Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Review the logs in your terminal
3. Open an issue on GitHub with error details
