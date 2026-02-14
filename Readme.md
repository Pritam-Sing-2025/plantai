# PlantAi - Plant Disease Detection System

## Project Overview
PlantAi is a full-stack web application designed to detect plant diseases from images. It uses a robust hybrid machine learning approach, combining deep learning models (Ensemble of EfficientNetV2, ResNet50V2, MobileNetV3) with visual heuristics to provide accurate and reliable predictions.

## Architecture
The project is divided into two main components:

### 1. Backend (`/backend`)
Built with **FastAPI** and **TensorFlow**, the backend handles image processing, model inference, and training.

- API Service (`main.py`): Exposes REST endpoints for prediction and model management.
- ML Engine (`model_service.py`):
- Ensemble Logic: Combines predictions from multiple models for higher accuracy.
- Visual Heuristics: Analyzes image properties (color histograms, leaf health ratios) to cross-check ML outputs and prevent common errors.
- Determinism: Ensures consistent results for the same image inputs despite probabilistic elements.
- Training Pipeline (`train_models.py`):
    *   Automates downloading the **PlantVillage** dataset from Kaggle.
    *   Trains three state-of-the-art architectures (EfficientNetV2, ResNet50V2, MobileNetV3) using transfer learning.

### 2. Frontend (`/frontend`)
Built with **React.js**, and **TailwindCSS v4**, the frontend provides a modern, responsive user interface.

**Technology Stack**: React.
**Features**:
    *   Image upload and preview.
    *   Real-time disease prediction display.
    *   Detailed confidence breakdowns and treatment recommendations.

## Key Features
- Multi-Model Ensemble: Aggregates insights from different neural network architectures.
- Hybrid Analysis: Uses computer vision techniques (HSV analysis) alongside deep learning to detect "healthy" vs "diseased" states more reliably.
- Detailed Insights: Returns not just the disease name, but also treatment advice, accuracy scores, and visual confidence metrics.


# Backend
1.  Navigate to `backend/`.
2.  Activate virtual environment: `source venv/bin/activate`
3.  Run server: `python main.py`

# Frontend
1.  Navigate to `frontend/`.
2.  Install dependencies: `npm install`
3.  Run dev server: `npm run dev`
4.  Open `http://localhost:5173` in your browser.

# Deployement
- The project is deployed on Vercel
- You can checkout here: ``