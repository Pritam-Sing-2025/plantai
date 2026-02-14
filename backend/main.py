from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model_service import get_prediction, MODELS
from typing import Optional
app = FastAPI(title="Plant Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running"}
@app.get("/models")
async def list_models():
    return {"models": list(MODELS.keys()) + ["Ensemble"]}
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("Ensemble")
):
    contents = await file.read()
    result = get_prediction(contents, model_name, filename=file.filename)
    return result
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9101)
