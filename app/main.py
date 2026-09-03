from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models/model.joblib"
app=FastAPI(title="MediScan Diagnostic AI",version="1.0.0")

class PredictRequest(BaseModel):
    features:list[float]=Field(min_length=30,max_length=30)

class PredictResponse(BaseModel):
    prediction:str
    probability:float
    model_version:str

def load_model():
    if not MODEL.exists(): raise HTTPException(503,"Model not trained. Run python ml/train.py")
    return joblib.load(MODEL)

@app.get("/health")
def health(): return {"status":"ok","model_ready":MODEL.exists()}

@app.get("/model-info")
def model_info():
    return {"model":"RandomForestClassifier","dataset":"sklearn breast cancer","features":30,"educational_use_only":True}

@app.post("/predict",response_model=PredictResponse)
def predict(payload:PredictRequest):
    bundle=load_model()
    model=bundle["model"]
    proba=float(model.predict_proba([payload.features])[0][1])
    # Dataset target: 0=malignant, 1=benign
    label="benign" if proba>=0.5 else "malignant"
    return {"prediction":label,"probability":round(proba,4),"model_version":bundle["version"]}
