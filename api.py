import glob
import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="IRIS Prediction API")

MODEL = None


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
def load_model():
    global MODEL

    mlmodels = glob.glob("mlruns/*/*/artifacts/model/MLmodel")

    if not mlmodels:
        raise RuntimeError("No MLflow model found.")

    model_path = os.path.dirname(mlmodels[-1])

    print(f"Loading model from {model_path}")

    MODEL = mlflow.pyfunc.load_model(model_path)


@app.get("/")
def home():
    return {"message": "IRIS API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisInput):

    df = pd.DataFrame([{
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
    }])

    prediction = MODEL.predict(df)

    return {
        "prediction": str(prediction[0])
    }