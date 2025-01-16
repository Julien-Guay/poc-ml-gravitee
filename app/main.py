from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Loading the model
MODEL_PATH = "models/linear_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API OK"}

@app.get("/predict")
def predict(feature: float):
    input_data = np.array([[feature]])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}