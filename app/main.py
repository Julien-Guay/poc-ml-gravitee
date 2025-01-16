from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle ML pré-entraîné
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Créer l'application FastAPI
app = FastAPI()

# Définir le format des requêtes en utilisant Pydantic
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Endpoint pour vérifier le statut de l'API
@app.get("/")
def read_root():
    return {"message": "L'API est en ligne et fonctionne"}

# Endpoint pour effectuer une prédiction
@app.post("/predict")
def predict(request: PredictionRequest):
    # Préparer les données d'entrée pour le modèle
    input_data = np.array([[request.feature1, request.feature2, request.feature3]])
    
    # Faire une prédiction
    prediction = model.predict(input_data)
    
    # Retourner la prédiction sous forme JSON
    return {"prediction": prediction[0]}
