from fastapi import FastAPI
from app.routes import predict, healthcheck  # Import des routes depuis le dossier routes

# Création de l'instance FastAPI
app = FastAPI(
    title="My ML API",
    description="API pour servir des modèles de machine learning avec FastAPI",
    version="1.0.0"
)

# Enregistrement des routes
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(healthcheck.router, prefix="/healthcheck", tags=["Healthcheck"])

# Root endpoint (optionnel)
@app.get("/")
async def root():
    return {"message": "Welcome to the ML API"}
