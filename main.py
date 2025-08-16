from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Iris Species Predictor API")

# Load the trained model and label encoder
model = joblib.load("model3.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # <-- Load encoder

# Define request format
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is running."}

# Prediction endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width,
                      features.petal_length, features.petal_width]])
    
    prediction = model.predict(data)[0]  # Numeric label
    decoded_prediction = label_encoder.inverse_transform([prediction])[0]  # Decode to species

    return {"predicted_species": decoded_prediction}
