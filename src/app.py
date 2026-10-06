import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

#Configuración de Paths y Carga
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final_model.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"

try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando modelos. Verifica las rutas: {e}")

#Definición de la API
app = FastAPI(title="Telco Churn Prediction API", version="1.0")

class PredictionResponse(BaseModel):
    customerID: str
    Churn_Probability: float
    Churn_Prediction: str

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el feature engineering exacto del entrenamiento."""
    X = df.copy()
    
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        
    if all(c in X.columns for c in ['TotalCharges', 'tenure', 'MonthlyCharges']):
        mask = X['TotalCharges'].isna() & (X['tenure'] == 0)
        X.loc[mask, "TotalCharges"] = X.loc[mask, "MonthlyCharges"]
        
        X['meanMonthlyCharge'] = np.where(
            X['tenure'] > 0, 
            X['TotalCharges'] / X['tenure'], 
            X['MonthlyCharges']
        )
        
    #Eliminar columnas que el modelo no usa
    cols_to_drop = ["customerID", "MultipleLines", "gender", "PhoneService"]
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    #Limpiar NaNs restantes (si los hay) para no romper el predict
    X = X.dropna()
    
    return X

@app.post("/predict", response_model=List[PredictionResponse])
def predict_churn(payload: List[Dict[str, Any]]):
    try:
        # 1. Convertir JSON a DataFrame
        df_raw = pd.DataFrame(payload)
        
        # 2. Guardar IDs para la respuesta
        customer_ids = df_raw.get("customerID", pd.Series([f"Unknown_{i}" for i in range(len(df_raw))]))
        
        # 3. Preprocesar
        X_processed = preprocess_data(df_raw)
        
        if X_processed.empty:
             raise HTTPException(status_code=400, detail="Los datos enviados quedaron vacíos tras la limpieza.")
        
        # 4. Predecir
        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # 5. Formatear respuesta
        resultados = []
        for cid, prob, pred in zip(customer_ids, probabilities, predictions):
            resultados.append(
                PredictionResponse(
                    customerID=str(cid),
                    Churn_Probability=round(float(prob), 4),
                    Churn_Prediction="Yes" if pred == 1 else "No"
                )
            )
            
        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "threshold_active": threshold}