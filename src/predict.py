import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "final_model.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"

#Cargar modelo y threshold
try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
except FileNotFoundError as e:
    print(f"Error al cargar los modelos: {e}")
    print("Asegúrate de haber entrenado el modelo y guardado los archivos .pkl en la carpeta 'models'.")
    sys.exit(1)

#cargar datos
if len(sys.argv) < 2:
    print("Uso: python src/predict.py <ruta_al_archivo_crudo.csv>")
    sys.exit(1)

input_path = Path(sys.argv[1])

try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de entrada '{input_path}'")
    sys.exit(1)

#Dejo el csv tal cual hice con el data clean

X = df.copy()

# 1. Asegurar que TotalCharges sea numérico
if 'TotalCharges' in X.columns:
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# 2. Imputar TotalCharges para clientes nuevos (tenure = 0)
if 'TotalCharges' in X.columns and 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
    mask = X['TotalCharges'].isna() & (X['tenure'] == 0)
    X.loc[mask, "TotalCharges"] = X.loc[mask, "MonthlyCharges"]

# 3. Limpiar los NaNs restantes y sincronizar con el df original
if 'TotalCharges' in X.columns:
    filas_validas = X['TotalCharges'].notna()
    X = X[filas_validas].copy()
    df = df[filas_validas].copy()

# 4. Calcular meanMonthlyCharge
if 'TotalCharges' in X.columns and 'tenure' in X.columns:
    X['meanMonthlyCharge'] = np.where(
        X['tenure'] > 0, 
        X['TotalCharges'] / X['tenure'], 
        X['MonthlyCharges']  # Si tenure es 0, su cargo medio es su cargo mensual
    )

# 5. Eliminar customerID de las features, pero conservarlo en los resultados
if "customerID" in X.columns:
    X = X.drop(columns=["customerID"])

#Dropeo las columnas que elimine durante el feature engineering
X = X.drop(columns=["MultipleLines","gender", "PhoneService"])


#Predicciones
try:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
except ValueError as e:
    print(f"\nError al predecir: {e}")
    print("Asegúrate de que el CSV de entrada tenga todas las columnas requeridas por el modelo.")
    sys.exit(1)

#resultado
results = df.copy()

results["Churn_Probability"] = probabilities
results["Churn_Prediction"] = predictions

results["Churn_Prediction"] = results["Churn_Prediction"].map({
    0: "No",
    1: "Yes"
})

#Mostrar y guardar
print(f"\nThreshold utilizado: {threshold:.4f}")

columnas_mostrar = ["Churn_Probability", "Churn_Prediction"]
if "customerID" in results.columns:
    columnas_mostrar.insert(0, "customerID")

print("\nPrimeras 10 predicciones:")
print(results[columnas_mostrar].head(10))

#Guardar
output_path = input_path.parent / f"predictions_{input_path.name}"
results.to_csv(output_path, index=False)

print(f"\n¡Predicciones generadas y guardadas exitosamente en:")
print(f"-> {output_path}")