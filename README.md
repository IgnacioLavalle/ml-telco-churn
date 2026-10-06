# Telco Churn Prediction - End-to-End ML Pipeline

Este proyecto implementa un pipeline completo de Machine Learning para predecir la tasa de abandono (Churn) de clientes en una empresa de telecomunicaciones. 

Abarca todo el ciclo de vida del dato: desde la limpieza y el feature engineering, pasando por el entrenamiento automatizado de modelos con balanceo de clases mediante smotenc, hasta el despliegue via API REST en un contenedor Docker y un flujo automatizado de reentrenamiento (MLOps).

## Tecnologías Principales
    -Desarrollo de Modelos: Scikit-Learn, modelos como LightGBM y XGBoost, Imbalanced-Learn (SMOTENC).
    -Ingeniería y Pipeline: Pandas, NumPy, Joblib, Pathlib.
    -Despliegue (Serving): FastAPI, Uvicorn, Docker.
    -MLOps: Validación cruzada estratificada (OOF), Optimización de Threshold, Champion vs. Challenger.

##  Estructura del Proyecto

ml-production-lab/
├── data/
│   ├── raw/                  # Datos originales (no modificados)
│   └── processed/            # Datos limpios listos para modelado (telco_engineered.csv)
├── notebooks/                # Notebooks explorativas
│   ├── eda                   # Exploratory Data Analysis
│   └── model_analysis        # Análisis del desempeño del modelo
├── models/                   # Pkls de los modelos generados dinámicamente
│   ├── final_model.pkl
│   ├── threshold.pkl
│   └── retrain_metadata.json
├── src/                      # Código principal
│   ├── data_prep.py          # Limpieza básica
│   ├── feature_eng.py        # Creación y selección de variables
│   ├── train.py              # Entrenamiento exhaustivo (GridSearch + SMOTE)
│   ├── retrain.py            # Script MLOps (Champion vs Challenger)
│   └── app.py                # API de FastAPI
├── Dockerfile                # Configuración del contenedor de producción
├── requirements.txt          # Dependencias y versiones del proyecto
└── README.md



# Instalación y uso local

    En bash:
    $ git clone <tu-repositorio>
    $ cd ml-production-lab
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

    Luego se puede entrenar el modelo via:
    $ python src/train.py

# Despliegue con Docker

    Construir la imagen:
    
    $ docker build -t churn-api:latest .

    Ejecutar el contenedor:

    $ docker run -d -p 8000:8000 --name churn-container churn-api:latest

    La API estará disponible en http://localhost:8000. La documentación está en http://localhost:8000/docs.

# Uso de la API 

    La API recibe peticiones POST con un array de perfiles de clientes y devuelve la probabilidad y la predicción de Churn.

    Ejemplo de Payload (JSON):

    [
        {
            "customerID": "8800-TEST",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5,
            "TotalCharges": "141.0"
        }
    ]

    Y devuelve algo así como:

    [
        {
            "customerID": "8800-TEST",
            "Churn_Probability": 0.7850,
            "Churn_Prediction": "Yes"
        }
    ]

# MLOps y Reentrenamiento

    Para mitigar el Data Drift, el proyecto incluye un script robusto de reentrenamiento (src/retrain.py):
    Al inyectar nuevos datos etiquetados en data/processed/, el script:

    - Evalúa el threshold óptimo de clasificación.

    - Entrena un modelo candidato (Challenger) con el 100% de la data.

    - Evalúa al modelo actual en producción (Champion) frente a la nueva data.

    - Red de seguridad: Solo sobrescribe los .pkl en la carpeta models/ si el modelo nuevo supera o iguala el rendimiento (F1-Score) del modelo actual.

    - Registra las métricas resultantes en retrain_metadata.json.