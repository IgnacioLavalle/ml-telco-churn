import time
import warnings
import pandas as pd
import joblib
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report

)

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "telco_engineered.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_model.pkl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

binary_features = ["SeniorCitizen"]
numeric_features = ["tenure", "meanMonthlyCharge", "MonthlyCharges", "TotalCharges"]

categorical_features = [
    col for col in X_train.columns 
    if col not in binary_features + numeric_features
]

#Agrupar todas las características categóricas (SMOTENC trata binarias y multi-clase igual)
all_categorical = binary_features + categorical_features

#FASE 1 (PRE-SMOTE): Imputar todo y escalar SOLO numéricas. NO hacer One-Hot Encode todavía.
pre_smote_processor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]), numeric_features),
        
        ("cat", SimpleImputer(strategy="most_frequent"), all_categorical)
    ]
)

# Calcular los índices de las columnas categóricas después del ColumnTransformer pre_smote.
# ColumnTransformer agrupará primero las numéricas (índices 0 a N-1) y luego las categóricas.
num_count = len(numeric_features)
cat_count = len(all_categorical)
cat_indices = list(range(num_count, num_count + cat_count))

# 3. FASE 2 (POST-SMOTE): Hacer One-Hot Encoding a las columnas categóricas (que ahora están imputadas y balanceadas).
post_smote_processor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_indices)
    ],
    remainder="passthrough" # Deja pasar las numéricas ya escaladas sin tocarlas
)

# Diccionario de modelos e hiperparametros
# NOTA: Se eliminaron class_weight y scale_pos_weight porque SMOTENC ya balancea las clases 1:1
modelos = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "classifier__C": [0.1, 1, 10]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10]
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1]
        }
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1),
        {
            "classifier__n_estimators": [50, 100, 150], 
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4], 
            "classifier__subsample": [0.8, 1.0] 
        }
    ),
    "SVC": (
        SVC(random_state=42, probability=True),
        [
            {
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 1, 10, 100]
            },
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": ["scale", 0.01, 0.1, 1]
            }
        ]
    )
}

# Entrenamiento de modelos
best_overall_model = None
best_overall_f1 = 0
best_model_name = ""
resultados = []

print("\nIniciando búsqueda de hiperparámetros...")

pbar = tqdm(modelos.items(), desc="Preparando...", leave=True)

for nombre, (modelo, param_grid) in pbar:
    pbar.set_description(f"Entrenando: {nombre}")
    
    start_time = time.time()
    
    pipeline = Pipeline([
        ("pre_smote", pre_smote_processor),
        ("smotenc", SMOTENC(
            sampling_strategy=0.7,
            categorical_features=cat_indices,
            random_state=42
        )),
        ("post_smote", post_smote_processor),
        ("classifier", modelo)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        error_score="raise"
    )
    
    grid_search.fit(X_train, y_train)
    
    tiempo_ejecucion = round(time.time() - start_time, 2)
    cv_f1 = grid_search.best_score_
    mejor_modelo_local = grid_search.best_estimator_
    
    tqdm.write(f"✔ {nombre.ljust(18)} | F1: {cv_f1:.4f} | Tiempo: {tiempo_ejecucion}s")
    MODEL_PATH = MODELS_DIR / f"best_{nombre}.pkl"

    joblib.dump(mejor_modelo_local, MODEL_PATH)
    
    resultados.append({
        "Modelo": nombre,
        "F1_CV": grid_search.best_score_,
        "Std_F1_CV": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
        "Tiempo_Segundos": tiempo_ejecucion,
        "Mejores_Parametros": grid_search.best_params_
    })
    
    if cv_f1 > best_overall_f1:
        best_overall_f1 = cv_f1
        best_overall_model = mejor_modelo_local
        best_model_name = nombre

pbar.set_description("¡Completado!")

MODEL_PATH = MODELS_DIR / "final_model.pkl"
joblib.dump(best_overall_model,MODEL_PATH)

# Resumen final
print("\n" + "*"*50)
print(f"El algoritmo ganador es: {best_model_name} (F1 CV: {best_overall_f1:.4f})")
print("*"*50)

df_resultados = pd.DataFrame(resultados).sort_values(by="F1_CV", ascending=False)
print("\nTabla de Posiciones Final:")
print(df_resultados.to_string(index=False))

# Evaluación final en test
print(f"\nGenerando métricas sobre el Test Set con {best_model_name}...")
y_pred = best_overall_model.predict(X_test)
y_proba = best_overall_model.predict_proba(X_test)


if y_proba.ndim == 2:
    y_proba = y_proba[:, 1]
    
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

try:
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
except Exception as e:
    print("ROC-AUC no disponible para este modelo de manera predeterminada sin probability=True (ej. SVC con kernel lineal).")

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
