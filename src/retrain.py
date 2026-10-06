from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "telco_engineered.csv"
MODEL_PATH = ROOT / "models" / "final_model.pkl"
THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"
METRICS_PATH = ROOT / "models" / "retrain_metadata.json"


# ============================================================
# Configuración
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# Función para encontrar threshold óptimo
# ============================================================

def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_proba
    )

    # precision_recall_curve devuelve una precision/recall
    # adicional al final, por eso usamos [:-1]
    f1_scores = np.divide(
        2 * (precisions[:-1] * recalls[:-1]),
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(precisions[:-1]),
        where=(precisions[:-1] + recalls[:-1]) != 0
    )

    optimal_idx = np.argmax(f1_scores)

    return (
        thresholds[optimal_idx],
        f1_scores[optimal_idx]
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("RETRAIN DEL MODELO")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Cargar datos
    # --------------------------------------------------------

    print("\n[1/6] Cargando datos...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Churn" not in df.columns:
        raise ValueError(
            "El dataset no contiene la columna 'Churn'."
        )

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Filas: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn positivo: {y.mean():.2%}")


    # --------------------------------------------------------
    # 2. Cargar modelo actual
    # --------------------------------------------------------

    print("\n[2/6] Cargando modelo actual...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}\n"
            "Entrená primero el modelo con train_model.py."
        )

    current_model = joblib.load(MODEL_PATH)

    print(
        f"Modelo cargado: "
        f"{current_model.named_steps['classifier'].__class__.__name__}"
    )


    # --------------------------------------------------------
    # 3. OOF predictions para elegir threshold
    # --------------------------------------------------------

    print("\n[3/6] Calculando predicciones OOF...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Usamos clone para asegurarnos de que cada fold
    # entrene un modelo completamente independiente.
    oof_model = clone(current_model)

    y_proba_oof = cross_val_predict(
        oof_model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    optimal_threshold, oof_f1 = find_optimal_threshold(
        y,
        y_proba_oof
    )

    oof_pred = (
        y_proba_oof >= optimal_threshold
    ).astype(int)

    oof_f1_check = f1_score(
        y,
        oof_pred
    )

    oof_roc_auc = roc_auc_score(
        y,
        y_proba_oof
    )

    oof_pr_auc = average_precision_score(
        y,
        y_proba_oof
    )

    print(f"Threshold óptimo: {optimal_threshold:.4f}")
    print(f"OOF F1:           {oof_f1:.4f}")
    print(f"OOF ROC-AUC:      {oof_roc_auc:.4f}")
    print(f"OOF PR-AUC:       {oof_pr_auc:.4f}")

    # --------------------------------------------------------
    # 3.5. Evaluación Champion vs Challenger (Red de seguridad)
    # --------------------------------------------------------
    print("\n[3.5/6] Comparando modelo actual vs. modelo nuevo...")
    
    # Cargar el threshold viejo para evaluar el modelo viejo justamente
    old_threshold = joblib.load(THRESHOLD_PATH)
    
    # Predecir sobre los datos nuevos usando el modelo viejo intacto
    y_proba_old = current_model.predict_proba(X)[:, 1]
    old_pred = (y_proba_old >= old_threshold).astype(int)
    old_f1 = f1_score(y, old_pred)
    
    print(f"F1 Modelo Actual (Champion): {old_f1:.4f}")
    print(f"F1 Modelo Nuevo (Challenger): {oof_f1:.4f}")
    
    if oof_f1 < old_f1:
        print("\n ALERTA: El modelo nuevo es peor que el actual.")
        print("El reentrenamiento se abortará para proteger el modelo en producción.")
        return # Termina la ejecución del script sin guardar nada
    else:
        print("\n El modelo nuevo es igual o mejor. Procediendo al despliegue...")
    # --------------------------------------------------------
    # 4. Entrenar modelo final con toda la data
    # --------------------------------------------------------

    print("\n[4/6] Entrenando modelo final con toda la data...")

    final_model = clone(current_model)

    final_model.fit(
        X,
        y
    )

    print("Entrenamiento final completado.")


    # --------------------------------------------------------
    # 5. Guardar modelo + threshold
    # --------------------------------------------------------

    print("\n[5/6] Guardando artefactos...")

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    joblib.dump(
        optimal_threshold,
        THRESHOLD_PATH
    )

    print(f"Modelo guardado en:    {MODEL_PATH}")
    print(f"Threshold guardado en: {THRESHOLD_PATH}")


    # --------------------------------------------------------
    # 6. Guardar metadata
    # --------------------------------------------------------

    print("\n[6/6] Guardando metadata...")

    classifier = final_model.named_steps["classifier"]

    metadata = {
        "retrain_timestamp": datetime.now().isoformat(),
        "dataset": str(DATA_PATH),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "model_type": classifier.__class__.__name__,
        "threshold": float(optimal_threshold),
        "oof_f1": float(oof_f1),
        "oof_f1_check": float(oof_f1_check),
        "oof_roc_auc": float(oof_roc_auc),
        "oof_pr_auc": float(oof_pr_auc),
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print(f"Metadata guardada en:   {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("RETRAIN COMPLETADO")
    print("=" * 60)

    print(f"\nModelo:    {classifier.__class__.__name__}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"OOF F1:    {oof_f1:.4f}")


if __name__ == "__main__":
    main()