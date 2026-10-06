import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "processed" / "telco_engineered.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"


df = pd.read_csv(INPUT_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Mejor modelo
best_model = joblib.load(MODEL_PATH)

print("Modelo cargado correctamente.")



#Evaluación con threshold 0.5

y_proba_test = best_model.predict_proba(X_test)[:, 1]

y_pred_test = (y_proba_test >= 0.5).astype(int)

print("\n" + "=" * 50)
print("EVALUACIÓN CON THRESHOLD 0.5")
print("=" * 50)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_test):.4f}")

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))


#Mejor threshold

print("\n" + "=" * 50)
print("BUSCANDO THRESHOLD ÓPTIMO")
print("=" * 50)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

y_proba_oof = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


precisions, recalls, thresholds = precision_recall_curve(
    y_train,
    y_proba_oof
)

f1_scores = np.divide(
    2 * (precisions[:-1] * recalls[:-1]),
    (precisions[:-1] + recalls[:-1]),
    out=np.zeros_like(precisions[:-1]),
    where=(precisions[:-1] + recalls[:-1]) != 0
)

optimal_idx = np.argmax(f1_scores)

optimal_threshold = thresholds[optimal_idx]
best_cv_f1 = f1_scores[optimal_idx]

print(f"\nThreshold óptimo: {optimal_threshold:.4f}")
print(f"F1 obtenido durante OOF: {best_cv_f1:.4f}")


#Evaluar con el thershold ganador

y_pred_optimal = (
    y_proba_test >= optimal_threshold
).astype(int)

print("\n" + "=" * 50)
print("EVALUACIÓN FINAL CON THRESHOLD OPTIMIZADO")
print("=" * 50)

print(f"\nThreshold utilizado: {optimal_threshold:.4f}")

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_optimal):.4f}")

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_optimal))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal))


print("\n" + "=" * 50)
print("COMPARACIÓN DE THRESHOLDS")
print("=" * 50)

thresholds_to_test = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

resultados_threshold = []

for threshold in thresholds_to_test:

    y_pred = (
        y_proba_test >= threshold
    ).astype(int)

    resultados_threshold.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    })

df_thresholds = pd.DataFrame(resultados_threshold)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(optimal_threshold, THRESHOLD_PATH)

print("\n")
print(df_thresholds.to_string(index=False))