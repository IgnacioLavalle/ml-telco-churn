import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "telco_engineered.csv"

df = pd.read_csv(INPUT_PATH)

df["meanMonthlyCharge"] = np.where(
    df["tenure"] == 0, #condicion
    df["MonthlyCharges"], #if true
    df["TotalCharges"] / df["tenure"] #else
)


df = df.drop(columns=["MultipleLines","gender", "PhoneService"])


df.to_csv(OUTPUT_PATH, index=False)
print("Feature Engineering completado. Archivo 'telco_engineered.csv' guardado.")