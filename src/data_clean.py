import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"

df = pd.read_csv(INPUT_PATH)

#Droppeo customerId, no sirve para predecir
df = df.drop(columns=["customerID"])

#Paso total charges a int, limpio los NaNs
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

#Hay muchos casos donde el totalCharge es NaN porque el tenure es 0 pero hay info en MonthlyCharge
mask = (
    df["TotalCharges"].isna() &
    (df["tenure"] == 0)
)

#En ese caso lo lleno con el valor de monthlyCharge
df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]

df = df.dropna(subset=["TotalCharges"])

#Paso churn a 0, 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df.to_csv(OUTPUT_PATH, index=False)