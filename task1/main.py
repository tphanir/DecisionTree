import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(url, names=cols)
le = LabelEncoder()
for c in df.columns:
    df[c] = le.fit_transform(df[c])

X = df.drop("class", axis=1).values
y = df["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]

# Evaluate
for model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"{model.name:20s} | Accuracy: {acc:.4f}")
