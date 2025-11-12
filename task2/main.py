import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import your base trees from Task 1
from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

# Import the new BaggingWrapper
from base.bagging_wrapper import BaggingWrapper
from constants import datasets

# -------------------------------------------------------------------------
# Define all splitting-criterion models, wrapped in Bagging
# -------------------------------------------------------------------------
# We use the unpruned trees from Task 1 as base estimators
# We can set n_estimators to a smaller number (e.g., 25) for faster testing

models_base = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]

n_estimators = 50

models = [
    BaggingWrapper(base_estimator=DT_Entropy(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Gini(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_GainRatio(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_ChiSquare(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Hellinger(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Twoing(), n_estimators=n_estimators)
]


# -------------------------------------------------------------------------
# Helper: load and encode categorical data (Same as Task 1)
# -------------------------------------------------------------------------
def load_and_preprocess(url, cols):
    df = pd.read_csv(url, names=cols)
    le = LabelEncoder()
    for c in df.columns:
        df[c] = le.fit_transform(df[c].astype(str))
    X = df.drop("class", axis=1, errors="ignore").values
    y = df["class"].values
    return train_test_split(X, y, test_size=0.3, random_state=42)


# -------------------------------------------------------------------------
# Run evaluation for each dataset and model
# -------------------------------------------------------------------------
results = []

for d in datasets:
    print(f"\n=== Dataset: {d['name']} ===")
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(d["url"], d["cols"])
    except Exception as e:
        print(f"Skipping {d['name']} due to error: {e}")
        continue

    for m1, m2 in zip(models_base,models):
        try:
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)

            preds1 = m1.predict(X_test)
            preds2 = m2.predict(X_test)
            
            acc1 = np.mean(preds1 == y_test)
            acc2 = np.mean(preds2 == y_test)
            results.append({"Dataset": d["name"], "Criterion": m1.name, "Accuracy": acc1})
            results.append({"Dataset": d["name"], "Criterion": m2.name, "Accuracy": acc2})
            print(f"{m1.name:25s} | Accuracy: {acc1:.4f}")
            print(f"{m2.name:25s} | Accuracy: {acc2:.4f}")
        except Exception as e:
            print(f"{m1.name:25s} | Failed ({e})")
            print(f"{m2.name:25s} | Failed ({e})")

# -------------------------------------------------------------------------
# Summarize results across datasets
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n=== Summary of Bagging Results ===")
print(df_results)

summary = df_results.groupby("Criterion")["Accuracy"].mean().sort_values(ascending=False)
print("\n=== Average Accuracy (Bagged) Across All 10 Datasets ===")
print(summary)

# Save results to a new file
df_results.to_csv("results/bagging_10_datasets.csv", index=False)
print("\nDetailed bagging results saved to results/bagging_10_datasets.csv")