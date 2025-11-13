import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Import your base trees from Task 1
from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

# Import the new PruningWrapper
from base.pruning_wrapper import PruningWrapper
from constants import datasets

# -------------------------------------------------------------------------
# Define all splitting-criterion models
# -------------------------------------------------------------------------

# 1. Base (unpruned) models from Task 1
models_base = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]

# 2. Pruned models (using the new wrapper)
models_pruned = [
    PruningWrapper(base_estimator=DT_Entropy()),
    PruningWrapper(base_estimator=DT_Gini()),
    PruningWrapper(base_estimator=DT_GainRatio()),
    PruningWrapper(base_estimator=DT_ChiSquare()),
    PruningWrapper(base_estimator=DT_Hellinger()),
    PruningWrapper(base_estimator=DT_Twoing())
]


# -------------------------------------------------------------------------
# Helper: load and encode categorical data
# (Using the improved version from plot.py that handles '?' as NA)
# -------------------------------------------------------------------------
def load_and_preprocess(url, cols):
    df = pd.read_csv(url, names=cols, na_values='?')
    df = df.dropna()  # Drop rows with missing values
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
        
        # Handle small datasets that might fail on validation split
        if len(y_train) < 20:
            print(f"Skipping {d['name']} (dataset too small for pruning split)")
            continue
            
    except Exception as e:
        print(f"Skipping {d['name']} due to error: {e}")
        continue

    # Zip and loop through base and pruned models
    for m_base, m_pruned in zip(models_base, models_pruned):
        try:
            # --- Base Model ---
            m_base.fit(X_train, y_train)
            preds_base = m_base.predict(X_test)
            acc_base = np.mean(preds_base == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_base.name, "Accuracy": acc_base})
            print(f"{m_base.name:25s} | Accuracy: {acc_base:.4f}")

            # --- Pruned Model ---
            m_pruned.fit(X_train, y_train)
            preds_pruned = m_pruned.predict(X_test)
            acc_pruned = np.mean(preds_pruned == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_pruned.name, "Accuracy": acc_pruned})
            print(f"{m_pruned.name:25s} | Accuracy: {acc_pruned:.4f}")

        except Exception as e:
            print(f"{m_base.name:25s} | Failed ({e})")
            print(f"{m_pruned.name:25s} | Failed ({e})")

# -------------------------------------------------------------------------
# Summarize results across datasets
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n=== Summary of Pruning Results ===")
print(df_results)

summary = df_results.groupby("Criterion")["Accuracy"].mean().sort_values(ascending=False)
print("\n=== Average Accuracy (Base vs. Pruned) Across All 10 Datasets ===")
print(summary)

# Create 'results' directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Save results to a new file
df_results.to_csv("results/pruning_10_datasets.csv", index=False)
print("\nDetailed pruning results saved to results/pruning_10_datasets.csv")