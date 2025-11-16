import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import ssl
import traceback  # <-- ADDED THIS IMPORT

# --- FIX for SSL Certificate Error ---
# This bypasses the certificate check for downloading datasets
ssl._create_default_https_context = ssl._create_unverified_context
# -------------------------------------------------------------------

# Import your base trees
from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

# Import BOTH wrappers
from base.pruning_wrapper import PruningWrapper
from base.bagging_wrapper import BaggingWrapper

from constants import datasets

# -------------------------------------------------------------------------
# Define all model types
# -------------------------------------------------------------------------

n_estimators = 50  # Number of trees for bagging ensembles

# 1. Base (unpruned) models
models_base = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]

# 2. Pruned models (Simple Pruning)
models_pruned = [
    PruningWrapper(base_estimator=DT_Entropy()),
    PruningWrapper(base_estimator=DT_Gini()),
    PruningWrapper(base_estimator=DT_GainRatio()),
    PruningWrapper(base_estimator=DT_ChiSquare()),
    PruningWrapper(base_estimator=DT_Hellinger()),
    PruningWrapper(base_estimator=DT_Twoing())
]

# 3. Bagged models (Bagging of unpruned trees)
models_bagged = [
    BaggingWrapper(base_estimator=DT_Entropy(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Gini(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_GainRatio(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_ChiSquare(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Hellinger(), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=DT_Twoing(), n_estimators=n_estimators)
]

# 4. Hybrid models (Bagging of PRUNED trees)
models_hybrid = [
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_Entropy()), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_Gini()), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_GainRatio()), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_ChiSquare()), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_Hellinger()), n_estimators=n_estimators),
    BaggingWrapper(base_estimator=PruningWrapper(base_estimator=DT_Twoing()), n_estimators=n_estimators)
]


# -------------------------------------------------------------------------
# Helper: load and encode categorical data
# (Using the improved version from pruning that handles '?' as NA)
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

    # Zip and loop through all FOUR model types
    for m_base, m_pruned, m_bagged, m_hybrid in zip(models_base, models_pruned, models_bagged, models_hybrid):
        
        print(f"--- Testing {m_base.name} ---")

        # --- 1. Base Model ---
        try:
            m_base.fit(X_train, y_train)
            preds_base = m_base.predict(X_test)
            acc_base = np.mean(preds_base == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_base.name, "Accuracy": acc_base})
            print(f"{m_base.name:30s} | Accuracy: {acc_base:.4f}")
        except Exception as e:
            print(f"{m_base.name:30s} | FAILED ({e})")
            traceback.print_exc() # Print full error stack

        # --- 2. Pruned Model ---
        try:
            m_pruned.fit(X_train, y_train)
            preds_pruned = m_pruned.predict(X_test)
            acc_pruned = np.mean(preds_pruned == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_pruned.name, "Accuracy": acc_pruned})
            print(f"{m_pruned.name:30s} | Accuracy: {acc_pruned:.4f}")
        except Exception as e:
            print(f"{m_pruned.name:30s} | FAILED ({e})")
            traceback.print_exc() # Print full error stack

        # --- 3. Bagged Model ---
        try:
            m_bagged.fit(X_train, y_train)
            preds_bagged = m_bagged.predict(X_test)
            acc_bagged = np.mean(preds_bagged == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_bagged.name, "Accuracy": acc_bagged})
            print(f"{m_bagged.name:30s} | Accuracy: {acc_bagged:.4f}")
        except Exception as e:
            print(f"{m_bagged.name:30s} | FAILED ({e})")
            traceback.print_exc() # Print full error stack

        # --- 4. Hybrid Model ---
        try:
            m_hybrid.fit(X_train, y_train)
            preds_hybrid = m_hybrid.predict(X_test)
            acc_hybrid = np.mean(preds_hybrid == y_test)
            results.append({"Dataset": d["name"], "Criterion": m_hybrid.name, "Accuracy": acc_hybrid})
            print(f"{m_hybrid.name:30s} | Accuracy: {acc_hybrid:.4f}")
        except Exception as e:
            print(f"{m_hybrid.name:30s} | FAILED ({e})")
            traceback.print_exc() # Print full error stack

# -------------------------------------------------------------------------
# Summarize results across datasets
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n=== Summary of All Model Results ===")
print(df_results)

summary = df_results.groupby("Criterion")["Accuracy"].mean().sort_values(ascending=False)
print("\n=== Average Accuracy (All Models) Across All Datasets ===")
print(summary)

# Create 'results' directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Save results to a new file
df_results.to_csv("results/all_models_10_datasets.csv", index=False)
print("\nDetailed results saved to results/all_models_10_datasets.csv")
