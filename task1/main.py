import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing


# -------------------------------------------------------------------------
# Define 10 categorical datasets from UCI Repository
# -------------------------------------------------------------------------
datasets = [
    {
        "name": "Car Evaluation",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "cols": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    },
    {
        "name": "Mushroom",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        "cols": [
            "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
            "gill-attachment", "gill-spacing", "gill-size", "gill-color",
            "stalk-shape", "stalk-root", "stalk-surface-above-ring",
            "stalk-surface-below-ring", "stalk-color-above-ring",
            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
            "ring-type", "spore-print-color", "population", "habitat"
        ]
    },
    {
        "name": "Nursery",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data",
        "cols": [
            "parents", "has_nurs", "form", "children", "housing",
            "finance", "social", "health", "class"
        ]
    },
    {
        "name": "Tic-Tac-Toe Endgame",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data",
        "cols": [
            "top-left", "top-middle", "top-right", "middle-left", "middle-middle",
            "middle-right", "bottom-left", "bottom-middle", "bottom-right", "class"
        ]
    },
    {
        "name": "Credit Approval",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
        "cols": [
            "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
            "A10", "A11", "A12", "A13", "A14", "A15", "class"
        ]
    },
    {
        "name": "Balance Scale",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
        "cols": ["class", "left-weight", "left-distance", "right-weight", "right-distance"]
    },
    {
        "name": "Hayes-Roth",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/hayes-roth/hayes-roth.data",
        "cols": ["id", "hobby", "age", "education", "marital-status", "class"]
    },
    {
        "name": "Congressional Voting Records",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
        "cols": [
            "class", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
            "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
            "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
            "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"
        ]
    },
    {
        "name": "Monks-1",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
        "cols": ["a1", "a2", "a3", "a4", "a5", "a6", "class"]
    },
    {
        "name": "Chess (King-Rook vs King-Pawn)",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data",
        "cols": [
            "white_king_file", "white_king_rank", "white_rook_file", "white_rook_rank",
            "black_king_file", "black_king_rank", "black_pawn_file", "black_pawn_rank", "class"
        ]
    }
]


# -------------------------------------------------------------------------
# Define all splitting-criterion models
# -------------------------------------------------------------------------
models = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]


# -------------------------------------------------------------------------
# Helper: load and encode categorical data
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

    for model in models:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = np.mean(preds == y_test)
            results.append({"Dataset": d["name"], "Criterion": model.name, "Accuracy": acc})
            print(f"{model.name:20s} | Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"{model.name:20s} | Failed ({e})")


# -------------------------------------------------------------------------
# Summarize results across datasets
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n=== Summary of Results ===")
print(df_results)

summary = df_results.groupby("Criterion")["Accuracy"].mean().sort_values(ascending=False)
print("\n=== Average Accuracy Across All 10 Datasets ===")
print(summary)

# Save results
df_results.to_csv("results_10_datasets.csv", index=False)
print("\nDetailed results saved to results_10_datasets.csv")
