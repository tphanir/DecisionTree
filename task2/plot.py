import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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


base_models = [
    DT_Entropy(),
    DT_Gini(),
    DT_GainRatio(),
    DT_ChiSquare(),
    DT_Hellinger(),
    DT_Twoing()
]

# -------------------------------------------------------------------------
# Helper: load and encode categorical data (Same as Task 1)
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
# Run plotting experiment
# -------------------------------------------------------------------------

# Define the steps for n_estimators.
# We test 1-10 individually, then every 5 up to 100.
# This gives high resolution at the start, where change is most rapid.
N_ESTIMATORS_STEPS = list(range(1, 11)) + list(range(15, 101, 5))

# Create 'plots' directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

print(f"Starting plot generation... Plots will be saved in 'plots/' directory.")
print(f"Testing n_estimators at steps: {N_ESTIMATORS_STEPS}\n")

for d in datasets:
    print(f"--- Processing Dataset: {d['name']} ---")
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(d["url"], d["cols"])
    except Exception as e:
        print(f"Skipping {d['name']} due to error: {e}")
        continue

    for model in base_models:
        print(f"  Plotting for Criterion: {model.name}...")
        
        # 1. Get baseline (single tree) accuracy
        model.fit(X_train, y_train)
        base_preds = model.predict(X_test)
        base_acc = np.mean(base_preds == y_test)
        
        # 2. Get bagged accuracy over n_estimators range
        bagged_accuracies = []
        for n in N_ESTIMATORS_STEPS:
            try:
                wrapper = BaggingWrapper(base_estimator=model, n_estimators=n)
                wrapper.fit(X_train, y_train)
                bagged_preds = wrapper.predict(X_test)
                bagged_acc = np.mean(bagged_preds == y_test)
                bagged_accuracies.append(bagged_acc)
            except Exception as e:
                print(f"    Failed at n={n} with error: {e}")
                bagged_accuracies.append(np.nan) # Append NaN if it fails
        
        # 3. Create and save the plot
        plt.figure(figsize=(12, 7))
        
        # Plot the bagged accuracy
        plt.plot(N_ESTIMATORS_STEPS, bagged_accuracies, marker='o', linestyle='-', label=f"Bagged Accuracy ({model.name})")
        
        # Plot the single tree accuracy as a horizontal line
        plt.axhline(y=base_acc, color='r', linestyle='--', label=f"Single Tree Accuracy ({base_acc:.4f})")
        
        # Find and mark the maximum bagged accuracy
        max_bagged_acc = np.nanmax(bagged_accuracies)
        max_n = N_ESTIMATORS_STEPS[np.nanargmax(bagged_accuracies)]
        plt.axhline(y=max_bagged_acc, color='g', linestyle=':', label=f"Max Bagged Acc ({max_bagged_acc:.4f} at n={max_n})")
        
        # Style the plot
        plt.title(f"Bagging Performance vs. Number of Estimators\nDataset: {d['name']} | Criterion: {model.name}", fontsize=16)
        plt.xlabel("Number of Estimators (B)", fontsize=12)
        plt.ylabel("Test Set Accuracy", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Set Y-axis limits. Start just below the min accuracy and go just above 1.0
        min_acc = min(base_acc, np.nanmin(bagged_accuracies))
        plt.ylim(max(0, min_acc - 0.05), 1.05)
        
        # 4. Save the plot to the 'plots' directory
        plot_filename = f"plots/{d['name']}_{model.name}.png".replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(plot_filename, dpi=100)
        plt.close() # Close the figure to free up memory

print("\n--- Plot generation complete! ---")