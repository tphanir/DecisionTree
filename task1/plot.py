import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import ssl
import traceback

# --- PLOTTING IMPORTS ---
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------

# --- FIX for SSL Certificate Error ---
# This bypasses the certificate check for downloading datasets
# (This was not in your main.py but is often needed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python may not have this attr
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -------------------------------------------------------------------

# Import ONLY your base trees
from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

# Import datasets
from constants import datasets

# -------------------------------------------------------------------------
# Define all splitting-criterion models
# (Copied from your main.py)
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
# (Copied from your main.py)
# -------------------------------------------------------------------------
def load_and_preprocess(url, cols):
    # Note: This version doesn't handle '?' missing values.
    # If your datasets have '?', this may need to be updated to:
    # df = pd.read_csv(url, names=cols, na_values='?')
    # df = df.dropna()
    df = pd.read_csv(url, names=cols)
    le = LabelEncoder()
    for c in df.columns:
        df[c] = le.fit_transform(df[c].astype(str))
    X = df.drop("class", axis=1, errors="ignore").values
    y = df["class"].values
    return train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------------------------------------------------
# PLOTTING FUNCTION
# -------------------------------------------------------------------------
def plot_base_criteria_by_dataset(df, save_path):
    """
    Plots a grouped bar chart showing how each of the 6 base
    splitting criteria perform on each of the 10 datasets.
    """
    print("\n--- Generating Plot: Base Model Performance by Dataset ---")
    
    if df.empty:
        print("  WARNING: No data found to plot. Skipping plot.")
        return

    # Define a color palette for the 6 criteria
    criteria_order = sorted(df['Criterion'].unique())
    palette = sns.color_palette("deep", len(criteria_order))
    
    # Create the plot
    plt.figure(figsize=(20, 10)) # Make it wide to fit datasets
    sns.barplot(
        data=df,
        x='Dataset',
        y='Accuracy',
        hue='Criterion', # We can use 'Criterion' directly
        hue_order=criteria_order,
        palette=palette
    )
    
    plt.title('Base Model Performance: Splitting Criteria vs. Dataset', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Place legend outside the plot
    plt.legend(title='Splitting Criterion', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=14)
    
    # Adjust ylim to make differences clearer
    min_acc = df['Accuracy'].min() - 0.1
    plt.ylim(max(0, min_acc), 1.0) 
    
    plt.tight_layout() # Adjusts plot to prevent labels from being cut off
    plt.savefig(save_path, bbox_inches='tight') # bbox_inches ensures legend is saved
    print(f"Saved plot to: {save_path}")
    plt.close()

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # --- PART 1: Run the Experiment ---
    # (Logic from your main.py)
    
    results = [] # To store dicts of {"Dataset": ..., "Criterion": ..., "Accuracy": ...}

    for d in datasets:
        print(f"\n=== Dataset: {d['name']} ===")
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess(d["url"], d["cols"])
            if len(y_train) < 20: # Add a check for tiny datasets
                print(f"Skipping {d['name']} (dataset too small)")
                continue
        except Exception as e:
            print(f"Skipping {d['name']} due to error: {e}")
            continue

        # Loop through only the base models
        for model in models:
            
            # --- Base Model ---
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = np.mean(preds == y_test)
                # Store the result
                results.append({"Dataset": d["name"], "Criterion": model.name, "Accuracy": acc})
                print(f"{model.name:30s} | Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"{model.name:30s} | FAILED ({e})")
                traceback.print_exc()


    # --- PART 2: Process Results In-Memory ---
    
    print("\n=== Experiment complete. Processing results for plotting... ===")
    
    df_results = pd.DataFrame(results)

    # --- PART 3: Generate the Plot ---
    
    # Set a nice theme for all plots
    sns.set_theme(style="whitegrid")
    
    # Create 'plots' directory if it doesn't exist
    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True) # exist_ok=True prevents error if it already exists

    # Call the plotting function
    plot_save_path = os.path.join(OUTPUT_DIR, "1_base_model_performance.png")
    plot_base_criteria_by_dataset(
        df_results, 
        plot_save_path
    )
    
    # --- (Optional) Save the data to CSV for your own records ---
    csv_path = os.path.join(OUTPUT_DIR, "base_models_10_datasets.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Full results also saved to {csv_path} for your records.")
    
    print("\nDone.")