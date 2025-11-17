import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import ssl
import os

# Disable SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------------------
# Import your actual model classes
# ------------------------------
from criteria.dt_entropy import DT_Entropy
from criteria.dt_gini import DT_Gini
from criteria.dt_gain_ratio import DT_GainRatio
from criteria.dt_chi_square import DT_ChiSquare
from criteria.dt_hellinger import DT_Hellinger
from criteria.dt_twoing import DT_Twoing

from base.pruning_wrapper import PruningWrapper
from base.bagging_wrapper import BaggingWrapper

from constants import datasets

# ------------------------------
# Logger
# ------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ------------------------------
# Normalize criterion names
# ------------------------------
def normalize_criterion(name):
    n = name.lower().replace("-", " ").replace("_", " ")

    if "entropy" in n: return "Entropy"
    if "gini" in n: return "Gini"
    if "gain" in n: return "Gain Ratio"
    if "chi" in n: return "Chi-Square"
    if "hell" in n: return "Hellinger"
    if "two" in n: return "Twoing"

    return name

# ------------------------------
# Load dataset
# ------------------------------
def load_dataset(url, cols):
    import pandas as pd
    log("Loading dataset...")
    df = pd.read_csv(url, names=cols, na_values="?").dropna()
    log(f"Rows after dropna: {len(df)}")

    le = LabelEncoder()
    for c in df.columns:
        df[c] = le.fit_transform(df[c].astype(str))

    X = df.drop("class", axis=1).values
    y = df["class"].values

    log("Dataset loaded & encoded.")
    return train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Build 24 models
# ------------------------------
def build_models():
    log("Building model objects...")
    n_est = 50

    base = [
        DT_Entropy(), DT_Gini(), DT_GainRatio(),
        DT_ChiSquare(), DT_Hellinger(), DT_Twoing()
    ]

    pruned = [
        PruningWrapper(DT_Entropy()),
        PruningWrapper(DT_Gini()),
        PruningWrapper(DT_GainRatio()),
        PruningWrapper(DT_ChiSquare()),
        PruningWrapper(DT_Hellinger()),
        PruningWrapper(DT_Twoing())
    ]

    bagged = [
        BaggingWrapper(DT_Entropy(), n_est),
        BaggingWrapper(DT_Gini(), n_est),
        BaggingWrapper(DT_GainRatio(), n_est),
        BaggingWrapper(DT_ChiSquare(), n_est),
        BaggingWrapper(DT_Hellinger(), n_est),
        BaggingWrapper(DT_Twoing(), n_est)
    ]

    hybrid = [
        BaggingWrapper(PruningWrapper(DT_Entropy()), n_est),
        BaggingWrapper(PruningWrapper(DT_Gini()), n_est),
        BaggingWrapper(PruningWrapper(DT_GainRatio()), n_est),
        BaggingWrapper(PruningWrapper(DT_ChiSquare()), n_est),
        BaggingWrapper(PruningWrapper(DT_Hellinger()), n_est),
        BaggingWrapper(PruningWrapper(DT_Twoing()), n_est)
    ]

    return base, pruned, bagged, hybrid

# ------------------------------
# Evaluate 24 models → accuracy list
# ------------------------------
def evaluate(dataset):
    log(f"Evaluating dataset: {dataset['name']}")
    X_train, X_test, y_train, y_test = load_dataset(dataset["url"], dataset["cols"])

    models_base, models_pruned, models_bagged, models_hybrid = build_models()

    accuracies = []

    for i in range(6):
        m_b  = models_base[i]
        m_p  = models_pruned[i]
        m_bg = models_bagged[i]
        m_h  = models_hybrid[i]

        base_name = normalize_criterion(m_b.name)
        log(f"\n------ Criterion: {base_name} ------")

        group = [
            (m_b,  "Base"),
            (m_p,  "Pruned"),
            (m_bg, "Bagged"),
            (m_h,  "Hybrid")
        ]

        for model, model_type in group:
            log(f"Fitting {model.name} ({model_type})...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(np.mean(preds == y_test))
            log(f"Accuracy: {acc:.4f}")
            accuracies.append((base_name, model_type, acc))

    return accuracies


# ------------------------------
# Plotting function
# ------------------------------
from matplotlib.lines import Line2D

def plot_final(acc_list, dataset_name, save_path):

    CRITERIA = ["Entropy", "Gini", "Gain Ratio", "Chi-Square", "Hellinger", "Twoing"]
    MODELS   = ["Base", "Pruned", "Bagged", "Hybrid"]

    COLORS = {
        "Base": "#1f77b4",
        "Pruned": "#2ca02c",
        "Bagged": "#ff7f0e",
        "Hybrid": "#d62728"
    }

    MARKERS = {
        "Base": "o",
        "Pruned": "s",
        "Bagged": "D",
        "Hybrid": "^"
    }

    OFFSETS = {
        "Base":   -0.28,
        "Pruned": -0.10,
        "Bagged":  0.10,
        "Hybrid":  0.28
    }

    LABEL_OFFSET_Y = {
        "Base": 0.004, "Pruned": 0.006,
        "Bagged": -0.008, "Hybrid": 0.009
    }

    LABEL_OFFSET_X = {
        "Base": 0.00, "Pruned": 0.01,
        "Bagged": 0.01, "Hybrid": -0.01
    }

    # ------------------------
    spacing = 1.30
    x_numeric = [i * spacing for i in range(len(CRITERIA))]
    # ------------------------

    all_acc = [x[2] for x in acc_list]
    ymin = min(all_acc)
    ymax = max(all_acc)
    yrange = ymax - ymin if ymax > ymin else 0.01

    y_lower = ymin - 0.02 * yrange
    y_upper = ymax + 0.06 * yrange

    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")

    col_half_width = 0.60
    for i in range(len(CRITERIA)):
        cx = i * spacing
        plt.axvspan(cx - col_half_width, cx + col_half_width,
                    color="gray", alpha=0.030, zorder=0)

    for i in range(len(CRITERIA) - 1):
        sep_x = (i + 0.5) * spacing
        plt.axvline(sep_x, color='gray', linestyle='-',
                    linewidth=0.6, alpha=0.28, zorder=1)

    for crit in CRITERIA:
        crit_idx = CRITERIA.index(crit)

        for mtype in MODELS:
            match = [x for x in acc_list if x[0] == crit and x[1] == mtype]
            if not match:
                continue

            acc = match[0][2]
            x_pos = crit_idx * spacing + OFFSETS[mtype]

            plt.scatter(
                x_pos, acc, s=150,
                color=COLORS[mtype], marker=MARKERS[mtype],
                edgecolor="black", linewidth=0.45, zorder=3
            )

            # plt.text(
            #     x_pos,
            #     acc + 0.01 * yrange,     # consistent offset for all models
            #     f"{acc:.2f}",
            #     ha='center', fontsize=7.5, fontweight='bold',
            #     color='black',
            #     bbox=dict(facecolor='white', alpha=0.60,
            #             edgecolor='none', pad=0.8),
            #     zorder=4
            # )

    plt.xticks(x_numeric, CRITERIA, fontsize=13)
    plt.ylim(y_lower, y_upper)
    plt.xlabel(None)
    plt.ylabel(None)
    # plt.xlabel("Splitting Criterion", fontsize=14)
    # plt.ylabel("Accuracy", fontsize=14)
    # plt.title(f"Model Comparison on {dataset_name}",
    #           fontsize=20, fontweight="bold", pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.33)

    proxies = [
        Line2D([0], [0], marker=MARKERS[m], color='w',
               markerfacecolor=COLORS[m], markeredgecolor='black',
               markersize=8, linestyle='None', markeredgewidth=0.6)
        for m in MODELS
    ]

    plt.legend(
        proxies, MODELS,
        title="Model Type",
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        frameon=True,
        fontsize=11,
        title_fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------
# MAIN — run for ALL datasets
# ------------------------------
if __name__ == "__main__":

    os.makedirs("plots", exist_ok=True)

    for dataset in datasets:
        log(f"\n\n=== Processing dataset: {dataset['name']} ===")
        acc_list = evaluate(dataset)

        safe_name = dataset["name"].replace(" ", "_").lower()
        save_path = f"plots/{safe_name}.png"

        plot_final(acc_list, dataset["name"], save_path)
        log(f"Saved plot to {save_path}")

    log("All datasets processed.")
