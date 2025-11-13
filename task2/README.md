# Task 2 — Decision Tree Optimizations: Bagging and Pruning

## Overview

This task builds directly on the foundations of **Task 1**.  
While Task 1 focused on *which splitting criterion* to use (Entropy, Gini, etc.), Task 2 focuses on *how to build the tree* to improve its performance, independent of the criterion.

The goal is to implement and analyze two powerful decision tree optimizations:

- **Bagging (Bootstrap Aggregating):** Reduces model variance  
- **Reduced Error Pruning (REP):** Controls model complexity and prevents overfitting  

Both optimizations are implemented as **wrappers** applicable to any of the Task 1 base tree models.

---

## Implemented Optimizations

### 1. Bagging (Bootstrap Aggregating)

Bagging trains multiple base estimators on bootstrap samples (sampling with replacement) and aggregates their predictions by majority vote.

- **Goal:** Reduce variance in decision trees  
- **Implementation:** `base/bagging_wrapper.py`  
- **Analysis Script:** `main_bagging.py`  
- **Results:** `results/bagging_10_datasets.csv`  
- **Theoretical Report:** `reports/bagging.tex`  

---

### 2. Reduced Error Pruning (REP)

REP is a post-pruning technique. It starts by training a very deep, overfit tree, then recursively prunes branches bottom-up using a validation set.

- **Goal:** Find optimal tree complexity by controlling bias and variance  
- **Implementation:** `base/pruning_wrapper.py`  
- **Analysis Script:** `main_pruning_comparison.py`
- **Results:** `results/pruning_10_datasets.csv`  
- **Theoretical Report:** `reports/pruning.tex`  

---

## Core Components (from Task 1)

This project relies on the modular base built in Task 1:

- `base/dt_base.py` — Abstract base class for all decision trees  
- `criteria/*.py` — Six splitting criteria:  
  Entropy, Gini, Gain Ratio, Chi-Square, Hellinger Distance, Twoing Rule  
- `constants.py` — Definitions of the 10 UCI datasets used for evaluation  

---

## How to Run

### Bagging vs. Base Comparison
```bash
python main.py
```

Outputs: results/bagging_10_datasets.csv

### Pruning vs. Base Comparison
```bash
python main_pruning_comparison.py
```

Outputs: results/pruning_10_datasets.csv


### Directory Structure
```
task2/
├── base/
│   ├── dt_base.py               # Abstract decision tree class (Task 1)
│   ├── bagging_wrapper.py       # Bagging implementation
│   └── pruning_wrapper.py       # Reduced Error Pruning implementation
│
├── criteria/
│   ├── dt_entropy.py
│   ├── dt_gini.py
│   ├── dt_gain_ratio.py
│   ├── dt_chi_square.py
│   ├── dt_hellinger.py
│   └── dt_twoing.py
│
├── plots/                       # All auto-generated line & bar plots
│   ├── *.png
│
├── reports/
│   ├── bagging.tex              # LaTeX theoretical analysis (Bagging)
│   └── pruning.tex              # LaTeX theoretical analysis (Pruning)
│
├── results/
│   ├── bagging_10_datasets.csv  # Bagging vs. base comparison
│   └── pruning_10_datasets.csv  # Pruning vs. base comparison
│
├── constants.py                 # Dataset definitions
├── main_bagging.py              # Bagging evaluation runner
├── main_pruning.py              # Pruning evaluation runner
└── README.md                   
```