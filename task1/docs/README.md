## Overview
This directory contains theoretical documentation for **six different Decision Tree splitting criteria**,  
implemented *from scratch* in the `criteria/` module.

Each criterion is explained in a separate `.md` file with:
- Mathematical formulation  
- Conceptual explanation  
- Strengths and limitations  
- Key references (where relevant)

The goal of Task 1 is to compare these six methods on the **UCI Car Evaluation dataset**  
and determine which yields the best classification performance.

---

## List of Implemented Criteria

| Criterion | File | Concept Summary |
|------------|------|-----------------|
| **Entropy** | [entropy.md](entropy.md) | Measures information uncertainty. Split chosen to maximize Information Gain (ID3). |
| **Gini Index** | [gini.md](gini.md) | Measures node impurity as probability of misclassification (CART). |
| **Gain Ratio** | [gain_ratio.md](gain_ratio.md) | Normalizes Information Gain to reduce bias toward multi-valued features (C4.5). |
| **Chi-Square** | [chi_square.md](chi_square.md) | Uses χ² test to check independence between attribute and target (CHAID). |
| **Hellinger Distance** | [hellinger.md](hellinger.md) | Measures divergence between class distributions; robust for imbalanced data. |
| **Twoing Rule** | [twoing.md](twoing.md) | Maximizes separation between class probabilities of child nodes (CART variant). |

---

## Research Context
The six criteria are derived from classic decision tree algorithms:

| Algorithm | Criterion | Reference |
|------------|------------|------------|
| ID3 | Entropy | Quinlan (1986), *Induction of Decision Trees* |
| C4.5 | Gain Ratio | Quinlan (1993), *Programs for Machine Learning* |
| CART | Gini Index, Twoing Rule | Breiman et al. (1984), *Classification and Regression Trees* |
| CHAID | Chi-Square | Kass (1980), *Applied Statistics* |
| HDDT | Hellinger Distance | Cieslak & Chawla (2008), *ECML* |

---

## Folder Structure
docs/
├── README_index.md # You are here
├── README_entropy.md
├── README_gini.md
├── README_gain_ratio.md
├── README_chi_square.md
├── README_hellinger.md
└── README_twoing.md

## Usage in the Project
Each criterion’s implementation resides in:
task1/criteria/dt_<criterion>.py

They all inherit from the common base class:
task1/base/dt_base.py

The evaluation script:
task1/main_compare_dt.py
trains all six trees, compares their accuracies, and outputs a metrics table.

---

## Credits and References
1. Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*. Machine Learning, 1(1).  
2. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.  
3. Breiman, L. et al. (1984). *Classification and Regression Trees (CART)*. Wadsworth International.  
4. Kass, G. V. (1980). *CHAID: An Exploratory Technique*. Applied Statistics, 29(2).  
5. Cieslak, D. A. & Chawla, N. V. (2008). *Hellinger Distance Decision Trees*. ECML.

---
