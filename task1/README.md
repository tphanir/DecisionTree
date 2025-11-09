# Task 1 — Decision Tree Splitting Criteria Comparison (From Scratch)

## Overview
This task aims to **implement and compare six Decision Tree splitting criteria** entirely from scratch  
using the **UCI Car Evaluation dataset**. Each team member implemented one criterion, and all were evaluated  
through a common modular framework defined in `base/dt_base.py`.

The objective was to determine which criterion performs best before proceeding to mathematical modeling  
and theoretical improvement in **Task 2**.

---

## Implemented Criteria

| No. | Criterion | Algorithm Reference | File |
|----:|------------|---------------------|------|
| 1 | Entropy | ID3 (Quinlan, 1986) | `criteria/dt_entropy.py` |
| 2 | Gini Index | CART (Breiman et al., 1984) | `criteria/dt_gini.py` |
| 3 | Gain Ratio | C4.5 (Quinlan, 1993) | `criteria/dt_gain_ratio.py` |
| 4 | Chi-Square | CHAID (Kass, 1980) | `criteria/dt_chi_square.py` |
| 5 | Hellinger Distance | HDDT (Cieslak & Chawla, 2008) | `criteria/dt_hellinger.py` |
| 6 | Twoing Rule | CART Variant | `criteria/dt_twoing.py` |

All six implementations extend a common abstract class defined in `base/dt_base.py`.  
The evaluation pipeline (`main_compare_dt.py`) ensures identical training/testing splits and metrics across all models.

---

## Results on UCI Car Evaluation Dataset

| Criterion | Accuracy |
|------------|-----------|
| Entropy | 0.8632 |
| Gini Index | 0.8632 |
| Gain Ratio | **0.8748** |
| Chi-Square | **0.8748** |
| Hellinger Distance | 0.6898 |
| Twoing Rule | 0.8632 |

**Best performing criteria:** Gain Ratio and Chi-Square  
Both achieved an accuracy of **0.8748**, outperforming the rest on this dataset.

---

## Directory Structure
task1/
├── base/ # Abstract base class for all trees
├── criteria/ # Six splitting criteria implementations
├── data/ # (Optional) dataset files
├── docs/ # Theoretical documentation for each criterion
├── utils/ # Metrics, plotting, data loading
├── main_compare_dt.py # Runs and compares all DT variants
└── README.md # Current file

---

## Next Steps (Task 2)

### 1. Select the Base Criterion
Since both **Gain Ratio** and **Chi-Square** performed best, choose one as your **base algorithm**  
for further theoretical and experimental improvement.

Suggested direction:  
> Use **Gain Ratio** as the base — it combines information-theoretic rigor and generalization strength.

---

### 2. Formulate Mathematical Model
- Develop a **mathematical enhancement** over the chosen criterion.  
  Possible directions:
  - Introduce a penalty or weighting factor for attribute cardinality.  
  - Blend statistical significance (like χ²) with normalized information gain.  
  - Optimize splitting using reinforcement or probabilistic weighting.

- Prepare **3–4 pages** of derivation with theorems, lemmas, and proofs.  
  (This forms the core of Task 2 in your report.)

---

### 3. Implement the New Criterion
- Add a new file, e.g.,  
  `criteria/dt_proposed.py`
- Integrate it into the same evaluation framework (just one class addition).  
- Compare its results against the existing six criteria on the same dataset.

---

### 4. Generate Comparison Graphs
- Use line or bar charts to visualize accuracy, precision, recall, and F1 scores.  
- Compare:
  - Old vs New criterion  
  - All criteria together  
- Save graphs in `reports/output_graphs/`.

---

### 5. Report Preparation (LaTeX)
Prepare an **8-page report** in LaTeX containing:
1. Title & Abstract  
2. Introduction & Related Work  
3. Mathematical Formulation (Proposed Model)  
4. Implementation Details  
5. Results & Discussion (with graphs and tables)  
6. Conclusion & Future Scope  
7. References  

Refer to `task1/docs/` for one-page write-ups of each splitting criterion.  

---

## References
1. Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*. Machine Learning, 1(1), 81–106.  
2. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.  
3. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees (CART)*.  
4. Kass, G. V. (1980). *CHAID: An Exploratory Technique for Investigating Large Quantities of Categorical Data*. Applied Statistics, 29(2), 119–127.  
5. Cieslak, D. A., & Chawla, N. V. (2008). *Hellinger Distance Decision Trees (HDDT)*. ECML Proceedings.

---

**Team:** 6 Members  
Each member implemented one splitting criterion and contributed to theoretical documentation in `docs/`.

**Next Milestone:** Move to **Task 2 – Mathematical Model Development and Improvement of Gain Ratio C