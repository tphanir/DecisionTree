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

```
task1/
├── base/ 
├── criteria/
├── docs/
├── main.py
└── README.md 

base : Abstract base class for all trees
criteria: Six splitting criteria implementations
docs: Theoretical documentation for each criterion
main.py: Runs and compares all DT variants
```
---

## Next Steps — Moving Toward Task 2

After completing Task 1, we now have a clear understanding of how different node-splitting strategies  
affect model performance. Among all six criteria, **Gain Ratio** and **Chi-Square** produced the best results.  

After team discussion, we decided to **proceed with Gain Ratio** as our base method for Task 2,  
for the following reasons:
- It builds on **information-theoretic principles** and provides a strong theoretical foundation.  
- It already corrects for **bias in Information Gain**, making it a balanced starting point.  
- It has consistent performance across categorical datasets like Car Evaluation.

---

### Step 1 — Define the Research Goal
In **Task 2**, we aim to **improve the Gain Ratio** mathematically.  
We will explore how to make it more adaptive by incorporating additional statistical measures or  
data-dependent weighting.

Possible ideas include:
- Introducing a **dynamic normalization factor** that depends on dataset entropy.  
- Combining **Gain Ratio** with **Chi-Square significance** for stronger statistical grounding.  
- Experimenting with **weighted entropy** to handle uneven attribute cardinalities.

---

### Step 2 — Develop and Derive the Mathematical Model
We will formally derive the proposed criterion:
- Express it as a modified Gain Ratio formulation.  
- Justify it mathematically using proofs or reasoning (3–4 pages).  
- Analyze its theoretical impact on impurity reduction and bias.

This step will form the **core theoretical section** of our final report.

---

### Step 3 — Implement the New Criterion
- Add a new file: `criteria/dt_proposed.py`  
- Implement the new splitting rule by extending the existing abstract base class.  
- Reuse the same `main.py` setup to evaluate it against all existing criteria.  
- Log and compare accuracy, precision, recall, and F1-score.

---

### Step 4 — Visualize and Interpret
- Plot results for all criteria (Accuracy, F1, etc.)  
- Visualize comparative performance between **existing vs proposed method**  
- Save plots under `reports/output_graphs/` for inclusion in the report.

---

### Step 5 — Document and Report
We will consolidate everything into a structured LaTeX report (approximately 8 pages) containing:
1. Title and Abstract  
2. Introduction and Related Work  
3. Mathematical Formulation of the Proposed Criterion  
4. Implementation Details  
5. Results and Discussion (with graphs and tables)  
6. Conclusion and Future Scope  
7. References  

The **Task 1 documentation (`docs/` folder)** will serve as the background theory for Section 2.

---

## References
1. Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*. Machine Learning, 1(1), 81–106.  
2. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.  
3. Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. (1984). *Classification and Regression Trees (CART)*.  
4. Kass, G. V. (1980). *CHAID: An Exploratory Technique for Investigating Large Quantities of Categorical Data*. Applied Statistics, 29(2), 119–127.  
5. Cieslak, D. A., and Chawla, N. V. (2008). *Hellinger Distance Decision Trees (HDDT)*. ECML Proceedings.

---

**Team Summary:**  
This task was completed collaboratively by a team of six members,  
each implementing one splitting criterion and contributing a theoretical overview in the `docs/` folder.

**Next Milestone:**  
Proceed to **Task 2 – Enhancing the Gain Ratio Criterion** through mathematical modeling, theoretical justification, and experimental validation.
