# Task 1 — Decision Tree Splitting Criteria Comparison (From Scratch)

## Overview
This task aims to **implement and compare six Decision Tree splitting criteria** entirely from scratch  
using multiple **categorical datasets** from the **UCI Machine Learning Repository**.  

Each team member implemented one criterion, and all were evaluated through a **common modular framework**  
defined in `base/dt_base.py`.  

Initially, experiments were conducted only on the **UCI Car Evaluation dataset**,  
but we later extended our evaluation to a **broader set of 10 datasets** (e.g., Mushroom, Nursery, Tic-Tac-Toe, Credit Approval, etc.)  
to assess the generalization of different criteria.

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

Each implementation extends a shared abstract base class in `base/dt_base.py`,  
ensuring identical data handling, training, and evaluation pipelines.

---

## Results Across Multiple Datasets

After expanding our evaluation to **10 categorical datasets**,  
we observed that **no single splitting criterion consistently outperformed others**.  
Performance varied depending on dataset characteristics such as:
- Number of features and class imbalance  
- Attribute cardinality (number of unique values per feature)  
- Noise and dependency structure among features  

For example:
- **Gain Ratio** and **Chi-Square** performed well on multi-class, balanced datasets.  
- **Entropy** and **Gini** were more stable on smaller or balanced datasets.  
- **Hellinger Distance** was effective for imbalanced datasets but underperformed elsewhere.  

This demonstrated that **the splitting criterion alone does not define tree performance** —  
rather, **the tree construction strategy** plays a larger role in improving generalization.

---

## Directory Structure

```
task1/
├── base/
│ ├── dt_base.py # Abstract base class for Decision Trees
├── criteria/
│ ├── dt_entropy.py
│ ├── dt_gini.py
│ ├── dt_gain_ratio.py
│ ├── dt_chi_square.py
│ ├── dt_hellinger.py
│ └── dt_twoing.py
├── docs/ # Theoretical writeups for each splitting criterion
├── main.py # Runs and compares all DT variants
└── README.md # Project documentation
```

---

## Insights from Task 1

Through the 10-dataset comparison, we conclude that:
- **Split criteria are dataset-dependent.**  
  No single formula (Entropy, Gini, Gain Ratio, etc.) universally dominates.  
- **Tree design and optimization matter more** than the impurity metric itself.  
- Improving **how the tree learns**, rather than **what it optimizes**,  
  can lead to consistent gains across datasets.

---

## Moving Forward — Tree Optimization Independent of Split Criteria

In the next phase, our goal is to make the **Decision Tree learning process more robust and adaptive**,  
irrespective of which splitting function is used.

### Proposed Optimization Directions

#### 1. Bagging (Bootstrap Aggregation)
- Train multiple Decision Trees on bootstrap samples and aggregate predictions by majority vote.  
- Reduces model variance and improves stability across datasets.  
- Implemented as a reusable `BaggingClassifier` in `base/bagging_wrapper.py`.

#### 2. Cost-Complexity Pruning
- Prevent overfitting by penalizing overly deep or complex trees.  
- Use an α-regularization term to balance between accuracy and simplicity.  
- Can be implemented as a post-pruning step that evaluates subtrees on validation sets.

#### 3. Oblique and Soft Splits
- Extend axis-aligned splits to **linear combinations of features (oblique splits)**.  
- Use **soft probabilistic splits** for smoother decision boundaries and differentiable tree models.  
- Enhances the expressive power of trees while keeping interpretability.

---

## Next Steps — Toward Task 2

1. **Evaluate bagged and pruned versions** of existing trees on the same 10 datasets  
   to assess consistency improvements.  
2. **Formulate a generalized optimization framework** for Decision Trees  
   that can be applied independent of the splitting criterion.  
3. **Develop theoretical justification** for variance reduction, generalization,  
   and complexity control, supported by mathematical derivations and empirical evidence.

---

## References
1. Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*. Machine Learning, 1(1), 81–106.  
2. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.  
3. Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. (1984). *Classification and Regression Trees (CART)*.  
4. Kass, G. V. (1980). *CHAID: An Exploratory Technique for Investigating Large Quantities of Categorical Data*. Applied Statistics, 29(2), 119–127.  
5. Cieslak, D. A., and Chawla, N. V. (2008). *Hellinger Distance Decision Trees (HDDT)*. ECML Proceedings.  
6. Breiman, L. (1996). *Bagging Predictors*. *Machine Learning*, 24(2), 123–140.

---

**Team Summary:**  
This task was completed collaboratively by a team of six members,  
each implementing one splitting criterion and contributing theoretical documentation in `docs/`.  

**Next Milestone:**  
Proceed to **Task 2 — Improving Decision Trees independent of Split Criteria**,  
by implementing bagging, pruning, and structural optimization techniques for enhanced generalization.