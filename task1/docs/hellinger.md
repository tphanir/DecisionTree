# Hellinger Distance Splitting Criterion

## Overview
Hellinger distance measures the divergence between two probability distributions (e.g., class distributions in left and right child nodes).  
It is especially effective for imbalanced datasets.

## Mathematical Formulation

\[
H(P, Q) = \sqrt{1 - \sum_{i=1}^{c} \sqrt{p_i q_i}}
\]
where \( P \) and \( Q \) are class probability distributions in left and right splits.

For splitting:
\[
Gain = 1 - H(P, Q)
\]

## Interpretation
- Lower Hellinger distance → distributions are similar (bad split)  
- Higher Hellinger gain → better separation

## Strengths
- Robust against class imbalance.  
- Unaffected by class priors.

## Limitations
- Slightly higher computational cost (square roots).  
- Less intuitive interpretation.

## Key Reference
- Cieslak, D. A. & Chawla, N. V. (2008). *Hellinger Distance Decision Trees (HDDT)*, *ECML*.
