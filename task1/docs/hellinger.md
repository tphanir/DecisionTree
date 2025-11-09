# Hellinger Distance Splitting Criterion

## Overview
Hellinger Distance measures the divergence between class-probability distributions of left and right child nodes.  
It is particularly effective for **imbalanced datasets**, where class priors differ significantly.

## Mathematical Formulation

Hellinger Distance:

$$
H(P, Q) = \sqrt{1 - \sum_{i=1}^{c} \sqrt{p_i \, q_i}}
$$

Hellinger Gain:

$$
Gain = 1 - H(P, Q)
$$

where:

- $P = (p_1, p_2, \ldots, p_c)$ = class probabilities in the left child  
- $Q = (q_1, q_2, \ldots, q_c)$ = class probabilities in the right child  

## Strengths
- Robust against class imbalance and skewed priors  
- Insensitive to overall class frequency differences  

## Limitations
- Slightly higher computational cost due to square roots  
- Less commonly available in standard ML libraries  

## Reference
Cieslak, D. A., & Chawla, N. V. (2008). *Hellinger Distance Decision Trees (HDDT)*. ECML Proceedings.
