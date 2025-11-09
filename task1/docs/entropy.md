# Entropy-Based Splitting Criterion (ID3 Algorithm)

## Overview
Entropy measures the impurity or uncertainty in a dataset.  
A split is chosen such that it **maximizes the reduction in entropy** (i.e., maximizes information gain).

## Mathematical Formulation

Entropy of a dataset \( S \):
\[
Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]
where \( p_i \) is the probability of class \( i \) in node \( S \).

Information Gain for splitting attribute \( A \):
\[
Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
\]

## Interpretation
- **High entropy** → classes mixed (uncertain).
- **Low entropy** → node is pure (certain).

## Strengths
- Intuitive and theoretically sound (from information theory).  
- Produces compact trees.

## Limitations
- Biased toward attributes with many distinct values.  
- Computationally expensive due to logarithmic terms.

## Key Reference
- Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*, *Machine Learning*, 1(1), 81–106.
