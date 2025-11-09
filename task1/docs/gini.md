# Gini Index Splitting Criterion (CART Algorithm)

## Overview
The Gini index measures how often a randomly chosen sample would be incorrectly classified if it were labeled according to the class distribution in a node.

## Mathematical Formulation

\[
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
\]
where \( p_i \) is the proportion of class \( i \) in node \( S \).

The impurity reduction (gain) from splitting on attribute \( A \):
\[
Gain(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)
\]

## Interpretation
- \( Gini = 0 \) → perfectly pure node.  
- \( Gini = 0.5 \) (for 2-class) → maximum impurity.

## Strengths
- Simple and fast (no logs).  
- Works well with continuous and categorical data.

## Limitations
- Slight bias toward larger class distributions.  

## Key Reference
- Breiman, L. et al. (1984). *Classification and Regression Trees (CART)*. Wadsworth International.
