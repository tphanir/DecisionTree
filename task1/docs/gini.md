# Gini Index Splitting Criterion (CART Algorithm)

## Overview
The Gini Index measures the impurity of a node by estimating the probability that a randomly chosen sample  
would be incorrectly classified if it were labeled according to the class distribution within that node.

## Mathematical Formulation

Gini impurity of a node:

$$
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
$$

Impurity reduction (gain) for a split on attribute $A$:

$$
Gain(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)
$$

where:

- $p_i$ = proportion of class $i$ samples within node $S$
- $S_v$ = subset of samples where attribute $A$ takes value $v$

## Strengths
- Fast to compute (no logarithms)  
- Handles both continuous and categorical attributes efficiently  

## Limitations
- Slight bias toward dominant classes  
- May produce slightly unbalanced splits in some datasets  

## Reference
Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).  
*Classification and Regression Trees (CART)*. Wadsworth International.
