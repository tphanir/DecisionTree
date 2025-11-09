# Entropy-Based Splitting Criterion (ID3 Algorithm)

## Overview
The Entropy criterion measures the amount of uncertainty or impurity in a dataset.  
It selects the attribute that gives the **maximum information gain** — i.e., the attribute that most reduces entropy after splitting.

## Mathematical Formulation

Entropy of a node:

$$
Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$

Information Gain for a split on attribute \(A\):

$$
Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
$$

where:
- \(p_i\) = proportion of samples in class \(i\) within node \(S\)
- \(S_v\) = subset of samples where attribute \(A\) takes value \(v\)

## Strengths
- Theoretically grounded in information theory  
- Produces interpretable, compact trees  

## Limitations
- Biased toward attributes with many unique values  
- Slightly computationally heavier (logarithms)

## Reference
Quinlan, J. R. (1986). *Induction of Decision Trees (ID3)*. *Machine Learning*, 1(1), 81–106.
