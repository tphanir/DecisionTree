# Gain Ratio Splitting Criterion (C4.5 Algorithm)

## Overview
Gain Ratio improves the Information Gain criterion by normalizing it with the **intrinsic information** of a split.  
This correction removes the bias of Information Gain toward attributes with many distinct values.

## Mathematical Formulation

Information Gain:

$$
Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
$$

Split Information:

$$
SplitInfo(A) = - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2 \left( \frac{|S_v|}{|S|} \right)
$$

Gain Ratio:

$$
GainRatio(S, A) = \frac{Gain(S, A)}{SplitInfo(A)}
$$

where:

- $S_v$ = subset of samples where attribute $A$ takes value $v$

## Strengths
- Corrects multi-valued attribute bias  
- Produces more balanced decision trees  

## Limitations
- Can be unstable when $SplitInfo(A)$ is close to zero  
- Slightly more computationally intensive  

## Reference
Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
