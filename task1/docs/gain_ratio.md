# Gain Ratio Splitting Criterion (C4.5 Algorithm)

## Overview
The Gain Ratio normalizes information gain by the "intrinsic information" of a split,  
correcting the bias of entropy-based information gain toward attributes with many distinct values.

## Mathematical Formulation

Information Gain:
\[
Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)
\]

Split Information:
\[
SplitInfo(A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)
\]

Gain Ratio:
\[
GainRatio(S, A) = \frac{Gain(S, A)}{SplitInfo(A)}
\]

## Interpretation
Higher Gain Ratio → better attribute choice (maximizes info gain per unit of intrinsic split).

## Strengths
- Removes bias toward attributes with many unique values.  
- Produces more balanced decision trees.

## Limitations
- SplitInfo can be small, making ratio unstable in rare cases.

## Key Reference
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
