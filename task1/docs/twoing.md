# Twoing Rule Splitting Criterion (CART Variant)

## Overview
The Twoing Rule is an alternative CART splitting function that maximizes the difference between class distributions of left and right child nodes.

## Mathematical Formulation

\[
Gain = 0.25 \times P_L \times P_R \times 
\left( \sum_{j=1}^{c} |p_{L,j} - p_{R,j}| \right)^2
\]
where:
- \( P_L, P_R \): Fraction of samples going left/right  
- \( p_{L,j}, p_{R,j} \): Class probabilities in left/right child nodes  

## Interpretation
- Measures the *separability* of class distributions between children.  
- High gain → better split that distinguishes classes strongly.

## Strengths
- Encourages well-balanced, class-separating splits.  
- Performs well in multi-class classification.

## Limitations
- Sensitive to class imbalance.  
- Slightly more computationally complex than Gini.

## Key Reference
- Breiman, L. et al. (1984). *Classification and Regression Trees (CART)*. Wadsworth International.
