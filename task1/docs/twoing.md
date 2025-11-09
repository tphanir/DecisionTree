# Twoing Rule Splitting Criterion (CART Variant)

## Overview
The Twoing Rule, introduced as part of CART, evaluates how well a split differentiates  
class distributions between the left and right branches. It is most useful in multi-class classification.

## Mathematical Formulation

Twoing Gain:

$$
Gain = 0.25 \times P_L \times P_R \times \left( \sum_{j=1}^{c} |p_{L,j} - p_{R,j}| \right)^2
$$

where:

- $P_L$, $P_R$ = fractions of samples sent to the left and right child nodes  
- $p_{L,j}$, $p_{R,j}$ = probabilities of class $j$ in the left and right child respectively  

## Strengths
- Encourages splits that strongly separate class distributions  
- Handles multi-class data effectively  

## Limitations
- Sensitive to class imbalance  
- May prefer slightly unbalanced splits  

## Reference
Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).  
*Classification and Regression Trees (CART)*. Wadsworth International.
