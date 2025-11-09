# Chi-Square Splitting Criterion (CHAID Algorithm)

## Overview
The Chi-Square ($\chi^2$) criterion tests whether an attribute and the target class are statistically independent.  
A split is chosen when the $\chi^2$ statistic shows significant association between the attribute and the class.

## Mathematical Formulation

Chi-Square statistic:

$$
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

where:

- $O_{ij}$ = observed frequency of class $i$ for attribute value $j$  
- $E_{ij}$ = expected frequency assuming independence  

The attribute with the highest $\chi^2$ value is chosen for splitting.

## Strengths
- Statistically interpretable (based on hypothesis testing)  
- Supports multi-way splits naturally  

## Limitations
- Requires large sample sizes for stability  
- Not ideal for continuous attributes without discretization  

## Reference
Kass, G. V. (1980). *An Exploratory Technique for Investigating Large Quantities of Categorical Data (CHAID)*.  
*Applied Statistics*, 29(2), 119–127.
