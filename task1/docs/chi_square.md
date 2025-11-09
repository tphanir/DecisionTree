# Chi-Square Splitting Criterion (CHAID Algorithm)

## Overview
The Chi-Square (χ²) criterion tests the statistical independence between an attribute and the target class.  
A split is made if the χ² statistic is large (indicating dependence).

## Mathematical Formulation

\[
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
\]
where:
- \( O_{ij} \): Observed frequency for class \( i \) and attribute value \( j \)  
- \( E_{ij} \): Expected frequency assuming independence

## Interpretation
A higher χ² value implies a more significant association between attribute and class.

## Strengths
- Handles multiway splits directly.  
- Provides statistical significance for each split.

## Limitations
- Requires sufficiently large sample sizes for valid χ² approximation.  
- Not ideal for continuous attributes.

## Key Reference
- Kass, G. V. (1980). *An exploratory technique for investigating large quantities of categorical data (CHAID)*. *Applied Statistics*, 29(2), 119–127.
