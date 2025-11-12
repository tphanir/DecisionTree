from base.dt_base import DecisionTreeBase
import numpy as np

class DT_Twoing(DecisionTreeBase):
    """
    Decision Tree using Twoing Rule (CART variant)
    Gain = 0.25 * P(L) * P(R) * (Σ |p(L,j) - p(R,j)|)²
    """

    def __init__(self):
        super().__init__("Twoing Rule")

    def criterion(self, y_left, y_right, y_parent):
        total = len(y_left) + len(y_right)
        pL, pR = len(y_left) / total, len(y_right) / total
        classes = np.unique(y_parent)
        diff_sum = np.sum([
            abs(np.sum(y_left == c) / len(y_left) - np.sum(y_right == c) / len(y_right))
            for c in classes
        ])
        return 0.25 * pL * pR * (diff_sum ** 2)
