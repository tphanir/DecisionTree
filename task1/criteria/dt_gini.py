from base.dt_base import DecisionTreeBase
import numpy as np

class DT_Gini(DecisionTreeBase):
    """
    Decision Tree using Gini Index (CART criterion)
    Gain = Gini(parent) - [w_left * Gini(left) + w_right * Gini(right)]
    """

    def __init__(self):
        super().__init__("Gini Index")

    def gini(self, y):
        probs = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(probs ** 2)

    def criterion(self, y_left, y_right, y_parent):
        parent_gini = self.gini(y_parent)
        left_gini = self.gini(y_left)
        right_gini = self.gini(y_right)
        w_left = len(y_left) / len(y_parent)
        w_right = len(y_right) / len(y_parent)
        gain = parent_gini - (w_left * left_gini + w_right * right_gini)
        return gain
