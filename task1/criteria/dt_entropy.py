from base.dt_base import DecisionTreeBase
import numpy as np

class DT_Entropy(DecisionTreeBase):
    """
    Decision Tree using Entropy (ID3 criterion)
    Gain = Entropy(parent) - [w_left * Entropy(left) + w_right * Entropy(right)]
    """

    def __init__(self):
        super().__init__("Entropy")

    def entropy(self, y):
        probs = np.bincount(y.astype(int)) / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def criterion(self, y_left, y_right, y_parent):
        parent_entropy = self.entropy(y_parent)
        left_entropy = self.entropy(y_left)
        right_entropy = self.entropy(y_right)
        w_left = len(y_left) / len(y_parent)
        w_right = len(y_right) / len(y_parent)
        info_gain = parent_entropy - (w_left * left_entropy + w_right * right_entropy)
        return info_gain
