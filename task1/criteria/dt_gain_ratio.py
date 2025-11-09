from base.dt_base import DecisionTreeBase
import numpy as np

class DT_GainRatio(DecisionTreeBase):
    """
    Decision Tree using Gain Ratio (C4.5 criterion)
    GainRatio = InfoGain / SplitInfo
    """

    def __init__(self):
        super().__init__("Gain Ratio")

    def entropy(self, y):
        probs = np.bincount(y.astype(int)) / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def split_info(self, y_left, y_right, y_parent):
        sizes = np.array([len(y_left), len(y_right)]) / len(y_parent)
        return -np.sum([s * np.log2(s) for s in sizes if s > 0])

    def criterion(self, y_left, y_right, y_parent):
        parent_entropy = self.entropy(y_parent)
        left_entropy = self.entropy(y_left)
        right_entropy = self.entropy(y_right)
        w_left = len(y_left) / len(y_parent)
        w_right = len(y_right) / len(y_parent)
        info_gain = parent_entropy - (w_left * left_entropy + w_right * right_entropy)
        split_info = self.split_info(y_left, y_right, y_parent)
        return info_gain / split_info if split_info != 0 else 0
