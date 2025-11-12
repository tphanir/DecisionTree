from base.dt_base import DecisionTreeBase
import numpy as np

class DT_Hellinger(DecisionTreeBase):
    """
    Decision Tree using Hellinger Distance
    H(P, Q) = sqrt(1 - Σ sqrt(p_i * q_i))
    Gain = 1 - HellingerDistance
    """

    def __init__(self):
        super().__init__("Hellinger Distance")

    def hellinger(self, y_left, y_right):
        n_classes = max(int(y_left.max()), int(y_right.max())) + 1
        p = np.bincount(y_left.astype(int), minlength=n_classes) / len(y_left)
        q = np.bincount(y_right.astype(int), minlength=n_classes) / len(y_right)
        return np.sqrt(1 - np.sum(np.sqrt(p * q)))

    def criterion(self, y_left, y_right, y_parent):
        # smaller Hellinger = better similarity, so we invert it
        return 1 - self.hellinger(y_left, y_right)
