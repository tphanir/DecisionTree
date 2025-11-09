from base.dt_base import DecisionTreeBase
import numpy as np

class DT_ChiSquare(DecisionTreeBase):
    """
    Decision Tree using Chi-Square (CHAID-like criterion)
    Chi² = Σ (Observed - Expected)² / Expected
    """

    def __init__(self):
        super().__init__("Chi-Square")

    def chi_square(self, y_left, y_right):
        total = len(y_left) + len(y_right)
        n_classes = max(int(y_left.max()), int(y_right.max())) + 1
        obs_left = np.bincount(y_left.astype(int), minlength=n_classes)
        obs_right = np.bincount(y_right.astype(int), minlength=n_classes)
        total_obs = obs_left + obs_right

        expected_left = total_obs * (len(y_left) / total)
        expected_right = total_obs * (len(y_right) / total)

        chi_left = np.sum((obs_left - expected_left) ** 2 / (expected_left + 1e-9))
        chi_right = np.sum((obs_right - expected_right) ** 2 / (expected_right + 1e-9))
        return chi_left + chi_right

    def criterion(self, y_left, y_right, y_parent):
        return self.chi_square(y_left, y_right)
