import numpy as np
from abc import ABC, abstractmethod
from collections import Counter

class DecisionTreeBase(ABC):
    """
    Abstract Base for Decision Trees with custom split criteria.
    """

    def __init__(self, name, max_depth=5, min_samples_split=2):
        self.name = name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    @abstractmethod
    def criterion(self, y_left, y_right, y_parent):
        """Calculate impurity or gain."""
        pass

    def fit(self, X, y):
        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.tree = self._build_tree(data, depth=0)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row, tree):
        if isinstance(tree, dict):
            feature, threshold = list(tree.keys())[0]
            if row[feature] <= threshold:
                return self._predict_row(row, tree[(feature, threshold)]['left'])
            else:
                return self._predict_row(row, tree[(feature, threshold)]['right'])
        else:
            return tree  # leaf node

    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        n_samples, n_features = X.shape

        # Stopping condition
        if len(set(y)) == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        best_gain = -np.inf
        best_split = None

        # Try all possible splits
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            for val in values:
                left = data[X[:, feature_idx] <= val]
                right = data[X[:, feature_idx] > val]
                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self.criterion(left[:, -1], right[:, -1], y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, val, left, right)

        # No valid split found
        if best_split is None:
            return Counter(y).most_common(1)[0][0]

        feature, threshold, left, right = best_split

        node = {(feature, threshold): {
            'left': self._build_tree(left, depth + 1),
            'right': self._build_tree(right, depth + 1)
        }}
        return node

