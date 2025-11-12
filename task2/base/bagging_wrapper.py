import numpy as np
from sklearn.utils import resample

import numpy as np
from collections import Counter
from copy import deepcopy

class BaggingWrapper:
    """
    A wrapper class to implement Bagging (Bootstrap Aggregating)
    for any given base decision tree estimator.
    """

    def __init__(self, base_estimator, n_estimators=100):
        """
        Initializes the Bagging wrapper.
        
        Args:
            base_estimator: An instance of your DecisionTreeBase class 
                            (e.g., DT_Entropy(), DT_Gini()).
            n_estimators (int): The number of trees (base estimators) to train.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        
        # Use the name of the base estimator for reporting
        self.name = f"Bagged ({self.base_estimator.name})"

    def _bootstrap_sample(self, X, y):
        """
        Creates a bootstrap sample (sampling with replacement).
        """
        n_samples = X.shape[0]
        # Generate random indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Fits 'n_estimators' copies of the base estimator on
        different bootstrap samples of the training data.
        """
        self.estimators = []
        for _ in range(self.n_estimators):
            # Create a bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Create a deep copy of the base estimator
            estimator = deepcopy(self.base_estimator)
            
            # Fit the estimator on the sample
            estimator.fit(X_sample, y_sample)
            
            # Store the trained estimator
            self.estimators.append(estimator)

    def _predict_row(self, row):
        """
        Gets a prediction for a single row from all estimators
        and returns the majority vote.
        """
        # Get predictions from all estimators
        predictions = [est._predict_row(row, est.tree) for est in self.estimators]
        
        # Return the most common prediction (majority vote)
        return Counter(predictions).most_common(1)[0][0]

    def predict(self, X):
        """
        Predicts the class for each sample in X using majority voting.
        """
        return np.array([self._predict_row(row) for row in X])
