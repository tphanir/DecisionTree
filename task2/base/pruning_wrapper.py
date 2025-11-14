import numpy as np
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import train_test_split

class PruningWrapper:
    """
    A wrapper class to implement Reduced Error Pruning (REP)
    for any given base decision tree estimator.
    
    This wrapper will:
    1. Internally split the training data into a sub-train and validation set.
    2. Grow a full (deep) tree on the sub-train set.
    3. Recursively prune the tree from the bottom up, checking accuracy
       against the validation set.
    """

    def __init__(self, base_estimator, validation_split=0.25, random_state=42):
        """
        Initializes the Pruning wrapper.
        
        Args:
            base_estimator: An instance of your DecisionTreeBase class.
            validation_split (float): The proportion of training data to
                                      hold out for validation.
        """
        self.base_estimator = base_estimator
        self.validation_split = validation_split
        self.random_state = random_state
        
        # This will hold the trained, high-depth estimator
        self.estimator_ = None 
        # This will hold the final, pruned tree structure
        self.tree_ = None 
        
        # --- FIX 1 ---
        # Add 'self.tree' for compatibility with BaggingWrapper
        self.tree = None 
        
        self.name = f"Pruned ({self.base_estimator.name})"

    def fit(self, X, y):
        """
        Fits the base estimator on a subset of the data and then
        prunes it using a validation set.
        """
        # 1. Split the *training* data into a sub-train and validation set
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X, y, 
            test_size=self.validation_split, 
            random_state=self.random_state
        )
        
        # Handle cases where the split is unusable (e.g., too few samples)
        if len(y_train_sub) == 0 or len(y_val) == 0:
            # Cannot prune, just fit on all data with default params
            self.estimator_ = deepcopy(self.base_estimator)
            self.estimator_.fit(X, y)
            self.tree_ = self.estimator_.tree
            
            # --- FIX 2 ---
            self.tree = self.tree_ # Assign to 'self.tree'
            return

        # 2. Grow a full (deep) tree on the sub-train set
        self.estimator_ = deepcopy(self.base_estimator)
        # Override default depth to grow a deep tree for pruning
        self.estimator_.max_depth = 99 
        self.estimator_.min_samples_split = 2 
        
        self.estimator_.fit(X_train_sub, y_train_sub)
        
        # 3. Start the recursive pruning process
        # The pruning happens in-place on a copy of the tree
        self.tree_ = deepcopy(self.estimator_.tree)
        
        self._prune_recursive(
            self.tree_, X_val, y_val, X_train_sub, y_train_sub
        )
        
        # --- FIX 3 ---
        self.tree = self.tree_ # Assign the final pruned tree to 'self.tree'

    def _prune_recursive(self, node, X_val, y_val, X_train, y_train):
        """
        Recursively prunes a tree.
        This function modifies the 'node' in place (or returns a new leaf).
        """
        # 1. Base Case: If node is a leaf, it cannot be pruned
        if not isinstance(node, dict):
            return node

        feature, threshold = list(node.keys())[0]
        subtree = node[(feature, threshold)]

        # 2. Filter data for children based on the split
        # We need data for both validation (for checking) and train (for making new leaves)
        left_mask_val = X_val[:, feature] <= threshold
        right_mask_val = X_val[:, feature] > threshold
        
        left_mask_train = X_train[:, feature] <= threshold
        right_mask_train = X_train[:, feature] > threshold

        # 3. Recurse on children *first* (bottom-up pruning)
        # Only recurse if the child has validation samples
        if np.any(left_mask_val):
            subtree['left'] = self._prune_recursive(
                subtree['left'], 
                X_val[left_mask_val], y_val[left_mask_val],
                X_train[left_mask_train], y_train[left_mask_train]
            )
        
        if np.any(right_mask_val):
            subtree['right'] = self._prune_recursive(
                subtree['right'], 
                X_val[right_mask_val], y_val[right_mask_val],
                X_train[right_mask_train], y_train[right_mask_train]
            )

        # 4. Pruning logic (Post-recursion)
        # Now that children are (possibly) pruned, check if we
        # should prune *this* node.
        
        # Accuracy of the current node (as a subtree) on validation data
        acc_subtree = self._calculate_accuracy(X_val, y_val, node)
        
        # Accuracy if we prune this node into a leaf
        # The leaf value is the majority class of the *training* data at this node
        if len(y_train) == 0:
             # This can happen if min_samples_split was hit
             # and no training data is left. Cannot prune.
            return node
            
        leaf_value = Counter(y_train).most_common(1)[0][0]
        
        # Calculate accuracy for this leaf value on the validation set
        if len(y_val) > 0:
            acc_leaf = np.mean(np.full(len(y_val), leaf_value) == y_val)
        else:
            acc_leaf = 0.0 # No validation data, don't prune

        # 5. Compare
        if acc_leaf >= acc_subtree:
            return leaf_value  # Prune the node
        else:
            return node  # Keep the subtree

    def _calculate_accuracy(self, X, y, tree):
        """Helper to get accuracy of a given tree on (X, y)"""
        if len(y) == 0:
            return 0.0
        
        # Use the estimator's _predict_row method
        predictions = np.array(
            [self.estimator_._predict_row(row, tree) for row in X]
        )
        return np.mean(predictions == y)

    def predict(self, X):
        """
        Predicts the class for each sample in X using the pruned tree.
        """
        if self.tree_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        
        # Use the base estimator's predict logic, but with the pruned tree
        return np.array(
            [self.estimator_._predict_row(row, self.tree_) for row in X]
        )

    def _predict_row(self, row, tree):
        """
         Pass-through method called by the BaggingWrapper.
        It passes the prediction request to the base estimator's
        _predict_row method, using the provided tree.
        """
        # The 'tree' argument is passed in by BaggingWrapper (it's self.tree)
        if tree is None:
            raise ValueError("Estimator not fitted. Call fit() first.")

        # Use the base estimator's traversal logic with the tree
        # that BaggingWrapper passed to us.
        return self.estimator_._predict_row(row, tree)