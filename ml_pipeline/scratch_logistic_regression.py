"""
Scratch Logistic Regression implementation using pure NumPy.
Strictly adheres to project SOP:
1. Weight initialization
2. Bias (intercept)
3. Linear combination: z = X @ weights + bias
4. Sigmoid function: sigma(z) = 1 / (1 + exp(-z))
5. Binary cross-entropy loss: L = -1/m * sum(y*log(p) + (1-y)*log(1-p))
6. Gradient calculation: dW = 1/m * X.T @ (p - y), db = 1/m * sum(p - y)
7. Gradient descent optimization
8. Weight updates: w := w - lr * dW, b := b - lr * db
9. Iteration/epoch handling with loss tracking
10. Probability prediction
11. Class prediction (with configurable decision threshold)
"""

import numpy as np


class ScratchLogisticRegression:
    """
    Binary Logistic Regression classifier implemented from scratch using NumPy.
    No scikit-learn or external ML library used internally.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_epochs: int = 1000,
        batch_size: int = 4096,
        tolerance: float = 1e-6,
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.verbose = verbose
        self.random_state = random_state

        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.feature_names = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation with numerical clipping to prevent overflow/underflow.
        """
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with epsilon clipping.
        """
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Fit the model using Mini-Batch / Full-Batch Gradient Descent.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        # 1. Weight initialization (small random or zeros)
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        y_float = y.astype(np.float64)
        use_batch = self.batch_size is not None and self.batch_size < n_samples

        for epoch in range(self.n_epochs):
            if use_batch:
                # Shuffle indices for mini-batch gradient descent
                indices = rng.permutation(n_samples)
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start_idx : start_idx + self.batch_size]
                    X_b = X[batch_idx]
                    y_b = y_float[batch_idx]
                    m_b = len(X_b)

                    # Linear combination & Sigmoid
                    z_b = np.dot(X_b, self.weights) + self.bias
                    p_b = self.sigmoid(z_b)

                    # Gradient calculation
                    dz_b = p_b - y_b
                    dw = np.dot(X_b.T, dz_b) / m_b
                    db = np.sum(dz_b) / m_b

                    # Weight updates
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            else:
                # Full-batch gradient descent
                z = np.dot(X, self.weights) + self.bias
                p = self.sigmoid(z)

                dz = p - y_float
                dw = np.dot(X.T, dz) / n_samples
                db = np.sum(dz) / n_samples

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Epoch loss monitoring
            z_full = np.dot(X, self.weights) + self.bias
            p_full = self.sigmoid(z_full)
            epoch_loss = self.binary_cross_entropy(y_float, p_full)
            self.loss_history.append(epoch_loss)

            if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                print(f"[Scratch LR] Epoch {epoch:4d}/{self.n_epochs}: Loss = {epoch_loss:.6f}")

            # Early stopping check
            if len(self.loss_history) > 1:
                loss_diff = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_diff < self.tolerance:
                    if self.verbose:
                        print(f"[Scratch LR] Converged at epoch {epoch} (loss delta {loss_diff:.2e} < {self.tolerance})")
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Returns shape (n_samples, 2) to match standard scikit-learn API:
        [:, 0] = P(y=0), [:, 1] = P(y=1)
        """
        z = np.dot(X, self.weights) + self.bias
        p1 = self.sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels using the specified decision threshold.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    def get_coefficients(self) -> dict:
        """
        Return the learned coefficients and intercept.
        """
        coef_dict = {name: float(w) for name, w in zip(self.feature_names, self.weights)}
        coef_dict["Intercept"] = float(self.bias)
        return coef_dict
