from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Every method in methods/ must inherit from this class."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model. Should store training history internally."""
        ...

    @abstractmethod
    def predict(self, X):
        """Return predictions for inputs X."""
        ...

    @abstractmethod
    def save(self, path):
        """Persist the trained model to disk."""
        ...

    @abstractmethod
    def load(self, path):
        """Load a trained model from disk."""
        ...